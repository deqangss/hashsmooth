import itertools

import tqdm
import gmpy2
from fast_poibin import PoiBin
import numpy as np

from hashsmooth import LSHTransformer
from hashsmooth.classifier_template import BasicClassifier
from hashsmooth.utils_hash import get_position, lower_confidence_interval, upper_confidence_interval

EPS = 1e-6


class HashSmooth(object):
    """
    LSH based smoother
    """
    ABSTAIN = -1

    def __init__(self, base_classifier: BasicClassifier,
                 num_of_classes: int,
                 hash_methods: list,
                 n_subfeatures: list,
                 k_subhashcodes: list,
                 max_radii: list,
                 n_grids: list,
                 default_mode=True,
                 ):
        """
        initialization
        :param base_classifier: basic block for establishing a smoothing classifier
        :param num_of_classes: number of the classes
        :param hash_methods: a list of LSH schemes
        :param n_subfeatures: a list of number of heterogeneous features
        :param k_subhashcodes: number of hash functions is utilized to compose the transformations
        :param max_radii: list of maximum radii
        :param n_grids: number of grids to split the radius
        :param default_mode: use the estimated confidence of the second class or not
        """
        self.base_classifier = base_classifier
        self.num_of_classes = num_of_classes
        if isinstance(hash_methods, LSHTransformer):
            hash_methods = [hash_methods]
        assert len(hash_methods) > 0 & all([isinstance(hashmethod, LSHTransformer) for hashmethod in hash_methods])
        self.randomized_trans = hash_methods
        if isinstance(n_subfeatures, int):
            n_subfeatures = [n_subfeatures]
        assert len(hash_methods) == len(n_subfeatures) & all(
            [isinstance(n_feature, int) for n_feature in n_subfeatures])
        assert all([n_subfeature >= 0 for n_subfeature in n_subfeatures])
        self.n_subfeatures = n_subfeatures
        if isinstance(k_subhashcodes, int):
            k_subhashcodes = [k_subhashcodes]
        assert len(n_subfeatures) == len(k_subhashcodes) & all(
            [isinstance(k_hashcode, int) for k_hashcode in k_subhashcodes])
        assert all([k_hashcode >= 0 for k_hashcode in k_subhashcodes])
        self.k_subhashcodes = k_subhashcodes
        assert len(max_radii) == len(self.k_subhashcodes)
        self.max_radii = max_radii
        assert len(n_grids) == len(max_radii)
        self.n_grids = n_grids

        self.default_mode = default_mode

    def certify(self, x: np.ndarray, n_selection: int, n_estimation: int, alpha: float) -> (int, float):
        """
        Monte Carlo algorithm for certification w.r.t. the metric space corresponding to LSH schemes, where the metric
        space relates to the input space and a measurement (though our method support multiple ones simotaniously).
        Moreover, the certification means that the perturbed input x cannot change the prediction of the randomized
        classifier with the probability at 1 - \alpha, as long as the degree of perturbations is within the radius
        characterized by the measurement.
        :param x: a batch of input vectors with shape number_of_instance x number_of_dimension
        :param n_selection: number of Monte Carlo samples for label selection
        :param n_estimation: number of Monte Carlo samples for probability estimation
        :param alpha: confidence is 1 - \alpha
        :return: (pred_class, cert_radius). Owing to the abstaining operation, (pred_class=-1, cert_radius=0) is used
        """
        if len(x.shape) == 1:
            x = x[np.newaxis, ...]
        assert len(x.shape) == 2 and n_selection > 0 and n_estimation > 0 and 0 < alpha < 1

        # first round sampling to predict the label
        n_data = x.shape[0]
        counts_selection = self._sample_lsh_funcs(x, n_data, n_selection)
        c_pred = counts_selection.argmax(axis=1)

        # second round sampling to estimate the probability
        counts_estimation = self._sample_lsh_funcs(x, n_data, n_estimation)
        n_targeted = counts_estimation[range(n_data), c_pred]

        # given the estimated probability, we calculate the radius
        prob_underlined = lower_confidence_interval(n_targeted, n_estimation, alpha)
        if self.default_mode:
            abstain_indicator = prob_underlined <= 0.5
            c_pred[abstain_indicator] = HashSmooth.ABSTAIN
            radii = np.ones_like(c_pred, dtype=object)
            radii[abstain_indicator] = 0.
            radii[~abstain_indicator] = self._calc_radius(prob_underlined[~abstain_indicator], 0.5)
        else:
            c_pred_runnerup = counts_selection.argsort()[:, -2]
            n_targeted_runnerup = counts_estimation[range(n_data), c_pred_runnerup]
            prob_upperlined = upper_confidence_interval(n_targeted_runnerup, c_pred_runnerup, alpha)
            abstain_indicator = prob_underlined <= prob_upperlined
            c_pred[abstain_indicator] = HashSmooth.ABSTAIN
            radii = np.ones_like(c_pred, dtype=object)
            radii[abstain_indicator] = 0.
            radii[abstain_indicator] = self._calc_radius(prob_underlined[abstain_indicator], prob_upperlined)
        return c_pred, radii

    def _sample_lsh_funcs(self, x: np.ndarray, n_data: int, n_samples: int):
        """
        randomly sample $n_samples$ examples and conduct prediction
        :param x: input
        :param n_samples: number of Monte Carlo samples
        """
        self.base_classifier.eval()
        counts = np.zeros((n_data, self.num_of_classes), dtype=int)
        for _ in tqdm.tqdm(range(n_samples)):
            data_lsh_codes = self._transform_wrapper(x)
            preds = self.base_classifier.predict(data_lsh_codes)
            preds_encoded = np.eye(self.num_of_classes, dtype=int)[preds]  # one-hot encodings
            counts += preds_encoded
        return counts

    def _transform_wrapper(self, input_vectors: np.ndarray) -> np.ndarray:
        """
        conduct the input transformation
        """
        assert len(input_vectors.shape) == 2, "Only support 2D array: batch_size x dimension.\n"
        assert np.sum(self.n_subfeatures) == input_vectors.shape[1], \
            f"Inconsistent dimension: {np.sum(self.n_subfeatures)} vs. {input_vectors.shape[1]}.\n"
        slice_index = 0
        input_transformed = []
        for hash_tran, n_subfeature in zip(self.randomized_trans, self.n_subfeatures):
            next_slice_index = slice_index + n_subfeature
            sub_input_vectors = input_vectors[:, slice_index: next_slice_index]
            input_transformed.append(hash_tran.transform(sub_input_vectors))
            slice_index = next_slice_index
        return np.hstack(input_transformed)

    def _calc_radius(self, probas, second_probas):
        """
        calculate the robustness radius
        :param probas: estimated confidences
        :param second_probas: getting thresholds for computing radii (i.e., probas are greater than second probas item-wisely)
        """
        batch_size = len(probas)
        threshold = (probas - second_probas) / 2.
        bound_mesh, radii_mesh = self._get_radius_grid(probas)
        # select the radius corresponding to the estimated bound smaller than the threshold
        pos_sel = np.apply_along_axis(np.searchsorted, 1, bound_mesh, threshold)
        radii = radii_mesh[range(batch_size), pos_sel]
        return radii

    def _get_radius_grid(self, probas: np.ndarray, regions=None):
        """
        obtain all bounds regarding the radius from 1 to the maximum radius
        :param probas: batch of probabilities
        :param regions: regions for calculating bounds. We can provide these bounds beforehand. If not, we will compute them
        """
        if regions is None:
            max_proba = np.max(probas)
            # get maximum radius for each sub-groud of features
            max_radii = []
            for i, k_hashcode in enumerate(self.k_subhashcodes):
                max_radius = self._get_max_radius(max_proba, k_hashcode,
                                                  min_radius=0.,
                                                  max_radius=self.max_radii[i],
                                                  n_grid=self.n_grids[i]
                                                  )
                max_radii.append(max_radius)
            regions = {}
            radii_splitting = []
            if len(max_radii) == 1:
                radii_splitting.append(
                    np.linspace(0, max_radii[0], self.n_grids[0] + 1))  # int(round(max_radii[0] * self.n_subfeatures[0])) + 1)
            elif not len(max_radii) <= 1:
                # position product
                for i, max_r in enumerate(max_radii):
                    radii_splitting.append(np.linspace(0, max_r, self.n_grids[i] + 1))  # int(round(max_r * self.n_subfeatures[i])) + 1)
            else:
                raise ValueError
            radii_mesh = list(itertools.product(*radii_splitting))
            for r_pos in radii_mesh:
                regions[r_pos] = self._calc_regions(self.k_subhashcodes, list(r_pos))

        # calculate bound for each radius
        shape = [len(e) for e in radii_splitting]
        radii4bounds = np.zeros([len(probas)] + shape)
        bounds4radii = np.zeros([len(probas)] + shape, dtype=tuple)
        for _radii, region in regions.items():
            idx_folded = radii_mesh.index(_radii)
            idx = get_position(idx_folded, shape)
            if np.sum(_radii) == 0:
                radii4bounds[idx] = 0.
                bounds4radii[idx] = _radii
            else:
                radii4bounds[idx] = self._calc_bound_batch(regions=region, confidence_ests=probas)
                bounds4radii[idx] = _radii
        return radii4bounds, bounds4radii

    def _get_max_radius(self, max_proba, k_hashcode, min_radius=0., max_radius=0.01, n_grid=100) -> float:
        """
        Search the maximum radius
        :param min_proba: the minimum probability to estimate the radius
        :param max_proba: the maximum probability to estimate the radius
        :param k_hashcode: number of hash codes
        :param max_radius: initial radius
        :param n_grid: decide the step in searching process
        """
        radii_steps = np.linspace(min_radius, max_radius, n_grid + 1)
        radius, lower_idx, upper_idx = 0., 0, n_grid
        while lower_idx < upper_idx:
            curr_idx = lower_idx + (upper_idx - lower_idx) // 2
            radius = radii_steps[curr_idx]
            _regions = self._calc_regions(k_hashcode, radius)
            _bound = self._calc_bound(_regions, max_proba)
            if _bound <= (max_proba - 0.5) / 2.0:
                if lower_idx == curr_idx:
                    break
                lower_idx = curr_idx
            else:
                upper_dix = curr_idx

        if abs(radius - max_radius) <= max_radius / n_grid:
            return self._get_max_radius(max_proba, k_hashcode, max_radius, 2 * max_radius, n_grid)
        else:
            return radius

    def _calc_regions(self, k_hashcodes: [list, int], curr_radii: [list, int]):
        """
        Construct regions along with (\Pr(y|x), \Pr(x|x'), \Pr(y|x) - \Pr(x|x'))
        :param k_hashcodes: a list of number of hash codes
        :param curr_radii: a list of radii for obtaining the collision probabilities
        :return: regions
        """
        if isinstance(k_hashcodes, int):
            k_hashcodes = [k_hashcodes]
        if isinstance(curr_radii, int):
            curr_radii = [curr_radii]
        assert np.all(np.array(k_hashcodes) >= 0)
        assert np.all(np.array(curr_radii) >= 0)
        assert len(curr_radii) == len(k_hashcodes)

        k_hashcodes = np.array(k_hashcodes, dtype=int)
        total_K = np.sum(k_hashcodes)
        # \Pr(y|x)
        px = np.zeros((total_K + 1,), dtype=float)
        px[0] = 1.

        # \Pr(y|x')
        # unfold probability element-wisely
        with gmpy2.context(precision=1000):
            # obtain collision probability
            curr_p_k = []
            for i, r in enumerate(curr_radii):
                curr_p_k.append(self.randomized_trans[i].get_collision_prob(r))

            probas = np.ones((total_K,), dtype=float) * -1
            cursor = 0
            for i, n in enumerate(k_hashcodes):
                probas[cursor: cursor + n] = curr_p_k[i]
                cursor += n
            pb = PoiBin(probas)
            px_prime = pb.pmf
            prob_diff = px - px_prime
            return np.column_stack((px, px_prime, prob_diff))

    def _calc_bound(self, regions: np.ndarray, confidence_est: float, duality=True) -> float:
        """
        calculate a bound w.r.t., the given the probability
        :param regions: an array of (px, px_prime, p_diff)
        :param confidence_est: estimated confidence
        :param duality: utilize the dual results
        :return: bound
        """
        assert regions.shape[1] == 3
        assert 0. <= confidence_est <= 1.
        # sort the regions
        sorted_regions = sorted(
            list(regions), key=lambda a: a[2], reverse=True)
        # we utilize the dual result
        if duality:
            return confidence_est * (1. - sorted_regions[0][1] / sorted_regions[0][0])
        else:
            p_clean, p_adver = 0., 0.
            for i, (px, px_prime, _) in enumerate(sorted_regions):
                if p_clean + px >= confidence_est:
                    p_clean_ast = px
                    p_adver_ast = px_prime
                    break
                else:
                    p_clean += px
                    p_adver += px_prime
            return (confidence_est - p_adver) * (1 - (p_adver_ast / p_clean_ast))

    def _calc_bound_batch(self, regions: np.ndarray, confidence_ests: np.ndarray, duality=True, is_sort=True):
        """
        calculate bounds w.r.t., the given the probabilities
        :param regions: an array of (px, px_prime, p_diff)
        :param confidence_ests: estimated confidences
        :param duality: utilize the dual results
        :param is_sort: whether sort the regions
        :return: bounds
        """
        assert regions.shape[1] == 3
        assert np.all(0. <= confidence_ests <= 1.)
        if not is_sort:
            sorted_regions = sorted(
                list(regions), key=lambda a: a[2], reverse=True)
            regions = np.stack(sorted_regions)
        if duality:
            return confidence_ests * (1 - (regions[0][1] / regions[0][0]))
        else:
            regions_zero_ext = np.vstack((np.zeros((1, regions.shape[1])), regions))
            proba_cumsum = np.cumsum(regions_zero_ext[:, :2], axis=0)
            idx_ast = (proba_cumsum[:, 0][:, None] >= confidence_ests).argmax(axis=0)
            idx_ast -= 1  # set indices back
            proba_ast = proba_cumsum[idx_ast]
            p_clean = proba_ast[:, 0]
            p_adver = proba_ast[:, 1]
            left_part_values = confidence_ests - p_adver

            # get the extra probabilities
            is_extra = ((confidence_ests - p_clean) > 0) & (idx_ast + 1 < len(regions))
            assert np.all(is_extra)
            right_part_values = 1. - (regions[idx_ast + 1][1] / regions[idx_ast + 1][0])
            return left_part_values * right_part_values
