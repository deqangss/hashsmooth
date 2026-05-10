import itertools
import multiprocessing
import warnings

import tqdm
import gmpy2
from fast_poibin import PoiBin
import math
import numpy as np
import torch

from hashsmooth import LSHTransformer
from hashsmooth.lsh_torch import LSHTransformerTorch
from hashsmooth.utils_hash import get_position, lower_confidence_interval, upper_confidence_interval, binom_test

EPS = 1e-6


class HashSmoothBase(object):
    """
    LSH based smoother. Esentially, the randomized smoother is a kind of input transformer
    """
    ABSTAIN = -1

    def __init__(self,
                 num_of_classes: int,
                 hash_method: LSHTransformer,
                 k_hashcode: int,
                 max_radius: float,
                 n_grid: int,
                 default_mode=True
                 ):
        """
        initialization
        :param num_of_classes: number of the classes
        :param hash_method: a LSH scheme
        :param k_hashcode: the number of hash functions is utilized to compose transformations
        :param max_radius: the maximum radius
        :param n_grid: number of grids to split the radius
        :param default_mode: use the estimated confidence of the second class or not
        """
        self.num_of_classes = num_of_classes
        if hash_method is None:
            warnings.warn("Initialization can be conducted in the method of 'transform_wrapper'.\n")
        else:
            assert isinstance(hash_method, (LSHTransformer, LSHTransformerTorch))
        self.hash_method = hash_method
        self.k_hashcode = k_hashcode

        if max_radius is None:
            warnings.warn("Initialization can be conducted in the method of '_calc_radius'.\n")

        self.max_radius = max_radius
        self.n_grid = n_grid
        self.default_mode = default_mode

    def _calc_radius(self, probas, second_probas, k_hashcode=None, max_radius=None, n_grid=None):
        """
        calculate the robustness radius
        :param probas: estimated confidences
        :param second_probas: getting thresholds for computing radii (i.e., probas are greater than second probas item-wisely)
        """
        batch_size = len(probas)
        if np.all((second_probas - 0.5)<= 1e-8): # binary classification
            threshold = probas - second_probas
        else:
            threshold = (probas - second_probas) / 2.
        bound_mesh, radii_mesh = self._get_radius_grid(probas, second_probas,None, k_hashcode, max_radius, n_grid)
        # select the radius corresponding to the estimated bound smaller than the threshold
        if len(bound_mesh.shape) == 2:
            pos_sel = np.diag(np.apply_along_axis(np.searchsorted, 1, bound_mesh, threshold))
            pos_sel = np.maximum(0, pos_sel - 1)
            radii = radii_mesh[range(batch_size), pos_sel]
        else:
            raise NotImplementedError
        return radii

    def _get_radius_grid(self, probas: np.ndarray, second_probas: np.ndarray, regions=None, k_hashcode=None, max_radius=None, n_grid=None):
        """
        obtain all bounds regarding the radius from 1 to the maximum radius
        :param probas: batch of probabilities
        :param regions: regions for calculating bounds. We can provide these bounds beforehand. If not, we will compute them
        """
        # if regions is None: close it

        k_hashcode = self.k_hashcode if self.k_hashcode is not None else k_hashcode
        max_radius = self.max_radius if self.max_radius is not None else max_radius
        n_grid = self.n_grid if self.n_grid is not None else n_grid

        max_proba = np.max(probas)
        min_proba = np.min(second_probas) if second_probas is not None else 0.5
        # get maximum radius for each sub-group of features
        max_radius = self._get_max_radius(max_proba, min_proba,
                                          k_hashcode,
                                          min_radius=0.,
                                          max_radius=max_radius,
                                          n_grid=n_grid
                                          )
        if max_radius <= 0:
            warnings.warn("The maximum radius should be greater than 0.")
        regions = {}
        radii_splitting = []
        radii_splitting.append(
            np.linspace(0, max_radius, n_grid + 1))

        radii_mesh = list(itertools.product(*radii_splitting))
        for r_pos in radii_mesh:
            regions[r_pos] = self._calc_regions(k_hashcode, list(r_pos))

        # calculate bound for each radius
        shape = [len(e) for e in radii_splitting]
        radii4bounds = np.zeros([len(probas)] + shape)
        bounds4radii = np.empty([len(probas)] + shape, dtype=tuple)
        idx_range = np.arange(len(probas))
        for _radii, region in regions.items():
            idx_folded = radii_mesh.index(_radii)
            idx = get_position(idx_folded, shape)
            if np.sum(_radii) == 0:
                radii4bounds[(idx_range, *idx)] = 0.
                bounds4radii[(idx_range, *idx)] = sum(_radii)
            else:
                radii4bounds[(idx_range, *idx)] = self._calc_bound_batch(regions=region, confidence_ests=probas)
                bounds4radii[(idx_range, *idx)] = sum(_radii)
        return radii4bounds, bounds4radii

    def _get_max_radius(self, max_proba, min_second_probas, k_hashcode, min_radius=0., max_radius=0.01, n_grid=100) -> float:
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
        if np.all(np.abs(min_second_probas - 0.5)<= 1e-8): # binary classification
            threshold_ = max_proba - min_second_probas
        else:
            threshold_ = (max_proba - min_second_probas) / 2.0
        while lower_idx <= upper_idx:
            curr_idx = lower_idx + (upper_idx - lower_idx + 1) // 2
            radius = radii_steps[curr_idx]
            _regions = self._calc_regions(k_hashcode, radius)
            _bound = self._calc_bound(_regions, max_proba)
            if _bound <= threshold_:
                if lower_idx == curr_idx:
                    break
                lower_idx = curr_idx
            else:
                if upper_idx == curr_idx:
                    break
                upper_idx = curr_idx
        if -1 * max_radius / n_grid < (radius - max_radius) < max_radius / n_grid:
            return self._get_max_radius(max_proba, min_second_probas, k_hashcode, max_radius, 2 * max_radius, n_grid)
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
        if isinstance(curr_radii, (int, float)):
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
                curr_p_k.append(self.hash_method.get_collision_prob(r))
            probas = np.ones((total_K,), dtype=float) * (-1)
            cursor = 0
            for i, n in enumerate(k_hashcodes):
                probas[cursor: cursor + n] = curr_p_k[i]
                cursor += n
            pb = PoiBin(probas)
            px_prime = pb.pmf[::-1]
            prob_diff = px - px_prime
            return np.column_stack((px, px_prime, prob_diff))

    def _calc_bound(self, regions: np.ndarray, confidence_est: float, duality=False) -> float:
        """
        calculate a bound w.r.t., the given the probability
        :param regions: an array of (px, px_prime, p_diff)
        :param confidence_est: estimated confidence
        :param duality: utilize the dual results
        :return: bound
        """
        assert regions.shape[1] == 3
        assert 0. <= confidence_est <= 1.
        with gmpy2.context(precision=1000):
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
        assert np.all(0. <= confidence_ests) and np.all(confidence_ests <= 1.)
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
            is_extra = ((confidence_ests - p_clean) > 0) & (idx_ast + 1 < len(regions_zero_ext))
            assert np.all(is_extra)
            right_part_values = 1. - (regions_zero_ext[idx_ast + 1][1] / regions_zero_ext[idx_ast + 1][0])
            return left_part_values * right_part_values
