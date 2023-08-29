import time
import numpy as np
import torch
import math
from statsmodels.stats.proportion import binom_test

from hashsmooth.core import HashSmooth
from sec_classifiers.hrat_malscan.hrat_malscan_det import MalScan
from hashsmooth.utils_hash import lower_confidence_interval, upper_confidence_interval


class HashSmooth4MalScan(HashSmooth):
    def __init__(self,
                 malscan_det: MalScan,
                 num_of_classes: int,
                 hash_methods: list,
                 n_subfeatures: list,
                 k_hashcode: int,
                 max_k,
                 max_radii: list,
                 n_grids: list,
                 default_mode: bool,
                 ):
        """
        Customized HashSmooth w.r.t. Malscan
        """
        HashSmooth.__init__(self,
                            malscan_det,
                            num_of_classes,
                            hash_methods,
                            n_subfeatures,
                            k_hashcode,
                            max_k,
                            max_radii,
                            n_grids,
                            default_mode)

    def eval(self):
        pass

    def certify(self, x: np.ndarray, label: int, adj_size: int, n_selection: int, n_estimation: int, alpha: float,
                hash_methods=[], n_subfeatures=[], k_per_instance=0, max_radii=[], n_grids=[],
                top_k=1, x_sensitive_dix=None, device='cpu', verbose=False
                ) -> (int, float):
        assert n_selection > 0 and n_estimation > 0 and 0 < alpha < 1

        # assert len(hash_methods) == len(n_subfeatures) == len(max_radii) == len(n_grids)

        # first round sampling to predict the label
        counts_selection, k_subhashcodes = self.sample_lsh_funcs_a_point(x, adj_size,
                                                                         n_selection,
                                                                         self.base_classifier.batch_size,
                                                                         n_subfeatures,
                                                                         k_per_instance,
                                                                         top_k, x_sensitive_dix, device, verbose)
        c_pred = counts_selection.argmax()
        if c_pred != label:
            return HashSmooth.ABSTAIN

        # second round sampling to estimate the probability
        counts_estimation, _1 = self.sample_lsh_funcs_a_point(x, adj_size,
                                                              n_estimation,
                                                              self.base_classifier.batch_size,
                                                              n_subfeatures,
                                                              k_per_instance,
                                                              top_k, x_sensitive_dix, device, verbose)
        n_targeted = counts_estimation[c_pred]

        # given the estimated probability, we calculate the radius
        prob_underlined = lower_confidence_interval(n_targeted, n_estimation, alpha)
        c_pred = np.array([c_pred])[None, ...]
        if self.default_mode:
            abstain_indicator = prob_underlined <= 0.5
            c_pred[abstain_indicator] = HashSmooth.ABSTAIN
            radii = np.ones_like(c_pred, dtype=object)
            radii[abstain_indicator] = 0.
            radii[~abstain_indicator] = self._calc_radius(prob_underlined[~abstain_indicator], 0.5,
                                                          k_subhashcodes,
                                                          max_radii,
                                                          n_grids
                                                          )
        else:
            c_pred_runnerup = counts_selection.argsort()[:, -2]
            n_targeted_runnerup = counts_estimation[c_pred_runnerup]
            prob_upperlined = upper_confidence_interval(n_targeted_runnerup, c_pred_runnerup, alpha)
            abstain_indicator = prob_underlined <= prob_upperlined
            c_pred[abstain_indicator] = HashSmooth.ABSTAIN
            radii = np.ones_like(c_pred, dtype=object)
            radii[abstain_indicator] = 0.
            radii[abstain_indicator] = self._calc_radius(prob_underlined[abstain_indicator], prob_upperlined,
                                                         k_subhashcodes,
                                                         max_radii,
                                                         n_grids
                                                         )
        return radii

    def predict(self, x: (np.ndarray, torch.Tensor), adj_size: int, n: int, n_subfeatures: list, k_per_instance: int,
                alpha: float,
                top_k=1, x_sensitive_dix=None, device='cpu', verbose=False):
        # hash-based transformations
        # with torch.no_grad():
        # if isinstance(x, np.ndarray):
        #     x = torch.from_numpy(x).to(device)
        counts, _1 = self.sample_lsh_funcs_a_point(x, adj_size, n, self.base_classifier.batch_size, n_subfeatures,
                                                   k_per_instance,
                                                   top_k, x_sensitive_dix, device, verbose)
        # prediction
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, 0.5) > alpha:
            return HashSmooth.ABSTAIN
        else:
            return top2[0]

    def sample_lsh_funcs_a_point(self, x, adj_size: int, n: int, batch_size=64, n_subfeatures=[], k_hashcode=16,
                                 top_k=1, x_sensitive_dix=None, device='cpu', verbose=False):
        assert n > 0
        self.base_classifier.eval()

        # debug:
        check_flag = x[:, 0] == x[:, 1]
        if np.any(check_flag):
            print(x[check_flag])
            exit(-1)

        values = x[:, 2].astype(int)
        nonzero_idx = values.nonzero()[0].copy()

        if len(n_subfeatures) == 0:
            n_subfeatures = [len(nonzero_idx)]
        if isinstance(k_hashcode, int) and k_hashcode > 1:
            k_hashcode = min(k_hashcode, self.max_k)
            k_subhashcodes = self.get_num_subhashcodes(n_subfeatures, k_hashcode)
        elif isinstance(k_hashcode, float) and 0. < k_hashcode <= 1.:
            k_hashcode_ = min(math.ceil(sum(n_subfeatures) * k_hashcode), self.max_k)
            k_subhashcodes = self.get_num_subhashcodes(n_subfeatures, k_hashcode_)
        else:
            k_subhashcodes = self.k_subhashcodes

        preds = []
        for idx in range(n // batch_size + 1):
            current_batch_size = min(batch_size, n)
            n -= current_batch_size
            if current_batch_size <= 0:
                break

            with torch.no_grad():
                nonzero_idx_sel = self.transform_wrapper(np.tile(nonzero_idx.copy(), (current_batch_size, 1)),
                                                         n_subfeatures, k_subhashcodes).squeeze()
                malscan_features = self.base_classifier.get_extra_feature_sp(x[nonzero_idx_sel],
                                                                             x_sensitive_dix,
                                                                             adj_size,
                                                                             device)
                pred_batch = self.base_classifier.predict(malscan_features,
                                                          adj_size,
                                                          top_k,
                                                          x_sensitive_dix=None,
                                                          device=device,
                                                          verbose=verbose)
            # x[:, 2] = 0
            # x[:, 2][nonzero_idx_sel] = values[nonzero_idx_sel]
            # pred = self.base_classifier.predict(x,
            #                                     adj_size,
            #                                     top_k,
            #                                     x_sensitive_dix,
            #                                     device,
            #                                     verbose
            #                                     )
            # assert isinstance(pred, np.ndarray), "Expected numpy array, but got {}.\n".format(type(pred))
            preds.append(pred_batch)
        return np.bincount(np.concatenate(preds).squeeze(), minlength=self.num_of_classes), k_subhashcodes
