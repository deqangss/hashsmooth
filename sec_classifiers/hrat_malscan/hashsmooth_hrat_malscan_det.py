import numpy as np
import torch
import time
from statsmodels.stats.proportion import binom_test

from hashsmooth.core import HashSmooth
from sec_classifiers.hrat_malscan.hrat_malscan_det import MalScan


class HashSmooth4MalScan(HashSmooth):
    def __init__(self,
                 malscan_det: MalScan,
                 num_of_classes: int,
                 hash_methods: list,
                 n_subfeatures: list,
                 k_subhashcodes: list,
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
                            k_subhashcodes,
                            max_radii,
                            n_grids,
                            default_mode)

    def eval(self):
        pass

    def predict(self, x: (np.ndarray, torch.Tensor), n: int, alpha: float,
                adj_size: int, top_k=1, x_sensitive_dix=None,
                n_subfeatures=[], device='cpu', verbose=False):
        # hash-based transformations
        # with torch.no_grad():
        # if isinstance(x, np.ndarray):
        #     x = torch.from_numpy(x).to(device)
        counts = self.sample_lsh_funcs_a_point(x, n, adj_size, top_k, x_sensitive_dix,
                                               n_subfeatures, self.base_classifier.batch_size, device, verbose)
        # prediction
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, 0.5) > alpha:
            return HashSmooth.ABSTAIN
        else:
            return top2[0]

    def sample_lsh_funcs_a_point(self, x, n, adj_size: int, top_k=1, x_sensitive_dix=None,
                                 n_subfeatures=[], batch_size=64, device='cpu', verbose=False):
        assert n > 0
        self.base_classifier.eval()
        values = x[:, 2].astype(int)
        nonzero_idx = values.nonzero()[0].copy()
        if len(n_subfeatures) == 0:
            n_subfeatures = [len(nonzero_idx)]
        preds = []
        for idx in range(n // batch_size + 1):
            current_batch_size = min(batch_size, n)
            n -= current_batch_size
            if current_batch_size <= 0:
                break

            nonzero_idx_sel = self.transform_wrapper(np.tile(nonzero_idx.copy(), (current_batch_size, 1)),
                                                     n_subfeatures).squeeze()
            print('testtest: ', nonzero_idx_sel.shape)
            with torch.no_grad():
                # malscan_features = self.get_extra_feature2(x[:, :2], nonzero_idx_sel, x_sensitive_dix, adj_size, device)
                pred_batch = self.base_classifier.predict(x[nonzero_idx_sel],
                                                          adj_size,
                                                          top_k,
                                                          x_sensitive_dix,
                                                          device,
                                                          verbose)
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
        return np.bincount(np.concatenate(preds).squeeze(), minlength=self.num_of_classes)

    @staticmethod
    def get_extra_feature2(x_position: np.ndarray,
                           x_value: (np.ndarray, torch.Tensor),
                           x_sensitive_dix: np.ndarray,
                           adj_size: int,
                           device='cpu'
                           ) -> torch.Tensor:
        adj_dense = torch.sparse_coo_tensor(x_position.T,
                                            x_value,
                                            size=(adj_size, adj_size)
                                            ).float().to(device).to_dense()[None, ...]
        # else:
        #     adj_dense = x_batch if len(x_batch.shape) == 3 else x_batch[None, ...]
        degree_fea = MalScan._degree_centrality(adj_dense, x_sensitive_dix, adj_size)
        katz_fea = MalScan._katz_feature(adj_dense, x_sensitive_dix, adj_size)
        return torch.cat([degree_fea, katz_fea], -1).squeeze()
