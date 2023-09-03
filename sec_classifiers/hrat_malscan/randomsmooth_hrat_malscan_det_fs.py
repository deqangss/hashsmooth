import numpy as np
import torch
import time
import math
from statsmodels.stats.proportion import proportion_confint, binom_test

from hashsmooth.core import BasicClassifier
from sec_classifiers.hrat_malscan.hrat_malscan_det import MalScan
from hashsmooth.utils_hash import lower_confidence_interval


class RandomTransformer(object):
    def __init__(self, keep_per_image, reuse_noise=False, seed=0):
        """
        codes are came from https://github.com/alevine0/randomizedAblation
        """
        self.keep_per_image = keep_per_image
        self.reuse_noise = reuse_noise
        self.seed = seed

    def transform(self, batch: torch.Tensor, keep_per_image: int) -> torch.tensor:
        """
        the method warks as the method of 'random_mask_batch_one_sample' in randmozedAblation repository
        :return: transfored input
        """
        self.keep_per_image = keep_per_image if keep_per_image > 0 else self.keep_per_image
        flat = batch.reshape(batch.shape[0], -1)
        out_c1 = torch.zeros(flat.shape, dtype=flat.dtype).cuda()
        out_c2 = torch.zeros(flat.shape, dtype=flat.dtype).cuda()

        if (self.reuse_noise):
            ones = torch.ones(flat.shape[1]).cuda()
            idx = torch.multinomial(ones, self.keep_per_image)
            # idx = np.random.choice(flat.shape[1], self.keep_per_image, replace=False)
            out_c1[:, idx] = flat[:, idx]
            # out_c2[:, idx] = 1. - flat[:, idx]
        else:
            idx = self._batch_choose(flat.shape[1], self.keep_per_image, flat.shape[0])
            idx_range = torch.arange(idx.shape[0]).cuda().unsqueeze(0).t()
            out_c1[(idx_range, idx)] = flat[(idx_range, idx)]
            # out_c2[(idx_range, idx)] = 1. - flat[(idx_range, idx)]
            # print(flat[0])
            # print(idx[0])
            # print(out_c1[0])
            # print(out_c2[0])
        # out = torch.stack([out_c1.reshape(batch.shape), out_c2.reshape(batch.shape)], dim=1).squeeze(2)
        out = out_c1.reshape(batch.shape)
        return out

    @staticmethod
    def _batch_choose(n, k, batches):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        out = torch.zeros((batches, k), dtype=torch.long).cuda()
        for i in range(k):
            out[:, i] = torch.randint(0, n - i, (batches,))
            if (i != 0):
                last_boost = torch.zeros(batches, dtype=torch.long).cuda()
                boost = (out[:, :i] <= (out[:, i] + last_boost).unsqueeze(0).t()).sum(dim=1)
                while (boost.eq(last_boost).sum() != batches):
                    last_boost = boost
                    boost = (out[:, :i] <= (out[:, i] + last_boost).unsqueeze(0).t()).sum(dim=1)
                out[:, i] += boost
        # end.record()
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))
        return out


class RandomSmooth(BasicClassifier):
    ABSTAIN = -1

    def __init__(self, base_classifier, number_of_classes, transform_method, max_k=1000, default_mode=True):
        self.base_classifier = base_classifier
        BasicClassifier.__init__(self, self.base_classifier.batch_size)
        self.number_of_classes = number_of_classes
        self.transform_method = transform_method
        self.default_mode = default_mode
        self.max_k = max_k

    def certify(self, x: np.ndarray, labels: np.ndarray, n_selection: int, n_estimation: int, k: (float, int),
                alpha: float,
                device='cpu'):
        if len(x.shape) == 1:
            x = x[np.newaxis, ...]
        guesses = self.sample_funcs(x, n_selection, k, device)
        bound_scores = self.sample_funcs(x, n_estimation, k, device)
        bound_selected_scores = torch.gather(bound_scores, 1, guesses.unsqueeze(1)).squeeze(0)
        if (len(bound_selected_scores.shape) == 1):
            bound_selected_scores = bound_selected_scores.unsqueeze(0)
        bound_selected_scores = self.lc_bound((bound_selected_scores * n_estimation).cpu().numpy(), n_estimation,
                                              alpha)
        radii = self.population_radius_for_majority(bound_selected_scores, x[0].nelement(),
                                                    self.transform_method.keep_per_image)
        radii[guesses != labels] = -1
        return radii

    def predict(self, x: np.ndarray, n_sampling: int, k_per_instance: (float, int), alpha: float, device='cpu'):
        self.base_classifier.eval()
        counts = self.sample_funcs(x, n_sampling, k_per_instance, device)
        # todo: need to batchize the following functionality
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return RandomSmooth.ABSTAIN
        else:
            return top2[0]

    def sample_funcs(self, batch: (np.ndarray, torch.Tensor), n_sampling, keep_per_image: int, device='cpu'):
        is_npy = False
        if isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch).to(device)
            is_npy = True
        expanded = batch.repeat_interleave(n_sampling, 0)  # shape: batch*num_samples x batch.shape[1:]
        masked = self.transform_method.transform(expanded, keep_per_image)
        votes = self.base_classifier.predict(masked)
        hard = torch.zeros((len(votes), self.number_of_classes)).cuda()
        hard.scatter_(1, votes.unsqueeze(1), 1)
        if not is_npy:
            return hard.reshape((batch.shape[0], n_sampling,) + hard.shape[1:]).mean(dim=1)
        else:
            return hard.reshape((batch.shape[0], n_sampling,) + hard.shape[1:]).mean(dim=1).cpu().numpy()

    @staticmethod
    def lc_bound(k, n, alpha):
        return proportion_confint(k, n, alpha=2 * alpha, method="beta")[0]

    @staticmethod
    def population_radius_for_majority(scores_of_true, size, keep):
        count = scores_of_true.shape[0]
        done = torch.zeros(count, dtype=torch.uint8)
        radii = torch.zeros(count, dtype=torch.long)
        radius = 0
        lhs = (1.5 - scores_of_true).squeeze(1)
        # print(lhs)
        while (done.sum() < count):
            rhs = math.factorial(size - radius) * math.factorial(size - keep) / (
                    math.factorial(size) * math.factorial(size - keep - radius))
            done[torch.tensor(lhs >= rhs)] = 1
            radii[torch.tensor(lhs < rhs)] = radius
            radius += 1
        return radii


class RandomSmooth4MalScan(RandomSmooth):
    def __init__(self, malscan_det: MalScan, number_of_classes: int, transform_method: RandomTransformer,
                 max_k: int, default_mode: bool):
        """
        Customized RandomSmooth w.r.t. Malscan
        """
        RandomSmooth.__init__(self, malscan_det, number_of_classes, transform_method, max_k, default_mode)

    def eval(self):
        pass

    def train(self):
        pass

    def certify(self, x: np.ndarray, label: int, adj_size: int, n_selection: int, n_estimation: int, k_per_instance: (float, int),
                alpha: float,
                top_k=1, x_sensitive_dix=None,
                device='cpu',
                verbose=False):
        with torch.no_grad():
            malscan_feature_vec = self.base_classifier.get_extra_feature(x,
                                                                         x_sensitive_dix,
                                                                         adj_size,
                                                                         True,
                                                                         device).float()
            counts_selection, k_per_instance = self.sample_funcs(malscan_feature_vec, n_selection,
                                                                 self.base_classifier.batch_size, k_per_instance,
                                                                 top_k, device, verbose)
            counts_estimation, _1 = self.sample_funcs(malscan_feature_vec, n_estimation, self.base_classifier.batch_size,
                                                      k_per_instance,
                                                      top_k, device, verbose)

            c_pred = counts_selection.argmax()
            n_targeted = counts_estimation[c_pred]

            prob_underlined = lower_confidence_interval(n_targeted, n_estimation, alpha)
            print(prob_underlined, n_targeted, n_estimation)
            radius = self.population_radius_for_majority(np.array([prob_underlined])[None, ...], n_estimation,
                                                         k_per_instance)

            if c_pred != label:
                return RandomSmooth.ABSTAIN
            else:
                return radius

    def predict(self, x: torch.Tensor, adj_size: int, n: int, k_per_instance: int, alpha: float,
                top_k=1, x_sensitive_dix=None, device='cpu', verbose=False):
        with torch.no_grad():
            if x_sensitive_dix is None:
                malscan_feature_vec = x.to(device)
            else:
                # get feature representations
                malscan_feature_vec = self.base_classifier.get_extra_feature(x,
                                                                             x_sensitive_dix,
                                                                             adj_size,
                                                                             True,
                                                                             device).float()
        counts, _1 = self.sample_funcs(malscan_feature_vec, n, self.base_classifier.batch_size, k_per_instance,
                                       top_k, device, verbose)
        # prediction
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, 0.5) > alpha:
            return self.ABSTAIN
        else:
            return top2[0]

    def sample_funcs(self, x: torch.Tensor, n, batch_size=64, k_per_instance=0,
                     top_k=1, device='cpu', verbose=False):
        assert n > 0
        self.base_classifier.eval()
        if 0 < k_per_instance <= 1:
            k_per_instance = min(math.ceil(len(x) * k_per_instance), self.max_k)
        preds = []
        for idx in range(n // batch_size + 1):
            current_batch_size = min(batch_size, n)
            n -= current_batch_size
            if current_batch_size <= 0:
                break

            with torch.no_grad():
                malscan_features = self.transform_method.transform(torch.tile(x[None, :], (current_batch_size, 1)),
                                                                   k_per_instance).squeeze()
                pred_batch = self.base_classifier.predict(malscan_features,
                                                          None,
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
        return np.bincount(np.concatenate(preds).squeeze(), minlength=self.number_of_classes), k_per_instance
