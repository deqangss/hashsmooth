import numpy as np
import torch
import time
import math
from statsmodels.stats.proportion import proportion_confint, binom_test

from hashsmooth.utils_hash import lower_confidence_interval


class RandomTransformer(object):
    def __init__(self, k_randomcode, mask_id=50246, reuse_noise=False, seed=0):
        """
        codes are came from https://github.com/alevine0/randomizedAblation
        """
        self.k_randomcode = k_randomcode
        self.reuse_noise = reuse_noise
        self.seed = seed
        self.mask_id = mask_id

    def transform(self, batch: torch.Tensor, k_randomcode=0) -> torch.tensor:
        """
        the method warks as the method of 'random_mask_batch_one_sample' in randmozedAblation repository
        :return: transfored input
        """
        self.k_randomcode = k_randomcode if k_randomcode > 0 else self.k_randomcode
        flat = batch.reshape(batch.shape[0], -1)
        out_c1 = torch.ones(flat.shape, dtype=flat.dtype).cuda() * self.mask_id
        out_c2 = torch.ones(flat.shape, dtype=flat.dtype).cuda() * self.mask_id

        if (self.reuse_noise):
            ones = torch.ones(flat.shape[1]).cuda()
            idx = torch.multinomial(ones, self.k_randomcode)
            # idx = np.random.choice(flat.shape[1], self.k_randomcode, replace=False)
            out_c1[:, idx] = flat[:, idx]
            # out_c2[:, idx] = 1. - flat[:, idx]
        else:
            idx = self._batch_choose(flat.shape[1], self.k_randomcode, flat.shape[0])
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


class RandomSmooth(object):
    ABSTAIN = -1

    def __init__(self, base_classifier, num_of_classes, transform_method, max_k=1000,
                 default_mode=True):
        self.base_classifier = base_classifier
        self.num_of_classes = num_of_classes
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
        hard = torch.zeros((len(votes), self.num_of_classes)).cuda()
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
        print(scores_of_true.shape)
        lhs = (1.5 - scores_of_true).squeeze(1)
        print(lhs)
        while (done.sum() < count):
            rhs = math.factorial(size - radius) * math.factorial(size - keep) / (
                    math.factorial(size) * math.factorial(size - keep - radius))
            done[torch.tensor(lhs >= rhs)] = 1
            radii[torch.tensor(lhs < rhs)] = radius
            radius += 1
        return radii