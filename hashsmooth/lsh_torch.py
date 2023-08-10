from abc import ABC, abstractmethod
import os
import warnings
from ast import literal_eval
import multiprocessing
import collections
import numpy as np
from scipy import sparse as sp_sparse
from scipy import integrate
from sklearn.preprocessing import MinMaxScaler
import torch

from tools.nn_model import get_simple_fc_model, train_sample_model
from hashsmooth import JaccardLSHTransformer, WeightedJaccardLSHTransformer, \
    PStableLSHTransformer, EditLSHTransformer, HammingLSHTransformer


class LSHTransformerTorch(ABC):
    def __init__(self, sub_k=128, null_value=0, seed=1):
        """
        abstract class for LSH transformations
        :param sub_k: a group of $sub_k$ hash codes
        :param null_value: an integer to fill the non-sampled positions
        :param seed: an integer of random seed
        """
        assert sub_k > 0 and isinstance(sub_k, int)
        assert seed > 0 and isinstance(seed, int)
        self.sub_k = sub_k
        self.null_value = null_value
        self.seed = seed
        self.random_generator_torch = torch.Generator()
        self.random_generator_torch.manual_seed(self.seed)

    @abstractmethod
    def transform(self, ipt: torch.Tensor) -> torch.Tensor:
        """
        lsh transformation for a data point
        :param ipt: an input data, e.g., a set of features or a representation vector
        :return: the transformed input that has the same feature type as that of the input
        """
        raise NotImplementedError

    @abstractmethod
    def get_collision_prob(self, distance):
        """
        get the collision probability for a given distance used by threat models
        """
        raise NotImplementedError


class JaccardLSHTransformerTorch(LSHTransformerTorch):
    def __init__(self, sub_k=128, null_value=0, seed=1):
        """
        The functionality is the same as JaccardLSHTransformer
        """
        super(JaccardLSHTransformerTorch, self).__init__(sub_k, null_value, seed)
        self.lsh_np = JaccardLSHTransformer(sub_k, null_value, seed)

    def transform(self, ipt: torch.Tensor) -> torch.Tensor:
        ipt_np = ipt.cpu().detach().numpy()
        ipt_tran = self.lsh_np.transform(ipt_np)
        elem_indicator = (ipt_np == ipt_tran) & (ipt_tran != self.null_value)
        elem_indicator_torch = torch.from_numpy(elem_indicator).to(ipt.device(), dtype=ipt.dtype)
        null_value_torch = torch.ones_like(ipt, device=ipt.device()) * self.null_value
        null_value_torch[elem_indicator] = 0.
        return ipt * elem_indicator_torch + null_value_torch

    def get_collision_prob(self, distance):
        return self.lsh_np.get_collision_prob(distance)


class WeightedJaccardLSHTransformerTorch(LSHTransformerTorch):
    def __init__(self, number_of_words, sub_k=128, null_value=0, seed=1):
        """
        The functionality is the same as WeightedJaccardLSHTransformer
        """
        super(WeightedJaccardLSHTransformerTorch, self).__init__(sub_k, null_value, seed)
        self.number_of_words = number_of_words
        self.lsh_np = WeightedJaccardLSHTransformer(self.number_of_words,
                                                    sub_k,
                                                    null_value,
                                                    seed
                                                    )

    def transform(self, ipt: torch.Tensor) -> torch.Tensor:
        ipt_np = ipt.cpu().detach().numpy()
        ipt_tran = self.lsh_np.transform(ipt_np)
        elem_indicator = (ipt_np == ipt_tran) & (ipt_tran != self.null_value)
        weights = np.copy(elem_indicator)
        weights[elem_indicator] = ipt_tran[elem_indicator] / ipt_np[elem_indicator]
        weights_torch = torch.from_numpy(weights).to(ipt.device(), dtype=ipt.dtype)
        null_value_torch = torch.ones_like(ipt, device=ipt.device()) * self.null_value
        null_value_torch[elem_indicator] = 0.
        return ipt * weights_torch + null_value_torch

    def get_collision_prob(self, distance):
        return self.lsh_np.get_collision_prob(distance)


class EditLSHTransformerTorch(LSHTransformerTorch):
    def __init__(self, number_of_words, sub_k=128, kmer_size=2, l_chucksize=1, null_value=0, seed=1):
        """
        The functionality is the same as EditLSHTransformer
        """
        super(EditLSHTransformerTorch, self).__init__(sub_k, null_value, seed)
        self.lsh_np = EditLSHTransformer(number_of_words,
                                         sub_k,
                                         kmer_size,
                                         l_chucksize,
                                         null_value,
                                         seed
                                         )

    def transform(self, ipt: torch.Tensor) -> torch.Tensor:
        ipt_np = ipt.cpu().detach().numpy()
        ipt_tran = self.lsh_np.transform(ipt_np)
        elem_indicator = (ipt_np == ipt_tran) & (ipt_tran != self.null_value)
        elem_indicator_torch = torch.from_numpy(elem_indicator).to(ipt.device(), dtype=ipt.dtype)
        null_value_torch = torch.ones_like(ipt, device=ipt.device()) * self.null_value
        null_value_torch[elem_indicator] = 0.
        return ipt * elem_indicator_torch + null_value_torch

    def get_collision_prob(self, distance):
        return self.lsh_np.get_collision_prob(distance)


class PStableLSHTransformerTorch(LSHTransformerTorch):
    def __init__(self, dimension, r, metric=1, sub_k=128, null_value=0, seed=1):
        """
        The functionality is the same as EditLSHTransformer
        """
        super(PStableLSHTransformerTorch, self).__init__(sub_k, null_value, seed)
        self.dimension = dimension
        self.r = r
        assert isinstance(self.dimension, int) and isinstance(self.r, (int, float))
        assert self.dimension > 0 and self.r > 0
        self.metric = metric
        self.basic_func = None
        self.scaler_minmax = (0., 0.)
        self.decoder = get_simple_fc_model(self.sub_k, self.dimension)

    def transform(self, ipt: torch.Tensor) -> torch.Tensor:
        return self._inverse_map(self._map(ipt))

    def _map(self, ipt: torch.Tensor) -> torch.Tensor:
        assert len(ipt.shape) == 2 and ipt.shape[1] == self.dimension
        a, b, self.basic_func = self._init_permutations()
        a = a.to(dtype=ipt.dtype, device=ipt.device())
        b = b.to(dtype=ipt.dtype, device=ipt.device())
        hash_codes = torch.floor(
            (torch.matmul(a, ipt[..., None]).squeeze() + b) / self.r
        )
        return hash_codes

    def _inverse_map(self, hash_codes_mapped):
        if self.scaler_minmax != (0., 0.):
            min_v, max_v = self.scaler_minmax
            hashcodes_scaled = (hash_codes_mapped - min_v) / (max_v - min-v)
            return self.decoder(hashcodes_scaled)
        else:
            raise NotFittedError("Model needs fitting first.\n")

    def _init_permutations(self):
        if self.metric == 2:
            a = torch.normal(0., 1., (self.sub_k, self.dimension), generator=self.random_generator_torch)
            b = torch.rand(self.sub_k, ) * self.r
            return a, b, self._f_gaussian
        elif self.metric == 1:
            a = torch.Tensor(size=(self.sub_k, self.dimension)).cauchy_(generator=self.random_generator_torch)
            b = torch.rand(self.sub_k, ) * self.r
            return a, b, self._f_cauchy
        else:
            raise ValueError(f"Support metric = 1 or 2, but got {self.metric}")

    def _f_gaussian(self, x):
        return np.e ** (-x ** 2 / 2) / np.sqrt(2 * np.pi)

    def _f_cauchy(self, x):
        return 1 / (np.pi * (1 + x ** 2))

    def train_decoder(self, hash_codes_mapped: torch.Tensor, x: torch.Tensor):
        min_v = hash_codes_mapped.min()
        max_v = hash_codes_mapped.max()
        self.scaler_minmax = (min_v, max_v)
        hashcodes_scale = (hash_codes_mapped - min_v) / (max_v - min_v)
        train_sample_model(self.decoder, hashcodes_scale, x)

    def get_collision_prob(self, distance):
        p, err = integrate.quad(lambda t: self.pstableProb(t, distance), 0, self.r)
        return 2 * p

    def _pstable_prob(self, x, d):
        if self.basic_func is None:
            if self.metric == 2:
                self.basic_func = self._f_gaussion
            elif self.metric == 1:
                self.basic_func = self._f_cauchy
            else:
                raise TypeError
        else:
            pass
        return self.basic_func(x / d) * (1. - x / self.r) / d


class HammingLSHTransformerTorch(LSHTransformerTorch):
    def __init__(self, dimension, sub_k=128, null_value=0, seed=1):
        super(HammingLSHTransformerTorch, self).__init__(sub_k, null_value, seed)
        self.lsh_np = HammingLSHTransformer(dimension, sub_k, null_value, seed)

    def transform(self, ipt: torch.Tensor) -> torch.Tensor:
        ipt_np = ipt.cpu().detach().numpy()
        ipt_tran = self.lsh_np.transform(ipt_np)
        elem_indicator = (ipt_np == ipt_tran) & (ipt_tran != self.null_value)
        elem_indicator_torch = torch.from_numpy(elem_indicator).to(ipt.device(), dtype=ipt.dtype)
        null_value_torch = torch.ones_like(ipt, device=ipt.device()) * self.null_value
        null_value_torch[elem_indicator] = 0.
        return ipt * elem_indicator_torch + null_value_torch

    def get_collision_prob(self, distance):
        return self.lsh_np.get_collision_prob(distance)