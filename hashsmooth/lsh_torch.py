from abc import ABC, abstractmethod
import os
import warnings
from ast import literal_eval
import multiprocessing
import collections
import numpy as np
from scipy import sparse as sp_sparse
from scipy import integrate
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler
import torch

from hashsmooth.utils_hash import mod_inverse, mod_inverse_torch
from hashsmooth.nn_model import get_simple_fc_model, train_sample_model
from hashsmooth import JaccardLSHTransformer, WeightedJaccardLSHTransformer, \
    PStableLSHTransformer, EditLSHTransformer, HammingLSHTransformer

from hashsmooth.lsh_np import _mersenne_primer, _mersenne_primer_large
_current_file_path = os.path.dirname(os.path.realpath(__file__))

class LSHTransformerTorch(ABC):
    def __init__(self, sub_k=128, null_value=0, seed=1, device='cpu'):
        """
        abstract class for LSH transformations
        :param sub_k: a group of $sub_k$ hash codes
        :param null_value: an integer to fill the non-sampled positions
        :param seed: an integer of random seed
        """
        assert sub_k >= 0 and isinstance(sub_k, int)
        assert seed >= 0 and isinstance(seed, int)
        if sub_k == 0:
            warnings.warn("Need to reinitialize the number of selected elements.\n")
        self.sub_k = sub_k
        self.null_value = null_value
        self.seed = seed
        self.device = device
        # self.random_generator_torch = torch.Generator(device=self.device) # cannot pickle the object
        # self.random_generator_torch.manual_seed(self.seed)
        self.random_generator_torch = None

    @abstractmethod
    def transform(self, ipt: torch.Tensor, sub_k_tmp=0) -> torch.Tensor:
        """
        lsh transformation for a data point
        :param ipt: an input data, e.g., a set of features or a representation vector
        :param sub_k_tmp: an alternative ways to initialize the number of selected elements
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
    def __init__(self, sub_k=128, null_value=0, seed=1, device='cpu'):
        """
        The functionality is the same as JaccardLSHTransformer
        """
        super(JaccardLSHTransformerTorch, self).__init__(sub_k, null_value, seed, device)
        self.offset = 0

    def _map(self, ipt: torch.Tensor, sub_k_tmp):
        """
        Map the input to a universal random space
        :param ipt: a tensor of word indices correspond to a vocabulary
        :param sub_k_tmp: an alternative number of selected elements
        """
        assert isinstance(ipt, torch.Tensor)
        assert len(ipt.shape) == 2, f"Expected a batch of vectors (e.g., 2D tensor), but got {len(ipt.shape)}.\n"

        # Convert dense tensor to sparse tensor in COO format
        ipt_sp = ipt.to_sparse()

        # Extract row and column indices from the COO sparse tensor
        row_indices, col_indices = ipt_sp.indices()

        # Count the number of non-zero elements in each row
        row_counts = torch.bincount(row_indices, minlength=ipt.shape[0])

        # Find the maximum number of non-zero elements in any row
        longest_num_nonzero = row_counts.max().item()

        # Initialize a tensor to store the non-zero column indices for each row
        ipt_nonzero_ind = torch.zeros((ipt.shape[0], longest_num_nonzero), dtype=torch.int64, device=self.device)

        # Group the non-zero column indices by row
        nonzero_list = torch.split(col_indices, row_counts.tolist())

        # Fill the non-zero indices into the `ipt_nonzero_ind` tensor
        for i, nonzero in enumerate(nonzero_list):
            ipt_nonzero_ind[i, :len(nonzero)] = nonzero

        if torch.min(ipt_nonzero_ind) == 0:
            self.offset = 1

        # Offset the non-zero indices
        ipt_nonzero_ind += self.offset

        r, c = ipt_nonzero_ind.shape
        k = self.sub_k if self.sub_k >= sub_k_tmp else sub_k_tmp

        # Initialize permutations (assuming `_init_permutations` is already converted to PyTorch)
        _x, _y = self._init_permutations(k, r)
        # Perform hashing operations
        ipt_ext = ipt_nonzero_ind[:, None, :]  # Shape: r, 1, c
        hash_code_tmp = (ipt_ext.to(torch.int64) * _x[..., None].to(torch.int64) + _y[..., None]) % _mersenne_primer
        hash_codes = hash_code_tmp.to(torch.int32).min(dim=-1).values

        return hash_codes, (_x, _y), ipt

    def _inverse_map(self, hash_codes_mapped):
        """
        Map hash codes back into the input space using PyTorch tensors.
        :param hash_codes_mapped: tuple containing hash codes, permutations, and original input tensor
        """
        hash_codes, permutations, ipt = hash_codes_mapped
        assert hash_codes.dtype == torch.int32

        _x, _y = permutations

        # Perform the modular inverse operation and mapping back to indices
        indices_tran = (hash_codes.to(torch.int64) - _y.to(torch.int64)) * mod_inverse_torch(_x, _mersenne_primer)
        indices_tran = torch.remainder(indices_tran, _mersenne_primer) - self.offset

        # Prepare the output tensor
        ipt_tran = torch.zeros_like(ipt)
        ipt_tran[:] = self.null_value
        n_instances = ipt.shape[0]

        # Use advanced indexing to map back into the input tensor
        ipt_tran[torch.arange(n_instances)[:, None], indices_tran] = ipt[
            torch.arange(n_instances)[:, None], indices_tran]

        return ipt_tran

    def _init_permutations(self, sub_k: int, r: int):
        """
        Generate random numbers for composing hash functions using PyTorch.
        :param sub_k: Number of permutations (hash functions)
        :param r: Number of rows (size of input)
        """
        # Generate two separate tensors for x and y permutations
        x_perms = torch.stack([
            torch.randint(1, _mersenne_primer, (r,), dtype=torch.int64, generator=self.random_generator_torch, device=self.device)
            for _ in range(sub_k)
        ], dim=1)

        y_perms = torch.stack([
            torch.randint(0, _mersenne_primer, (r,), dtype=torch.int64, generator=self.random_generator_torch, device=self.device)
            for _ in range(sub_k)
        ], dim=1)
        # Combine x_perms and y_perms into a single tensor
        perms = torch.stack([x_perms, y_perms], dim=0)  # Shape will be (2, r, sub_k)
        # Permute the tensor to match the expected output shape (r, sub_k, 2)
        return perms  # Shape becomes (r, sub_k, 2)

    def transform(self, ipt: torch.Tensor,sub_k_tmp=0) -> torch.Tensor:
        assert sub_k_tmp >= 0 and isinstance(sub_k_tmp, int)
        return self._inverse_map(self._map(ipt, sub_k_tmp))

    def get_collision_prob(self, distance: float):
        assert 0. <= distance <= 1.
        return 1 - distance


class WeightedJaccardLSHTransformerTorch(LSHTransformerTorch):
    def __init__(self, number_of_words, sub_k=128, null_value=0, seed=1, device='cpu'):
        """
        The functionality is the same as WeightedJaccardLSHTransformer
        """
        super(WeightedJaccardLSHTransformerTorch, self).__init__(sub_k, null_value, seed, device)
        self.number_of_words = number_of_words
        self.lsh_np = WeightedJaccardLSHTransformer(self.number_of_words,
                                                    sub_k,
                                                    null_value,
                                                    seed
                                                    )

    def transform(self, ipt: torch.Tensor, sub_k_tmp: int) -> torch.Tensor:
        with torch.no_grad():
            ipt_np = ipt.cpu().numpy()
            # ipt_trans = np.empty_like(ipt_np)
            # for i, ipt_array in enumerate(ipt_np):
            #     ipt_trans[i] = self.lsh_np.transform(ipt_array.reshape([1, -1]), sub_k_tmp)
            ipt_trans = self.lsh_np.transform(ipt_np, sub_k_tmp)

            elem_indicator = (ipt_trans != self.null_value)
            weights = np.copy(elem_indicator).astype(float)
            weights[elem_indicator] = ipt_trans[elem_indicator] / ipt_np[elem_indicator]
            weights_torch = torch.from_numpy(weights).to(device=ipt.device, dtype=ipt.dtype)
            null_value_torch = torch.ones_like(ipt, device=ipt.device) * self.null_value
            null_value_torch[torch.from_numpy(elem_indicator)] = 0.

        return ipt * weights_torch + null_value_torch

    def get_collision_prob(self, distance):
        return self.lsh_np.get_collision_prob(distance)


class EditLSHTransformerTorch(LSHTransformerTorch):
    def __init__(self, number_of_words, sub_k=128, kmer_size=2, l_chucksize=1, null_value=0, pad_value=1, position_fixed=False, seed=1, device='cpu'):
        """
        The functionality is the same as EditLSHTransformer
        """
        super(EditLSHTransformerTorch, self).__init__(sub_k, null_value, seed, device)
        self.lsh_np = EditLSHTransformer(number_of_words,
                                         sub_k,
                                         kmer_size,
                                         l_chucksize,
                                         null_value,
                                         pad_value,
                                         position_fixed,
                                         seed
                                         )

    def transform(self, ipt: torch.Tensor, sub_k_tmp: int) -> torch.Tensor:
        with torch.no_grad():
            ipt_np = ipt.detach().cpu().numpy()
            ipt_tran = self.lsh_np.transform(ipt_np, sub_k_tmp)
            return torch.from_numpy(ipt_tran).to(ipt.device, dtype=ipt.dtype)

    def get_collision_prob(self, distance):
        return self.lsh_np.get_collision_prob(distance)


class PStableLSHTransformerTorch(LSHTransformerTorch):
    def __init__(self, dimension, r, metric=1, sub_k=128, null_value=0, seed=1, device='cpu'):
        """
        The functionality is the same as EditLSHTransformer
        """
        super(PStableLSHTransformerTorch, self).__init__(sub_k, null_value, seed, device)
        self.dimension = dimension
        self.r = r
        assert isinstance(self.dimension, int) and isinstance(self.r, (int, float))
        assert self.dimension > 0 and self.r > 0
        self.metric = metric
        self.basic_func = None
        self.scaler_minmax = (0., 0.)
        self.decoder = get_simple_fc_model(self.sub_k, self.dimension)

    def transform(self, ipt: torch.Tensor, sub_k_tmp: int) -> torch.Tensor:
        return self._inverse_map(self._map(ipt, sub_k_tmp))

    def _map(self, ipt: torch.Tensor, sub_k_tmp: int) -> torch.Tensor:
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
    def __init__(self, dimension, sub_k=128, null_value=0, seed=1, device='cpu'):
        super(HammingLSHTransformerTorch, self).__init__(sub_k, null_value, seed, device)
        self.dimension = dimension

    def _map(self, ipt: torch.Tensor, sub_k_tmp: int):
        assert len(ipt.shape) == 2
        if ipt.dtype != torch.int:
            ipt = ipt.to(torch.int)
        assert ((ipt == 0) | (ipt == 1)).all(), "Expect binary array. Exit!\n"

        batch_size = ipt.shape[0]
        sub_k = self.sub_k if self.sub_k >= sub_k_tmp else sub_k_tmp
        permutations = np.array([self._init_permutations(sub_k) for _ in range(batch_size)])
        hash_codes = torch.empty((batch_size, sub_k), dtype=ipt.dtype, device=ipt.device)
        hash_codes[:] = ipt[np.array(range(batch_size))[:, np.newaxis], permutations]
        return hash_codes, permutations

    def _inverse_map(self, hash_codes_mapped):
        hash_codes, permutations = hash_codes_mapped
        assert len(hash_codes.shape) == 2
        batch_size_ = hash_codes.shape[0]
        input_rtn = torch.ones((batch_size_, self.dimension), dtype=torch.int, device=self.device) * self.null_value
        input_rtn[np.array(range(batch_size_))[:, np.newaxis], permutations] = hash_codes
        return input_rtn


    def _init_permutations(self, sub_k, replace=True):
        return np.random.choice(self.dimension, sub_k, replace=replace)


    def get_collision_prob(self, distance):
        assert 0. <= distance <= 1., "Expected the normalized distance.\n"
        return 1 - distance

    def transform(self, ipt: torch.Tensor, sub_k_tmp: int=0) -> torch.Tensor:
        assert sub_k_tmp >= 0 and isinstance(sub_k_tmp, int)
        return self._inverse_map(self._map(ipt, sub_k_tmp))


def test_jaccard_dist():
    input_array = np.random.randint(1, 10000, (2, 100))
    jaccard_lsh = JaccardLSHTransformerTorch(sub_k=3, null_value=-1, seed=2345)
    i: int = 1
    while i <= 10:
        input_transf = jaccard_lsh.transform(torch.from_numpy(input_array))
        input_transf = torch.masked_select(input_transf, input_transf != jaccard_lsh.null_value).numpy()
        abc = set(input_transf.flatten())
        print(abc)
        print(set(input_array.flatten()))
        assert abc.issubset(set(input_array.flatten()))
        i += 1


def test_hamming_dist():
    input_array = np.random.randint(0, 2, (2, 100))
    hamming_lsh = HammingLSHTransformerTorch(dimension=100, sub_k=3, null_value=-1, seed=2345)
    i: int = 1
    while i <= 10:
        print(i)
        input_transf = hamming_lsh.transform(torch.from_numpy(input_array)).cpu().numpy()
        flag = input_transf == 1
        print(np.sum(flag, axis=1))
        assert np.all(input_array[flag] == 1), "inconsistent outputs"
        i+=1

if __name__ == "__main__":
    # test_pstable_dist()
    for _ in range(3):
        test_hamming_dist()
    # test_w_jaccard_lsh()