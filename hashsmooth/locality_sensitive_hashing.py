from abc import ABC, abstractmethod
import os
import warnings
from ast import literal_eval
import multiprocessing
import collections
import numpy as np
from scipy import sparse as sp_sparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import NotFittedError

from hashsmooth.utils_hash import mod_inverse, array2kmer, str_quantifying
from tools.utils import read_pickle, dump_pickle
from tools.nn_model import get_simple_fc_model, train_sample_model

_mersenne_primer = np.int((1 << 31) - 1)  # https://en.wikipedia.org/wiki/Mersenne_prime
_mersenne_primer_large = np.int((1 << 61) - 1)
_current_file_path = os.path.dirname(os.path.realpath(__file__))


class LSHTransformer(ABC):
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
        self.random_generator_np = np.random.RandomState(seed=self.seed)

    def transform(self, ipt) -> np.ndarray:
        """
        lsh transformation for a data point
        :param ipt: an input data, e.g., a set of features or a representation vector
        :return: the transformed input that has the same feature type as that of the input
        """
        return self._inverse_map(self._map(ipt))  # it should emerge in a pair-wise fashion

    @abstractmethod
    def _map(self, ipt):
        """
        mapping input by a type of LSH functions
        :param ipt: an input data, e.g., a set of features or a representation vector
        """
        raise NotImplementedError("Not implemented yet.\n")

    @abstractmethod
    def _inverse_map(self, hash_codes_mapped):
        """
        mapping the hash codes back to the input space, on which the classifier takes as input
        :param hash_codes_mapped: hash codes
        """
        raise NotImplementedError("Not implemented yet.\n")


class JaccardLSHTransformer(LSHTransformer):
    def __init__(self, sub_k=128, null_value=0, seed=1):
        """
        LSH transformations in terms of Jaccard distance
        We use the universal hashing https://en.wikipedia.org/wiki/Universal_hashing
        """
        super(JaccardLSHTransformer, self).__init__(sub_k, null_value, seed)
        self.offset = 0

    def _map(self, ipt):
        """
        map the input to a universal random space
        :param ipt: an array of word indices correspond to a vocabulary
        """
        assert isinstance(ipt, np.ndarray)
        assert len(ipt.shape) == 2, f"Expected a batch of vectors (e.g., 2D array), but got {len(ipt.shape)}.\n"
        if ipt.dtype != np.int:
            warnings.warn(f"Convert the {ipt.dtype} input to be unsigned int.\n")
            ipt = ipt.astype(np.uint32)
        if np.min(ipt) == 0:
            self.offset = 1
        ipt += self.offset
        _x, _y = self._init_permutations()  # keep the same data format
        r, c = ipt.shape
        hash_codes = np.ones(shape=(r, self.sub_k), dtype=np.uint32) * _mersenne_primer
        # In case of long sentence, we split a batch of sentence element-wisely
        for idx_c in range(c):
            ipt_columnwise = ipt[:, idx_c:idx_c + 1]
            hash_code_tmp = (ipt_columnwise.astype(np.uint64) * np.tile(_x, (r, 1)).astype(
                np.uint64) + _y) % _mersenne_primer
            hash_codes = np.stack([hash_code_tmp.astype(np.uint32), hash_codes]).min(axis=0)
        return hash_codes, (_x, _y)

    def _inverse_map(self, hash_codes_mapped):
        """
        map hash codes back into the input space
        :param hash_codes_mapped: pair of hash codes and permutations
        """
        hash_codes, permutations = hash_codes_mapped
        assert hash_codes.dtype == np.uint32
        _x, _y = permutations
        return np.mod(
            (hash_codes.astype(np.int) - _y.astype(np.int)) * mod_inverse(_x, _mersenne_primer),
            _mersenne_primer) - self.offset

    def _init_permutations(self):
        """
        generate random numbers for compositing hash functions
        """
        return np.array([(self.random_generator_np.randint(1, _mersenne_primer, dtype=np.uint32),
                          self.random_generator_np.randint(0, _mersenne_primer, dtype=np.uint32)) for _ in
                         range(self.sub_k)],
                        dtype=np.uint32
                        ).T


class WeightedJaccardLSHTransformer(LSHTransformer):
    def __init__(self, number_of_words, sub_k=128, null_value=0, seed=1):
        """
        LSH transformations in terms of weighted jaccard distance, and please see the link of
        http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/36928.pdf
        to get more details.
        :param number_of_words: number of all words in the vocabulary
        :param sub_k: number of the hash functions
        :param null_value: an integer to fill the non-sampled positions
        :param seed: random seed
        """
        super(WeightedJaccardLSHTransformer, self).__init__(sub_k, null_value, seed)
        self.number_of_words = number_of_words

    def _map(self, ipt: np.ndarray):
        """
        map the input to hash codes
        the input contains the corresponding occurrence of each word in the vocabulary
        I.e., each entity of the input is an integer
        :param ipt: a batch of input vector
        :return: $sub_k$ of hash codes
        """
        assert isinstance(ipt, (np.ndarray, sp_sparse.spmatrix)), \
            f"Expected 2D numpy array, but got {type(ipt)}."
        assert ipt.ndim == 2
        assert ipt.shape[1] == self.number_of_words, \
            f"Each instance has the dimension of input as {self.number_of_words}, but got {ipt.shape[1]}."

        sp_ipt = sp_sparse.csr_matrix(ipt, dtype=np.float32, copy=True)
        sp_ipt.sort_indices()

        batch_size = sp_ipt.shape[0]
        batch_hash_codes = [None for _ in range(batch_size)]

        # obtain the indices of nonzero values
        r_idx, c_idx = sp_ipt.nonzero()

        # obtain permutations
        rk, beta_k, ln_ck = self._init_permutations()
        rk_c_merged = np.array(rk, copy=True)[:, c_idx]
        betak_c_merged = np.array(beta_k, copy=True)[:, c_idx]
        ln_ck_c_merged = np.array(ln_ck, copy=True)[:, c_idx]
        log_data = np.log(sp_ipt[r_idx, c_idx].A1)
        log_data = np.vstack([log_data] * self.sub_k)  # sub_k x number_of_words

        # intermediates stated in the paper
        t = np.floor(log_data / rk_c_merged + betak_c_merged)  # sub_k x number_of_words
        ln_y = (t - betak_c_merged) * rk_c_merged
        ln_a = ln_ck_c_merged - ln_y - rk_c_merged
        # obtain the hash codes instance-wisely
        r_rear_idx = sp_ipt.indptr.tolist()[
                     1:]  # remove the head idx of first instance for facilitating the implementation
        r_rear_idx.append(sp_ipt.nnz)
        instance_head_idx = 0
        for current_idx in range(ipt.shape[0]):
            instance_next_idx = r_rear_idx[current_idx]

            # the instance of zero vector is not permitted
            if instance_head_idx == instance_next_idx:
                break

            instance_span = c_idx[instance_head_idx: instance_next_idx]
            instance_ln_a = ln_a[:, instance_head_idx: instance_next_idx]
            instance_argmin = np.argmin(instance_ln_a, axis=1)
            instance_k = instance_span[instance_argmin]

            hash_codes = np.zeros((self.sub_k, 2), dtype=int)
            hash_codes[:, 0] = instance_k
            # We here make a difference from the paper (using y rather than t) because we need hash values
            # hash_codes[:, 1] = t[range(self.sub_k), instance_head_idx + instance_argmin]
            hash_codes[:, 1] = np.ceil(np.exp(ln_y))[range(self.sub_k), instance_head_idx + instance_argmin]
            batch_hash_codes[current_idx] = hash_codes

            instance_head_idx = instance_next_idx

        return batch_hash_codes

    def _inverse_map(self, hash_codes_mapped):
        """
        map the hash codes back to the input space
        :param hash_codes_mapped: a list of hash codes returned by the _amp function
        """
        assert isinstance(hash_codes_mapped, list)
        assert len(hash_codes_mapped) > 0
        input_rtn = np.ones(shape=(len(hash_codes_mapped), self.number_of_words), dtype=np.int) * self.null_value
        n_proc = 1 if multiprocessing.cpu_count() // 2 <= 1 else multiprocessing.cpu_count() // 2
        with multiprocessing.Pool(n_proc) as pool:
            for idx, input_transf in enumerate(pool.imap(_wrapper_w_jaccard, zip(hash_codes_mapped, input_rtn))):
                input_rtn[idx] = input_transf
        # for i, hash_codes in enumerate(hash_codes_mapped):
        #     b_input[i][hash_codes[:, 0]] = hash_codes[:, 1]
        return input_rtn

    def _init_permutations(self):
        """
        generate random numbers for compositing hash functions
        """
        rk = self.random_generator_np.gamma(2, 1, (self.sub_k, self.number_of_words)).astype(np.float32)
        ln_ck = np.log(self.random_generator_np.gamma(2, 1, (self.sub_k, self.number_of_words))).astype(np.float32)
        beta_k = self.random_generator_np.uniform(0, 1, (self.sub_k, self.number_of_words)).astype(np.float32)
        return rk, ln_ck, beta_k


def _wrapper_w_jaccard(args):
    _codes, _transf = args[0], args[1]
    _transf[_codes[:, 0]] = _codes[:, 1]
    return _transf


class EditLSHTransformer(LSHTransformer):
    def __init__(self, number_of_words, sub_k=128, kmer_size=2, l_chucksize=1, null_value=0, seed=1):
        """
        LSH transformations for the edit distance. We leverage the method proposed by the following paper:
        Guillaume Marçais and others, Locality-sensitive hashing for the edit distance, Bioinformatics, 35(14): https://doi.org/10.1093/bioinformatics/btz354
        The method is implemented in the manner with our understanding, which may be different from the official version.
        Please adapt it with caution.
        :param number_of_words: number of all words in the vocabulary
        :param sub_k: a group of $sub_k$ hash codes
        :param kmer_size:token size for splitting a sequence
        :param l_chucksize: number of kmer to composite a chuck. The default value is $sub_k$, because we do not need it
        :param null_value: an integer to fill the non-sampled positions
        :param seed: an integer of random seed
        """
        super(EditLSHTransformer, self).__init__(sub_k, null_value, seed)
        self.number_of_words = number_of_words
        if self.number_of_words >= (1 << 16) - 1:
            warnings.warn("Too much words triggers the collision.\n")
        self.kmer_size = kmer_size
        self.l_chucksize = l_chucksize
        self.hashcode2input_dict = collections.defaultdict(str)
        self.dict_saving_path = os.path.join(_current_file_path, "res/hashcodes2input.dict")
        if os.path.exists(self.dict_saving_path):
            self.hashcode2input_dict = read_pickle(self.dict_saving_path)
        self._x = 1
        self._y = 0

    def _map(self, ipt: np.ndarray):
        """
        the input is a batch of sequences with same dimension
        :param ipt: 2D numpy.array
        """
        assert len(ipt.shape) == 2
        # convert the sequence indices to kmer strings, e.g.,: 1,2,3 --> 1,2;2,3 (kmer_size = 2)
        kmer_batch = array2kmer(ipt)
        # quantify the kmer strings
        kmer_quantification = str_quantifying(kmer_batch)
        print("max:", np.max(kmer_quantification))
        # record the mapping for bijection
        prev_len = len(self.hashcode2input_dict.items())
        self.hashcode2input_dict.update([(k, v) for k, v in zip(kmer_quantification.flatten(), kmer_batch.flatten())])
        curr_len = len(self.hashcode2input_dict.items())
        # conduct mapping: need positions for retaining the sequential relationship
        _x, _y = self._init_permutations()
        r, c = kmer_quantification.shape
        hash_codes = np.ones(shape=(r, self.sub_k), dtype=np.uint32) * _mersenne_primer
        positions = np.zeros(shape=(r, self.sub_k), dtype=np.uint32)
        # the same as that of jaccard LSH, note: there are duplications for each instance
        for idx_c in range(c):
            last_hash_codes = hash_codes
            kmer_q_columnwise = kmer_quantification[:, idx_c:idx_c + 1]
            hash_code_tmp = (kmer_q_columnwise.astype(np.uint64) * np.tile(_x, (r, 1)).astype(
                np.uint64) + _y) % _mersenne_primer
            hash_codes = np.stack([hash_code_tmp.astype(np.uint32), hash_codes]).min(axis=0)
            positions[hash_codes < last_hash_codes] = idx_c  # update positions
        if curr_len > prev_len:
            dump_pickle(self.hashcode2input_dict, self.dict_saving_path)
        # we just need the hash codes, so we stop here and return
        return hash_codes, positions, _x, _y

    def _inverse_map(self, hash_codes_mapped) -> np.ndarray:
        """
        mapping hash codes (which is returned by the method of '_map') back to the input space.
        :param hash_codes_mapped: a pair of hash codes, positions, and permutations
        """
        if len(self.hashcode2input_dict) <= 0:
            self.hashcode2input_dict = read_pickle(self.dict_saving_path)
        assert len(self.hashcode2input_dict) > 0, "No mapping records. Exit!"
        hash_codes, positions, _x, _y = hash_codes_mapped
        batch_size_, sub_k_ = hash_codes.shape
        assert sub_k_ == self.sub_k
        assert hash_codes.dtype == np.uint32
        kmer_encodings = np.mod(
            (hash_codes.astype(np.int) - _y.astype(np.int)) * mod_inverse(_x, _mersenne_primer),
            _mersenne_primer)
        kmer_decode = lambda kmer_end: literal_eval(self.hashcode2input_dict.get(kmer_end))
        kmer_decodings = np.stack(np.vectorize(kmer_decode)(kmer_encodings), axis=-1)
        input_rtn = np.ones(shape=(batch_size_, self.number_of_words), dtype=int) * self.null_value
        for c_idx in range(sub_k_):
            for r_idx in range(batch_size_):
                input_rtn[r_idx, positions[r_idx, c_idx]: positions[r_idx, c_idx] + self.kmer_size] = \
                    kmer_decodings[r_idx, c_idx]
        return input_rtn

    def _init_permutations(self):
        """
        generate random numbers for compositing hash functions
        """
        return np.array([(self.random_generator_np.randint(1, _mersenne_primer, dtype=np.uint32),
                          self.random_generator_np.randint(0, _mersenne_primer, dtype=np.uint32)) for _ in
                         range(self.sub_k)],
                        dtype=np.uint32
                        ).T


class PStableLSHTransformer(LSHTransformer):
    def __init__(self, dimension, r, metric=1, sub_k=128, null_value=0, seed=1):
        """
        Codes are adapted from the repository: https://github.com/CharlesLiu7/p-stable-lsh-python,
        which implements the lsh method proposed in the paper below:
        Mayur Datar, Nicole Immorlica, Piotr Indyk, and Vahab S. Mirrokni. Locality-sensitive hashing scheme based on p-stable distributions. SCG '04. ACM, New York, NY, USA, 253262.
        :param dimension: dimension of the input vector
        :param r: the radius or width stated $r$ in the paper
        :param metric: the $l_p$ metric measures pertubations (p=1 or p=2)
        :param sub_k: a group of $sub_k$ hash codes
        :param null_value: an integer to fill the non-sampled positions
        :param seed: an integer of random seed
        """
        super(PStableLSHTransformer, self).__init__(sub_k, null_value, seed)
        self.dimension = dimension
        self.r = r
        assert isinstance(self.dimension, int) and isinstance(self.r, (int, float))
        assert self.dimension > 0 and self.r > 0
        self.metric = metric
        self.basic_func = None
        # build a model
        self.scaler = MinMaxScaler()
        self.decoder = get_simple_fc_model(self.sub_k, self.dimension)

    def _map(self, ipt: np.ndarray) -> np.ndarray:
        """
        conduct p-stable distribution based lsh transformations
        :param ipt: a batch of representation vector
        """
        assert len(ipt.shape) == 2 and ipt.shape[1] == self.dimension
        a, b, self.basic_func = self._init_permutations()
        hash_codes = np.floor((np.transpose(np.dot(a, np.expand_dims(ipt, -1)).squeeze(), [1, 0]) + b) / self.r)

        return hash_codes

    def _inverse_map(self, hash_codes_mapped):
        """
        leverage the auto-encoder strategy. Please train the decoder first.
        :param hash_codes_mapped: hash codes
        """
        try:
            return self.decoder(self.scaler.transform(hash_codes_mapped))
        except NotFittedError:
            raise NotFittedError("Model needs fitting first.\n")

    def _f_gaussion(self, x: np.ndarray):
        """
        Standard gaussian noises corresponds to x
        :param x: 2D array
        """
        return np.e ** (-x ** 2 / 2) / np.sqrt(2 * np.pi)

    def _f_cauchy(self, x: np.ndarray):
        """
        Standard cauchy noises corresponds to x
        :param x: 2D array
        """
        return 1 / (np.pi * (1 + x ** 2))

    def _init_permutations(self):
        if self.metric == 2:
            a = np.array([self.random_generator_np.normal(size=self.dimension) for _ in range(self.sub_k)])
            b = np.array([self.random_generator_np.uniform(0, self.r) for _ in range(self.sub_k)])
            return a, b, self._f_gaussion
        elif self.metric == 1:
            a = np.array([self.random_generator_np.standard_cauchy(self.dimension) for _ in range(self.sub_k)])
            b = np.array([self.random_generator_np.uniform(0, self.r) for _ in range(self.sub_k)])
            return a, b, self._f_cauchy
        else:
            raise ValueError(f"Support metric = 1 or 2, but got {self.metric}")

    def train_decoder(self, hash_codes_mapped: np.ndarray, x: np.ndarray):
        self.scaler.fit(hash_codes_mapped)
        hash_codes_mapped = self.scaler.transform(hash_codes_mapped)
        train_sample_model(self.decoder, hash_codes_mapped, x)


class HammingLSHTransformer(LSHTransformer):
    def __init__(self, dimension, sub_k=128, null_value=0, seed=1):
        """
        bit sampling method
        :param dimension: number of works
        :param sub_k: a group of $sub_k$ hash codes
        :param null_value: an integer to fill the non-sampled positions
        :param seed: an integer of random seed
        """
        super(HammingLSHTransformer, self).__init__(sub_k, null_value, seed)
        self.dimension = dimension

    def _map(self, ipt: np.ndarray):
        """
        sampling multiple bits
        :param ipt: 2D array
        """
        assert len(ipt.shape) == 2
        if ipt.dtype != int:
            ipt = ipt.astype(np.int)
        assert ((ipt == 0) | (ipt == 1)).all(), "Expect binary array. Exit!\n"

        batch_size = ipt.shape[0]
        permutations = np.array([self._init_permutations() for _ in range(batch_size)])
        hash_codes = np.empty(shape=(batch_size, self.sub_k), dtype=np.ndarray)
        hash_codes[:] = ipt[np.array(range(batch_size))[:, np.newaxis], permutations]
        return hash_codes, permutations

    def _inverse_map(self, hash_codes_mapped: np.ndarray):
        """
        mapping lsh codes back to the input space
        """
        hash_codes, permutations = hash_codes_mapped
        assert len(hash_codes.shape) == 2
        batch_size_ = hash_codes.shape[0]
        input_rtn = np.ones(shape=(batch_size_, self.dimension)) * self.null_value
        input_rtn[np.array(range(batch_size_))[:, np.newaxis], permutations] = hash_codes
        return input_rtn

    def _init_permutations(self, replace=True):
        return np.random.choice(self.dimension, self.sub_k, replace=replace)


def test_pstable_dist():
    np.random.seed(0)
    x = np.random.uniform(0, 1, (5, 6))
    pstable_lsh = PStableLSHTransformer(dimension=x.shape[1], r=3.0, metric=2, sub_k=2)
    hash_codes = pstable_lsh._map(x)


if __name__ == "__main__":
    test_pstable_dist()
