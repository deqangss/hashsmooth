from __future__ import division
from __future__ import print_function

import math
import hashlib
import struct
from ast import literal_eval
import numpy as np
from statsmodels.stats.proportion import proportion_confint, binom_test


def mod_inverse(x, m: int):
    """
    the inverse value of x under the modulo m for the LCG computation
    return -1 if does not exist (i.e., a and m are not co-prime)
    :param x: positive integers, np.array
    :param m: a prime integer
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert np.all(x > 0), "Support positive integers solely.\n"
    assert is_prime(m), f"{m} is not prime.\n"
    g, a, b = extended_gcd(x.astype(int), m)
    if isinstance(g, np.ndarray):
        if np.all(g == 1):
            return np.mod(a, m)
        else:
            raise ValueError(f"Expected a primer number, but got {m}.\n")
    elif isinstance(g, int):
        if g == 1:
            return a % m
        else:
            raise ValueError(f"Expected a primer number, but got {m}.\n")


def extended_gcd(x, m: int):
    """
    return a tuple of gcd(x, m), a, and b, wherein xa + mb = gcd(x, m)
    :param x: positive integers
    :param m: a big prime number
    """
    assert x.dtype == int
    m = np.ones_like(x, dtype=np.int32) * m
    last_remainder, remainder = x, m
    a, last_a, b, last_b = np.zeros_like(x), np.ones_like(x), np.ones_like(x), np.zeros_like(x)
    flag = remainder == 0
    remainder_collector, a_collector, b_collector = \
        np.empty(x.shape, dtype=int), np.empty(x.shape, dtype=int), np.empty(x.shape, dtype=int)
    while not np.all(flag):
        last_remainder, (quotient, remainder) = remainder, np.divmod(last_remainder, remainder)
        a, last_a = last_a - quotient * a, a
        b, last_b = last_b - quotient * b, b
        flag_last, flag = flag, remainder == 0
        flag_collector = flag.copy()
        flag_collector[flag_last] = False
        remainder_collector[flag_collector], a_collector[flag_collector], b_collector[flag_collector] = \
            last_remainder[flag_collector], last_a[flag_collector], last_b[flag_collector]
    # sign_a, sign_b = np.sign(x), np.sign(m)
    return remainder_collector, a_collector, b_collector


def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def array2kmer(x_2darray: np.ndarray, size=2):
    """
    covert 2D array to kmer, e.g., a 3x3 array to 2mer string
    array([[1,2,3],[4,5,6],[7,8,9]])->array([['1,2','2,3'],['4,5','5,6'],['7,8','8,9']])
    """
    assert isinstance(x_2darray, np.ndarray)
    assert len(x_2darray.shape) == 2
    assert x_2darray.shape[1] >= size
    r, c = x_2darray.shape
    kmer_array_list = np.empty(shape=(r, c - size + 1), dtype=list)
    for c in range(c - size + 1):
        kmer_array_list[:, c] = x_2darray[:, c: c + size].tolist()
    list2str = lambda list_: ','.join([str(e) for e in list_])
    v_list2str = np.vectorize(list2str)
    return v_list2str(kmer_array_list)


def str_quantifying(kmer_batch):
    def _hash_q(ipt_str):
        # we use 32-bit unsigned int values
        qutif_value = struct.unpack('<H',
                                    hashlib.sha1(ipt_str.encode('utf-8')).digest()[:2])
        return qutif_value[0]

    if isinstance(kmer_batch, str):
        return _hash_q(kmer_batch)
    elif isinstance(kmer_batch, np.ndarray):
        v_hash_q = np.vectorize(_hash_q)
        return v_hash_q(kmer_batch)
    else:
        raise TypeError


def lower_confidence_interval(n_targeted, n_estimation, alpha):
    """
    lower bound of confidence interval
    :param n_targeted: number of successes
    :param n_estimation: number of experiment trials
    :param alpha: confidence level
    """
    return proportion_confint(n_targeted, n_estimation, 2 * alpha, method='beta')[0]


def upper_confidence_interval(n_targeted, n_estimation, alpha):
    """
    upper bound of confidence interval
    :param n_targeted: number of successes
    :param n_estimation: number of experiment trials
    :param alpha: confidence level
    """
    return proportion_confint(n_targeted, n_estimation, alpha=2 * alpha, method="beta")[1]


def get_position(pos: int, shape: list) -> tuple:
    assert pos <= np.cumprod(shape)[-1]
    position_dim = []
    curr_pos = pos
    for dim in shape[::-1]:
        next_pos = curr_pos // dim
        curr_idx = curr_pos % dim
        position_dim.append(curr_idx)
        curr_pos = next_pos
    return tuple(position_dim[::-1])


if __name__ == "__main__":
    # lr = np.array([[5],
    #                [1],
    #                [7],
    #                [5],
    #                [6]], dtype=np.int)
    # x = np.random.randint(1, 11, (5, 3))
    # y = np.random.randint(0, 11, (5, 3))
    # hv = (x * lr + y) % 11
    # # print(hv - y)
    # print((hv - y) * mod_inverse(x, 11) % 11)
    # # assert np.all((hv - y) * mod_inverse(lr, 11) % 11 == x)
    # x = np.random.randint(0, 3, (5, 6))
    # x_kmer = array2kmer(x, 2)
    # print(x_kmer)
    # print(str_quantifying(x_kmer))
    shape = [6, 11, 7]
    pos = 452
    print(get_position(pos, shape))
