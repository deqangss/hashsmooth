import unittest
import numpy as np

from hashsmooth.locality_sensitive_hashing import JaccardLSHTransformer,\
    WeightedJaccardLSHTransformer, \
    EditLSHTransformer, \
    HammingLSHTransformer


class TestLSHTransformation(unittest.TestCase):
    def setUp(self) -> None:
        """
        settings before running the tests
        """
        self.seed = 2345
        self.sub_k = 5
        self.null_value = 0
        self.batch_size = 16
        self.num_of_word = 100000
        self.max_index = 100000

    def test_jaccard_lsh(self):
        input_array = np.random.randint(1, self.max_index, (self.batch_size, 100))
        jaccard_lsh = JaccardLSHTransformer(sub_k=self.sub_k, null_value=self.null_value, seed=self.seed)
        i: int = 1
        while i <= 10:
            input_transf = jaccard_lsh.transform(input_array)
            self.assertTrue(set(input_transf.flatten()).issubset(set(input_array.flatten())))
            i += 1

    def test_w_jaccard_lsh(self):
        # each entity is the corresponding occurrence of words in vocabulary
        input_array = np.random.randint(0, 100, (self.batch_size, self.num_of_word), dtype=np.uint32)
        w_jaccard_lsh = WeightedJaccardLSHTransformer(number_of_words=self.num_of_word,
                                                      sub_k=self.sub_k,
                                                      null_value=self.null_value,
                                                      seed=self.seed)
        i = 1
        while i <= 10:
            input_transf = w_jaccard_lsh.transform(input_array)
            self.assertTrue(np.all(input_transf <= input_array))
            self.assertTrue(np.all(input_transf >= 0))
            i += 1

    def test_editdistance_lsh(self):
        # each row is a sequence
        input_seq_array = np.random.randint(1, self.num_of_word, (self.batch_size, self.max_index), dtype=int)
        edit_lsh = EditLSHTransformer(number_of_words=self.num_of_word,
                                      kmer_size=2,
                                      sub_k=self.sub_k,
                                      null_value=self.null_value,
                                      seed=self.seed
                                      )
        i = 1
        while i <= 10:
            input_transf = edit_lsh.transform(input_seq_array)
            origin_elements = set(input_seq_array.flatten())
            origin_elements.add(self.null_value)
            self.assertTrue(set(input_transf.flatten()).issubset(origin_elements))
            i += 1

    def test_hamming_distance(self):
        binary_input_array = np.random.randint(0, 2, (self.batch_size, self.num_of_word))
        hamming_lsh = HammingLSHTransformer(dimension=self.num_of_word,
                                            sub_k=self.sub_k,
                                            null_value=self.null_value,
                                            seed=self.seed
                                            )
        input_transf = hamming_lsh.transform(binary_input_array)
        self.assertTrue(((input_transf == 0) | (input_transf == 1)).all())
        self.assertTrue(set(np.flatnonzero(input_transf)).issubset(set(np.flatnonzero(binary_input_array))))


if __name__ == "__main__":
    unittest.main()