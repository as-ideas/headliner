import os
import unittest

import numpy as np
from numpy import array
from numpy.testing import assert_array_equal

from headliner.embeddings import read_glove, embedding_to_matrix


class TestEmbeddings(unittest.TestCase):

    def test_read_glove(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'resources/small_glove.txt')
        glove = read_glove(file_path, vector_dim=3)
        assert_array_equal(array([1, 2, 3]), glove['a'])

    def test_embedding_to_matrix(self):
        embedding = {'a': np.array(2), 'b': np.array(3), 'c': np.array(4)}
        token_index = {'a': 1, 'b': 2, 'd': 3}
        matrix = embedding_to_matrix(embedding, token_index, 1)
        np.testing.assert_array_equal(matrix[1], np.array(2))
        np.testing.assert_array_equal(matrix[2], np.array(3))
        # random values for zero index and tokens not in embedding
        self.assertTrue(-1 < float(matrix[0]) < 1)
        self.assertTrue(-1 < float(matrix[3]) < 1)
