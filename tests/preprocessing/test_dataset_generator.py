import unittest

import numpy as np

from headliner.preprocessing.dataset_generator import DatasetGenerator


class TestDatasetGenerator(unittest.TestCase):

    def test_generate_dataset(self):
        data = [([1, 1], [2, 2]), ([1, 1, 1], [2, 2])]
        batch_generator = DatasetGenerator(batch_size=1)

        # batch size = 1
        batches_iter = iter(batch_generator(lambda: data))
        batches = next(batches_iter)
        print(batches[0].numpy().tolist())
        expected = [[[1, 1]], [[2, 2]]]
        np.testing.assert_array_equal(expected[0], batches[0].numpy().tolist())
        np.testing.assert_array_equal(expected[1], batches[1].numpy().tolist())

        # batch size = 2
        batch_generator = DatasetGenerator(batch_size=2)
        batches_iter = iter(batch_generator(lambda: data))
        batches = next(batches_iter)
        expected = [[[1, 1, 0], [1, 1, 1]], [[2, 2], [2, 2]]]
        np.testing.assert_array_equal(expected[0], batches[0].numpy().tolist())
        np.testing.assert_array_equal(expected[1], batches[1].numpy().tolist())

        # batch size = 2, rank = 3
        data = [([1, 1], [0, 1], [2, 2]),
                ([1, 1, 1], [1, 1, 1], [3, 3, 3])]
        batch_generator = DatasetGenerator(batch_size=2, rank=3)
        batches_iter = iter(batch_generator(lambda: data))
        batches = next(batches_iter)
        expected = [[[1, 1, 0], [1, 1, 1]], [[0, 1, 0], [1, 1, 1]], [[2, 2, 0], [3, 3, 3]]]
        np.testing.assert_array_equal(expected[0], batches[0].numpy().tolist())
        np.testing.assert_array_equal(expected[1], batches[1].numpy().tolist())
