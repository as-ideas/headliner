import itertools
import random
import unittest

from headliner.preprocessing.bucket_generator import BucketGenerator


class TestBucketGenerator(unittest.TestCase):

    def test_generate_batches_nonrandom(self):
        data = [[i] * i for i in range(10, 0, -1)]
        bucket_generator = BucketGenerator(lambda e: len(e),
                                           batch_size=2,
                                           buffer_size_batches=100,
                                           shuffle=False)
        buckets_gen = bucket_generator(data)
        result = [el for el in buckets_gen]
        expected = [[i] * i for i in range(1, 11)]
        self.assertEqual(expected, result)

    def test_generate_batches_random(self):
        data = [[i] * i for i in range(100, 0, -1)]
        random.shuffle(data)
        bucket_generator = BucketGenerator(lambda e: len(e),
                                           batch_size=2,
                                           buffer_size_batches=100,
                                           batches_to_bucket=10,
                                           shuffle=True,
                                           seed=42)
        buckets_gen = bucket_generator(data)
        result = [el for el in buckets_gen]

        # check whether all elements are returned
        expected_elements = list(itertools.chain.from_iterable(data))
        expected_elements.sort()
        result_elements = list(itertools.chain.from_iterable(result))
        result_elements.sort()
        self.assertEqual(expected_elements, result_elements)

        # check whether sequences of similar length are bucketed together
        # -> compare sum of length difference within batches against non-bucketed data
        raw_total_length_diff = 0
        result_total_length_diff = 0
        for i in range(0, len(data), 2):
            first_seq_raw, second_seq_raw = data[i:i + 2]
            first_seq_bucket, second_seq_bucket = result[i:i + 2]
            raw_total_length_diff += abs(len(second_seq_raw) - len(first_seq_raw))
            result_total_length_diff += abs(len(second_seq_bucket) - len(first_seq_bucket))
        self.assertTrue(result_total_length_diff < raw_total_length_diff / 4)
