import unittest

from headliner.preprocessing.keras_tokenizer import KerasTokenizer
from headliner.preprocessing.vectorizer import Vectorizer


class TestVectorizer(unittest.TestCase):

    def test_vectorize(self):
        data = [('a b c', 'd')]
        tokenizer_encoder = KerasTokenizer()
        tokenizer_decoder = KerasTokenizer()
        tokenizer_encoder.fit([data[0][0]])
        tokenizer_decoder.fit([data[0][1]])
        vectorizer = Vectorizer(tokenizer_encoder,
                                tokenizer_decoder,
                                max_input_len=5,
                                max_output_len=3)
        data_vectorized = [vectorizer(d) for d in data]
        self.assertEqual([([1, 2, 3], [1, 0, 0])], data_vectorized)

        data = [('a b c', 'd d d d')]
        data_vectorized = [vectorizer(d) for d in data]
        self.assertEqual([([1, 2, 3], [1, 1, 1])], data_vectorized)

        data = [('a a b b b b b', 'd d d d')]
        data_vectorized = [vectorizer(d) for d in data]
        self.assertEqual([([1, 1, 2, 2, 2], [1, 1, 1])], data_vectorized)
