import unittest

from keras_preprocessing.text import Tokenizer
from headliner.preprocessing.vectorizer import Vectorizer


class TestPreprocessor(unittest.TestCase):

    def test_vectorize(self):
        data = [('a b c', 'd')]
        tokenizer_encoder = Tokenizer()
        tokenizer_decoder = Tokenizer()
        tokenizer_encoder.fit_on_texts([data[0][0]])
        tokenizer_decoder.fit_on_texts([data[0][1]])
        vectorizer = Vectorizer(tokenizer_encoder, tokenizer_decoder)
        data_vectorized = [vectorizer(d) for d in data]
        self.assertEqual([([1, 2, 3], [1])], data_vectorized)

        data = [('a b c', 'd d d d')]
        data_vectorized = [vectorizer(d) for d in data]
        self.assertEqual([([1, 2, 3], [1, 1, 1, 1])], data_vectorized)
