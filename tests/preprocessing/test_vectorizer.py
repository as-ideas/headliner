import unittest

from keras_preprocessing.text import Tokenizer

from headliner.preprocessing.vectorizer import Vectorizer


class TestPreprocessor(unittest.TestCase):

    def test_vectorize(self):
        text_encoder = 'a b c'
        text_decoder = 'd'
        data = [(text_encoder, text_decoder)]
        tokenizer_encoder = Tokenizer()
        tokenizer_decoder = Tokenizer()
        tokenizer_encoder.fit_on_texts([text_encoder])
        tokenizer_decoder.fit_on_texts([text_decoder])
        vectorizer = Vectorizer(tokenizer_encoder, tokenizer_decoder)
        data_vectorized = [vectorizer(d) for d in data]
        self.assertEqual([([1, 2, 3], [1])], data_vectorized)
