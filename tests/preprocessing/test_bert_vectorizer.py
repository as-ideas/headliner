import unittest

from transformers import BertTokenizer

from headliner.preprocessing.bert_vectorizer import BertVectorizer
from headliner.preprocessing.keras_tokenizer import KerasTokenizer


class TestBertVectorizer(unittest.TestCase):

    def test_vectorize(self):
        data = ('[CLS] I love my dog. [SEP] [CLS] He is the best. [SEP]', '[CLS] Dog. [SEP]')
        tokenizer_encoder = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer_decoder = KerasTokenizer()
        tokenizer_decoder.fit([data[1]])
        vectorizer = BertVectorizer(tokenizer_encoder,
                                    tokenizer_decoder,
                                    max_input_len=50,
                                    max_output_len=3)

        data_vectorized = vectorizer(data)
        expected = ([101, 1045, 2293, 2026, 3899, 1012, 102, 101, 2002, 2003, 1996, 2190, 1012, 102],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 2, 3])
        self.assertEqual(expected, data_vectorized)
        input_decoded = vectorizer.decode_input(expected[0])
        expected = '[CLS] i love my dog. [SEP] [CLS] he is the best. [SEP]'
        self.assertEqual(expected, input_decoded)


