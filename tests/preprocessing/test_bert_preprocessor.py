import unittest

from spacy.lang.en import English
from headliner.preprocessing.bert_preprocessor import BertPreprocessor


class TestBertPreprocessor(unittest.TestCase):

    def test_preprocessing(self):
        nlp = English()
        preprocessor = BertPreprocessor(nlp=nlp)
        data = ('I love my dog. He is the best. He eats and poops.', 'Me and my dog.')
        data_preprocessed = preprocessor(data)
        self.assertEqual(('[CLS] I love my dog. [SEP] [CLS] He is the best. [SEP] [CLS] He eats and poops. [SEP]',
                          '[CLS] Me and my dog. [SEP]'), data_preprocessed)
