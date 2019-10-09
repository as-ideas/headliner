import unittest
from headliner.preprocessing.keras_tokenizer import KerasTokenizer


class TestKerasTokenizer(unittest.TestCase):

    def test_keras_tokenizer(self):
        tokenizer = KerasTokenizer(filters='', lower=False, oov_token='<unk>')
        tokenizer.fit(['a b c d'])
        encoded = tokenizer.encode('a b e')
        self.assertEqual([2, 3, 1], encoded)
        decoded = tokenizer.decode(encoded)
        self.assertEqual('a b <unk>', decoded)
        self.assertEqual(5, tokenizer.vocab_size)
        self.assertEqual({'a', 'b', 'c', 'd', '<unk>'}, tokenizer.token_index.keys())
        self.assertEqual({1, 2, 3, 4, 5}, set(tokenizer.token_index.values()))

