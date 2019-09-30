import unittest

from headliner.preprocessing.preprocessor import Preprocessor


class TestStringNormalizer(unittest.TestCase):

    def test_preprocessing(self):
        preprocessor = Preprocessor()
        data = (('First text!', 'first head'), ('2-nd täxt', 'Second head'))
        data_preprocessed = [preprocessor(d) for d in data]
        self.assertEqual(('first text !', '<start> first head <end>'), data_preprocessed[0])
        self.assertEqual(('#-nd täxt', '<start> second head <end>'), data_preprocessed[1])

        preprocessor = Preprocessor(start_token='<start>',
                                    end_token='<end>',
                                    lower_case=True,
                                    hash_numbers=False)
        data_preprocessed = [preprocessor(d) for d in data]
        self.assertEqual(('2-nd täxt', '<start> second head <end>'), data_preprocessed[1])

        preprocessor = Preprocessor(start_token='<start>',
                                    end_token='<end>',
                                    lower_case=False,
                                    hash_numbers=True)
        data_preprocessed = [preprocessor(d) for d in data]
        self.assertEqual(('#-nd täxt', '<start> Second head <end>'), data_preprocessed[1])

