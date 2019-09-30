import os
import shutil
import tempfile
import unittest

from keras_preprocessing.text import Tokenizer

from headliner.model.summarizer import Summarizer
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.preprocessing.vectorizer import Vectorizer


class TestSummarizer(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix='TestSummarizerTmp')

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_serde_happy_path(self) -> None:
        preprocessor = Preprocessor()
        tokenizer = Tokenizer()
        vectorizer = Vectorizer(tokenizer, tokenizer)
        summarizer = Summarizer(lstm_size=10,
                                max_head_len=10,
                                embedding_size=10)
        summarizer.init_model(preprocessor=preprocessor,
                              vectorizer=vectorizer)
        save_dir = os.path.join(self.temp_dir, 'summarizer_serde_happy_path')
        summarizer.save(save_dir)
        summarizer_loaded = Summarizer.load(save_dir)
        self.assertEqual(10, summarizer_loaded.lstm_size)
        self.assertEqual(10, summarizer_loaded.max_head_len)
        self.assertIsNotNone(summarizer_loaded.preprocessor)
        self.assertIsNotNone(summarizer_loaded.vectorizer)
        self.assertIsNotNone(summarizer_loaded.encoder)
        self.assertIsNotNone(summarizer_loaded.decoder)
        self.assertIsNotNone(summarizer_loaded.optimizer)
