import os
import shutil
import tempfile
import unittest

from keras_preprocessing.text import Tokenizer

from headliner.model.summarizer import Summarizer
from headliner.model.summarizer_attention import SummarizerAttention
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.preprocessing.vectorizer import Vectorizer


class TestSummarizer(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix='TestSummarizerAttentionTmp')

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_serde_happy_path(self) -> None:
        preprocessor = Preprocessor()
        tokenizer = Tokenizer()
        vectorizer = Vectorizer(tokenizer_encoder=tokenizer,
                                tokenizer_decoder=tokenizer,
                                max_output_len=8)
        summarizer = SummarizerAttention(lstm_size=10,
                                         embedding_size=10)
        summarizer.init_model(preprocessor=preprocessor,
                              vectorizer=vectorizer)
        save_dir = os.path.join(self.temp_dir, 'summarizer_serde_happy_path')
        summarizer.save(save_dir)
        summarizer_loaded = Summarizer.load(save_dir)
        self.assertEqual(10, summarizer_loaded.lstm_size)
        self.assertIsNotNone(summarizer_loaded.preprocessor)
        self.assertIsNotNone(summarizer_loaded.vectorizer)
        self.assertIsNotNone(summarizer_loaded.encoder)
        self.assertIsNotNone(summarizer_loaded.decoder)
        self.assertIsNotNone(summarizer_loaded.optimizer)
        self.assertEqual(8, summarizer_loaded.vectorizer.max_output_len)