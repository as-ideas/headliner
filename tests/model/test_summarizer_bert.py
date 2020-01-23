import os
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from transformers import BertTokenizer

from headliner.losses import masked_crossentropy
from headliner.model.bert_summarizer import BertSummarizer
from headliner.preprocessing.bert_vectorizer import BertVectorizer
from headliner.preprocessing.keras_tokenizer import KerasTokenizer
from headliner.preprocessing.preprocessor import Preprocessor


class TestSummarizerBert(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(42)
        tf.random.set_seed(42)
        self.temp_dir = tempfile.mkdtemp(prefix='TestSummarizerBertTmp')

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_serde_happy_path(self) -> None:
        preprocessor = Preprocessor(start_token='[CLS]', end_token='[SEP]')

        tokenizer_encoder = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer_decoder = KerasTokenizer(oov_token='<unk>')
        tokenizer_decoder.fit(['a b c {} {}'.format(
            preprocessor.start_token, preprocessor.end_token)])
        vectorizer = BertVectorizer(tokenizer_encoder, tokenizer_decoder)
        summarizer = BertSummarizer(num_layers_encoder=1,
                                    num_layers_decoder=1,
                                    bert_embedding_encoder='bert-base-uncased',
                                    num_heads=2,
                                    max_prediction_len=3,
                                    embedding_size_encoder=768,
                                    embedding_size_decoder=10,
                                    embedding_encoder_trainable=False)
        summarizer.init_model(preprocessor=preprocessor,
                              vectorizer=vectorizer)

        # we need at least a train step to init the weights
        train_step = summarizer.new_train_step(masked_crossentropy, batch_size=1, apply_gradients=True)
        train_seq = tf.convert_to_tensor(np.array([[1, 1, 1]]), dtype=tf.int32)
        train_step(train_seq, train_seq, train_seq)

        save_dir = os.path.join(self.temp_dir, 'summarizer_serde_happy_path')
        summarizer.save(save_dir)
        summarizer_loaded = BertSummarizer.load(save_dir)
        self.assertEqual(1, summarizer_loaded.num_layers_encoder)
        self.assertEqual(1, summarizer_loaded.num_layers_decoder)
        self.assertEqual(2, summarizer_loaded.num_heads)
        self.assertEqual(3, summarizer_loaded.max_prediction_len)
        self.assertEqual(768, summarizer_loaded.embedding_size_encoder)
        self.assertEqual(10, summarizer_loaded.embedding_size_decoder)
        self.assertIsNotNone(summarizer_loaded.preprocessor)
        self.assertIsNotNone(summarizer_loaded.vectorizer)
        self.assertIsNotNone(summarizer_loaded.transformer)
        self.assertFalse(summarizer_loaded.transformer.encoder.embedding.trainable)
        self.assertTrue(summarizer_loaded.transformer.decoder.embedding.trainable)
        self.assertIsNotNone(summarizer_loaded.optimizer_encoder)
        self.assertIsNotNone(summarizer_loaded.optimizer_decoder)

        pred = summarizer.predict_vectors('a c', '')
        pred_loaded = summarizer_loaded.predict_vectors('a c', '')
        np.testing.assert_almost_equal(pred['logits'], pred_loaded['logits'], decimal=6)
