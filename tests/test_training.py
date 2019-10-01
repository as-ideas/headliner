import unittest

import numpy as np
import tensorflow as tf
from keras_preprocessing.text import Tokenizer

from headliner.losses import masked_crossentropy
from headliner.model.summarizer import Summarizer
from headliner.model.summarizer_attention import SummarizerAttention
from headliner.preprocessing.dataset_generator import DatasetGenerator
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.preprocessing.vectorizer import Vectorizer


class TestTraining(unittest.TestCase):

    def setUp(self) -> None:
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_training(self) -> None:
        data = [('a b', 'c'), ('a b c', 'd')]
        tokenizer_encoder = Tokenizer(filters='')
        tokenizer_decoder = Tokenizer(filters='')
        tokenizer_encoder.fit_on_texts(['a b c <start> <end>'])
        tokenizer_decoder.fit_on_texts(['c d <start> <end>'])
        vectorizer = Vectorizer(tokenizer_encoder, tokenizer_decoder)
        preprocessor = Preprocessor()
        batch_generator = DatasetGenerator(2)
        data_prep = [preprocessor(d) for d in data]
        data_vecs = [vectorizer(d) for d in data_prep]
        dataset = batch_generator(lambda: data_vecs)

        summarizer_attention = SummarizerAttention(lstm_size=10, embedding_size=10)

        summarizer_attention.init_model(preprocessor=preprocessor,
                                        vectorizer=vectorizer,
                                        embedding_weights_encoder=None,
                                        embedding_weights_decoder=None)

        summarizer = Summarizer(lstm_size=10, embedding_size=10)

        summarizer.init_model(preprocessor=preprocessor,
                              vectorizer=vectorizer,
                              embedding_weights_encoder=None,
                              embedding_weights_decoder=None)

        loss_func = masked_crossentropy

        loss_attention = 0
        for e in range(0, 10):
            for source_seq, target_seq in dataset.take(-1):
                loss_attention = summarizer_attention.train_step(source_seq,
                                                                 target_seq,
                                                                 loss_func)

        self.assertAlmostEqual(1.5810251235961914, float(loss_attention), 10)
        output_attention = summarizer_attention.predict_vectors('a c', '')
        expected_first_logits = np.array([-0.069454,  0.00272,  0.007199, -0.039547,  0.014357])
        np.testing.assert_allclose(expected_first_logits, output_attention['logits'][0], atol=1e-6)
        self.assertEqual('a c', output_attention['preprocessed_text'][0])
        self.assertEqual('<end>', output_attention['predicted_text'])

        loss = 0
        for e in range(0, 10):
            for source_seq, target_seq in dataset.take(-1):
                loss = summarizer.train_step(source_seq,
                                             target_seq,
                                             loss_func)

        self.assertAlmostEqual(1.5771859884262085, float(loss), 10)
        output = summarizer.predict_vectors('a c', '')
        expected_first_logits = np.array([-0.03838864, 0.01226684, 0.01055636, -0.05209339, 0.02549592])
        np.testing.assert_allclose(expected_first_logits, output['logits'][0], atol=1e-6)
        self.assertEqual('a c', output['preprocessed_text'][0])
        self.assertEqual('<end>', output['predicted_text'])
