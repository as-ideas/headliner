import unittest

import numpy as np
import tensorflow as tf

from headliner.losses import masked_crossentropy
from headliner.model.basic_summarizer import BasicSummarizer
from headliner.model.attention_summarizer import AttentionSummarizer
from headliner.model.transformer_summarizer import TransformerSummarizer
from headliner.preprocessing.dataset_generator import DatasetGenerator
from headliner.preprocessing.keras_tokenizer import KerasTokenizer
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.preprocessing.vectorizer import Vectorizer


class TestTraining(unittest.TestCase):

    def setUp(self) -> None:
        tf.random.set_seed(42)
        np.random.seed(42)
        self.data = [('a b', 'c'), ('a b c', 'd')]
        tokenizer_encoder = KerasTokenizer(lower=False, filters='')
        tokenizer_decoder = KerasTokenizer(lower=False, filters='')
        tokenizer_encoder.fit(['a b c <start> <end>'])
        tokenizer_decoder.fit(['c d <start> <end>'])
        self.vectorizer = Vectorizer(tokenizer_encoder=tokenizer_encoder,
                                     tokenizer_decoder=tokenizer_decoder,
                                     max_output_len=3)
        self.preprocessor = Preprocessor()
        batch_generator = DatasetGenerator(2)
        data_prep = [self.preprocessor(d) for d in self.data]
        data_vecs = [self.vectorizer(d) for d in data_prep]
        self.dataset = batch_generator(lambda: data_vecs)
        self.loss_func = masked_crossentropy

    def test_training_summarizer_attention(self) -> None:
        attention_summarizer = AttentionSummarizer(lstm_size=10,
                                                   embedding_size=10)
        attention_summarizer.init_model(preprocessor=self.preprocessor,
                                        vectorizer=self.vectorizer,
                                        embedding_weights_encoder=None,
                                        embedding_weights_decoder=None)
        loss_attention = 0
        train_step = attention_summarizer.new_train_step(loss_function=self.loss_func,
                                                         batch_size=2)
        for _ in range(10):
            for source_seq, target_seq in self.dataset.take(-1):
                loss_attention = train_step(source_seq, target_seq)
                print(str(loss_attention))

        self.assertAlmostEqual(1.577033519744873, float(loss_attention), 5)
        output_attention = attention_summarizer.predict_vectors('a c', '')
        expected_first_logits = np.array([-0.077805,  0.012667,  0.021359, -0.04872,  0.014989])
        np.testing.assert_allclose(expected_first_logits, output_attention['logits'][0], atol=1e-6)
        self.assertEqual('<start> a c <end>', output_attention['preprocessed_text'][0])
        self.assertEqual('d <end>', output_attention['predicted_text'])

    def test_training_summarizer_basic(self) -> None:
        basic_summarizer = BasicSummarizer(lstm_size=10,
                                     embedding_size=10)
        basic_summarizer.init_model(preprocessor=self.preprocessor,
                              vectorizer=self.vectorizer,
                              embedding_weights_encoder=None,
                              embedding_weights_decoder=None)
        loss = 0
        train_step = basic_summarizer.new_train_step(loss_function=self.loss_func,
                                               batch_size=2)
        for e in range(0, 10):
            for source_seq, target_seq in self.dataset.take(-1):
                loss = train_step(source_seq, target_seq)

        self.assertAlmostEqual(1.5850255489349365, float(loss), 5)
        output = basic_summarizer.predict_vectors('a c', '')
        expected_first_logits = np.array([-0.00621 ,  0.007277,  0.015851, -0.034298,  0.044253])
        np.testing.assert_allclose(expected_first_logits, output['logits'][0], atol=1e-6)
        self.assertEqual('<start> a c <end>', output['preprocessed_text'][0])
        self.assertEqual('<end>', output['predicted_text'])

    def test_training_summarizer_transformer(self):
        transformer_summarizer = TransformerSummarizer(num_heads=1,
                                                       num_layers=1,
                                                       feed_forward_dim=20,
                                                       embedding_size=10,
                                                       dropout_rate=0,
                                                       max_prediction_len=3)
        transformer_summarizer.init_model(preprocessor=self.preprocessor,
                                          vectorizer=self.vectorizer,
                                          embedding_weights_encoder=None,
                                          embedding_weights_decoder=None)
        loss_transformer = 0
        train_step = transformer_summarizer.new_train_step(loss_function=self.loss_func,
                                                           batch_size=2)
        for e in range(0, 10):
            for source_seq, target_seq in self.dataset.take(-1):
                loss_transformer = train_step(source_seq, target_seq)
                print(str(loss_transformer))

        self.assertAlmostEqual(1.3421446084976196, float(loss_transformer), 5)
        output_transformer = transformer_summarizer.predict_vectors('a c', '')
        expected_first_logits = np.array([-0.514366,  1.416978, -0.679771, -0.488442, -0.022602])
        np.testing.assert_allclose(expected_first_logits, output_transformer['logits'][0], atol=1e-6)
        self.assertEqual('<start> a c <end>', output_transformer['preprocessed_text'][0])
        self.assertEqual('c c c', output_transformer['predicted_text'])
