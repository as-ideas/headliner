import tensorflow as tf
import logging
import os
import unittest
import numpy as np
from tensorflow.python.keras.callbacks import Callback
from headliner.model import SummarizerAttention
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.trainer import Trainer


class TestTrainer(unittest.TestCase):

    def setUp(self) -> None:
        tf.random.set_seed(42)
        np.random.seed(42)

    def test_init_from_config(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'resources/trainer_test_config.yaml')
        trainer = Trainer.from_config(config_path)
        self.assertEqual(9, trainer.max_output_len)
        self.assertEqual(1, trainer.batch_size)
        self.assertEqual(7, trainer.max_vocab_size_encoder)
        self.assertEqual(6, trainer.max_vocab_size_decoder)
        self.assertEqual('glove.txt', trainer.embedding_path_encoder)
        self.assertEqual(None, trainer.embedding_path_decoder)
        self.assertEqual(4, trainer.steps_per_epoch)
        self.assertEqual('tensor_dir', trainer.tensorboard_dir)
        self.assertEqual('model_save_path', trainer.model_save_path)
        self.assertTrue(trainer.use_bucketing)
        self.assertIsNone(trainer.shuffle_buffer_size)
        self.assertEqual(5, trainer.bucketing_buffer_size_batches)
        self.assertEqual(6, trainer.bucketing_batches_to_bucket)
        self.assertEqual(7, trainer.steps_to_log)
        self.assertEqual(logging.DEBUG, trainer.logger.level)

    def test_init(self) -> None:
        preprocessor = Preprocessor(start_token='<custom_start_token>', lower_case=False, hash_numbers=False)
        trainer = Trainer(max_output_len=9,
                          batch_size=1,
                          max_vocab_size_encoder=2,
                          max_vocab_size_decoder=3,
                          embedding_path_encoder='glove.txt',
                          steps_per_epoch=4,
                          tensorboard_dir='tensor_dir',
                          model_save_path='model_save_path',
                          shuffle_buffer_size=10,
                          bucketing_buffer_size_batches=5,
                          bucketing_batches_to_bucket=6,
                          steps_to_log=7,
                          logging_level=logging.DEBUG,
                          preprocessor=preprocessor)

        self.assertEqual(1, trainer.batch_size)
        self.assertEqual(2, trainer.max_vocab_size_encoder)
        self.assertEqual(3, trainer.max_vocab_size_decoder)
        self.assertEqual('glove.txt', trainer.embedding_path_encoder)
        self.assertIsNone(trainer.embedding_path_decoder)
        self.assertEqual(4, trainer.steps_per_epoch)
        self.assertEqual('tensor_dir', trainer.tensorboard_dir)
        self.assertEqual('model_save_path', trainer.model_save_path)
        self.assertFalse(trainer.use_bucketing)
        self.assertEqual(10, trainer.shuffle_buffer_size)
        self.assertEqual(5, trainer.bucketing_buffer_size_batches)
        self.assertEqual(6, trainer.bucketing_batches_to_bucket)
        self.assertEqual(7, trainer.steps_to_log)
        self.assertEqual(9, trainer.max_output_len)
        self.assertEqual(logging.DEBUG, trainer.logger.level)
        self.assertEqual('<custom_start_token>', trainer.preprocessor.start_token)
        self.assertEqual(False, trainer.preprocessor.lower_case)
        self.assertEqual(False, trainer.preprocessor.hash_numbers)

    def test_init_model(self) -> None:
        logging.basicConfig(level=logging.INFO)
        data = [('a b', 'a'), ('a b c', 'b')]
        summarizer = SummarizerAttention(lstm_size=16,
                                         embedding_size=10)
        trainer = Trainer(batch_size=2,
                          steps_per_epoch=10,
                          max_vocab_size_encoder=10,
                          max_vocab_size_decoder=10,
                          max_output_len=3)
        trainer.train(summarizer, data, num_epochs=1)
        # encoding dim and decoding dim are num unique tokens + 4 (pad, start, end, oov)
        self.assertIsNotNone(summarizer.vectorizer)
        self.assertEqual(7, summarizer.vectorizer.encoding_dim)
        self.assertEqual(6, summarizer.vectorizer.decoding_dim)
        self.assertEqual(3, summarizer.vectorizer.max_output_len)

    def test_train(self) -> None:

        class LogCallback(Callback):

            def __init__(self):
                super().__init__()

            def on_epoch_end(self, epoch, logs=None):
                self.logs = logs

        data = [('a b', 'a'), ('a b c', 'b')]

        summarizer = SummarizerAttention(lstm_size=16,
                                         embedding_size=10)
        log_callback = LogCallback()
        trainer = Trainer(batch_size=2,
                          steps_per_epoch=10,
                          max_vocab_size_encoder=10,
                          max_vocab_size_decoder=10,
                          max_output_len=3)

        trainer.train(summarizer,
                      data,
                      num_epochs=2,
                      callbacks=[log_callback])

        logs = log_callback.logs
        self.assertAlmostEqual(1.7162796020507813, logs['loss'], 6)
