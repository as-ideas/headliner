import logging
import os
import unittest

from headliner.preprocessing.preprocessor import Preprocessor
from headliner.trainer import Trainer


class TestTrainer(unittest.TestCase):

    def test_init_from_config(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, 'resources/trainer_test_config.yaml')
        preprocessor = Preprocessor(start_token='<custom_start_token>')
        trainer = Trainer.from_config(config_path, preprocessor=preprocessor)
        self.assertEqual(1, trainer.batch_size)
        self.assertEqual(2, trainer.max_vocab_size)
        self.assertEqual('glove.txt', trainer.glove_path)
        self.assertEqual(4, trainer.steps_per_epoch)
        self.assertEqual('tensor_dir', trainer.tensorboard_dir)
        self.assertEqual('model_save_path', trainer.model_save_path)
        self.assertEqual(5, trainer.bucketing_buffer_size_batches)
        self.assertEqual(6, trainer.bucketing_batches_to_bucket)
        self.assertEqual(logging.DEBUG, trainer.logger.level)
        self.assertEqual('<custom_start_token>', trainer.preprocessor.start_token)

    def test_init(self) -> None:
        preprocessor = Preprocessor(start_token='<custom_start_token>', lower_case=False, hash_numbers=False)
        trainer = Trainer(batch_size=1,
                          max_vocab_size=2,
                          glove_path='glove.txt',
                          steps_per_epoch=4,
                          tensorboard_dir='tensor_dir',
                          model_save_path='model_save_path',
                          bucketing_buffer_size_batches=5,
                          bucketing_batches_to_bucket=6,
                          logging_level=logging.DEBUG,
                          preprocessor=preprocessor)

        self.assertEqual(1, trainer.batch_size)
        self.assertEqual(2, trainer.max_vocab_size)
        self.assertEqual('glove.txt', trainer.glove_path)
        self.assertEqual(4, trainer.steps_per_epoch)
        self.assertEqual('tensor_dir', trainer.tensorboard_dir)
        self.assertEqual('model_save_path', trainer.model_save_path)
        self.assertEqual(5, trainer.bucketing_buffer_size_batches)
        self.assertEqual(6, trainer.bucketing_batches_to_bucket)
        self.assertEqual(logging.DEBUG, trainer.logger.level)
        self.assertEqual('<custom_start_token>', trainer.preprocessor.start_token)
        self.assertEqual(False, trainer.preprocessor.lower_case)
        self.assertEqual(False, trainer.preprocessor.hash_numbers)

