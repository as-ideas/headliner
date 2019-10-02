import unittest
from unittest.mock import Mock

from headliner.callbacks.validation_callback import ValidationCallback


class TestValidationCallback(unittest.TestCase):

    def test_on_epoch_end(self):
        mock_summarizer = Mock()
        mock_summarizer.new_train_step.return_value = lambda input_seq, output_seq: 0.5
        mock_scorer_a, mock_scorer_b = Mock(), Mock()
        mock_scorer_a.return_value = 1
        mock_scorer_b.return_value = 2
        val_dataset_mock = Mock()
        val_dataset_mock.take.return_value = [(1, 2), (1, 2)]
        loss_function_mock = Mock()
        validation_callback = ValidationCallback(summarizer=mock_summarizer,
                                                 val_dataset=val_dataset_mock,
                                                 loss_function=loss_function_mock,
                                                 batch_size=1)
        logs = {}
        validation_callback.on_epoch_end(0, logs=logs)
        self.assertEqual({'loss_val'}, logs.keys())
        self.assertAlmostEqual(0.5, logs['loss_val'], places=10)
