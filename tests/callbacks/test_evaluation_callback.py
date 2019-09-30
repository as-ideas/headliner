import unittest
from unittest.mock import Mock

from headliner.callbacks.evaluation_callback import EvaluationCallback


class TestEvaluationCallback(unittest.TestCase):

    def test_on_epoch_end(self):
        mock_summarizer = Mock()
        mock_summarizer.predict_vectors.return_value = None
        mock_scorer_a, mock_scorer_b = Mock(), Mock()
        mock_scorer_a.return_value = 1
        mock_scorer_b.return_value = 2
        val_data = [('a', 'b'), ('c', 'd')]
        evaluation_callback = EvaluationCallback(summarizer=mock_summarizer,
                                                 scorers={'mock_score_a': mock_scorer_a,
                                                          'mock_score_b': mock_scorer_b},
                                                 val_data=val_data,
                                                 print_num_examples=0)
        logs = {}
        evaluation_callback.on_epoch_end(0, logs=logs)
        self.assertEqual({'mock_score_a', 'mock_score_b'}, logs.keys())
        self.assertAlmostEqual(1.0, logs['mock_score_a'], places=10)
        self.assertAlmostEqual(2.0, logs['mock_score_b'], places=10)
