from typing import Dict, Callable, Iterable, Tuple

import tensorflow as tf

from headliner.model.summarizer import Summarizer
from headliner.utils.logger import get_logger


class EvaluationCallback(tf.keras.callbacks.Callback):
    """
    Callback for custom scoring methods.
    """

    def __init__(self,
                 summarizer: Summarizer,
                 scorers: Dict[str, Callable[[Dict], float]],
                 val_data: Iterable[Tuple[str, str]],
                 print_num_examples=5) -> None:
        """
        Initializes the Callback.

        Args:
            summarizer: Summarizer that predicts over the validation data.
            scorers: Dictionary of {scorer_name: scorer}, where each scorer maps a prediction to a score.
            val_data: Raw validation data to predict on.
            print_num_examples: Number of prediction examples to output for eyeballing the prediction quality.
        """

        super().__init__()
        self.summarizer = summarizer
        self.scorers = scorers
        self.val_data = val_data
        self.logger = get_logger(__name__)
        self.print_num_examples = print_num_examples

    def on_epoch_end(self, batch, logs=None) -> None:
        if logs is None:
            logs = {}
        val_scores = {score_name: 0. for score_name in self.scorers.keys()}
        count_val = 0
        for d in self.val_data:
            count_val += 1
            input_text, target_text = d
            prediction = self.summarizer.predict_vectors(input_text, target_text)
            if count_val <= self.print_num_examples:
                self.logger.info('\n(input) {} \n(target) {} \n(prediction) {}\n'.format(
                    prediction['preprocessed_text'][0],
                    prediction['preprocessed_text'][1],
                    prediction['predicted_text']
                ))
            elif len(self.scorers) == 0:
                break
            for score_name, scorer in self.scorers.items():
                score = scorer(prediction)
                val_scores[score_name] += score
        for score_name, score in val_scores.items():
            logs[score_name] = float(score / count_val)
