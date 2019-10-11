from typing import Dict, Union

import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from headliner.evaluation.scorer import Scorer


class BleuScorer(Scorer):
    """
    Provides BLEU score for a model prediction.
    """

    def __init__(self, tokens_to_ignore=None, weights=(0.25, 0.25, 0.25, 0.25)) -> None:
        """
        Initializes the scorer.

        Args:
            tokens_to_ignore: Tokens to be removed before comparing input and output text.
            weights: Custom weights for 1,2,3,4 grams, e.g. (1, 0, 0, 0) will only measure 1-gram overlaps.
        """
        self.tokens_to_exclude = tokens_to_ignore or []
        self.weights = weights

    def __call__(self, prediction: [Dict[str, Union[str, np.array]]]) -> float:
        tokens_predicted = prediction['predicted_text'].split()
        tokens_output = prediction['preprocessed_text'][1].split()
        tokens_predicted_filtered = [t for t in tokens_predicted if t not in self.tokens_to_exclude]
        tokens_output_filtered = [t for t in tokens_output if t not in self.tokens_to_exclude]
        return sentence_bleu([tokens_output_filtered], tokens_predicted_filtered, weights=self.weights)
