import numpy as np
from typing import Dict, Union
from nltk.translate.bleu_score import sentence_bleu


class BleuScorer:
    """
    Provides BLEU score for a model prediction.
    """

    def __init__(self, tokens_to_ignore=None, weights=(0.25, 0.25, 0.25, 0.25)) -> None:
        self.tokens_to_exclude = tokens_to_ignore
        self.weights = weights

    def __call__(self, prediction: [Dict[str, Union[str, np.array]]]):
        tokens_predicted = prediction['predicted_text'].split()
        tokens_output = prediction['preprocessed_text'][1].split()
        tokens_predicted_filtered = [t for t in tokens_predicted if t not in self.tokens_to_exclude]
        tokens_output_filtered = [t for t in tokens_output if t not in self.tokens_to_exclude]
        return sentence_bleu([tokens_output_filtered], tokens_predicted_filtered, weights=self.weights)
