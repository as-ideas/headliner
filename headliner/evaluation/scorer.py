import abc
import numpy as np
from typing import Dict, Union


class Scorer(abc.ABC):

    def __call__(self, prediction: [Dict[str, Union[str, np.array]]]) -> float:
        """
        Evaluates prediction.

        Args:
            prediction: Dictionary providing all information about a model prediction such as
            output string, logits etc.

        Returns: Prediction score as float.
        """

        raise NotImplementedError()