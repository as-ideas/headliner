import abc
from typing import Dict, Union

import numpy as np


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
