import abc
from abc import abstractmethod
from typing import Callable, Dict, Union

import numpy as np
import tensorflow as tf

from headliner.preprocessing import Preprocessor, Vectorizer


class Summarizer(abc.ABC):

    def __init__(self):
        self.vectorizer: Union[Vectorizer, None] = None
        self.preprocessor: Union[Preprocessor, None] = None
        self.embedding_size: Union[int, None] = None

    @abstractmethod
    def init_model(self,
                   preprocessor: Preprocessor,
                   vectorizer: Vectorizer,
                   embedding_weights_encoder=None,
                   embedding_weights_decoder=None) -> None:
        """
        Initializes the model and provides necessary information for compilation.

        Args:
            preprocessor: Preprocessor object that preprocesses text for training and prediction.
            vectorizer: Vectorizer object that performs vectorization of the text.
            embedding_weights_encoder (optional): Matrix to initialize the encoder embedding.
            embedding_weights_decoder (optional): Matrix to initialize the decoder embedding.
        """

        pass

    @abstractmethod
    def predict(self, text: str) -> str:
        """
        Predicts summary of an input text.
        """

        pass

    @abstractmethod
    def predict_vectors(self, input_text: str, target_text: str) -> Dict[str, Union[str, np.array]]:
        """
        Predicts summary of an input text and outputs information needed for evaluation:
        output logits, input tokens, output tokens, predicted tokens, preprocessed text,
        attention alignment.

        Args:
            input_text: Text used as input for prediction.
            target_text: Text used for evaluation.

        Returns: Dictionary with prediction information such as
            preprocessed_text, logits, alignment, predicted_sequence, predicted_text.

        """

        pass

    @abstractmethod
    def new_train_step(self,
                       loss_function: Callable[[tf.Tensor], tf.Tensor],
                       batch_size: int,
                       apply_gradients=True) -> Callable[[tf.Tensor, tf.Tensor], float]:
        """
        Initializes the train_step function to train the model on batches of data.

        Args:
            loss_function: Loss function to perform backprop on.
            batch_size: Batch size to use for training.
            apply_gradients: Whether to apply the gradients, i.e.
                False if you want to validate the model on test data.

        Returns: Train step function that is applied to a batch and returns the loss.

        """

        pass

    @abstractmethod
    def save(self, out_path: str) -> None:
        """
        Saves the model to a file.

        Args:
            out_path: Path to directory for saving the model.

        """

        pass

    @staticmethod
    @abstractmethod
    def load(in_path: str):
        """
        Loads the model from a file.

        Args:
            in_path: Path to the model directory.

        Returns: Instance of the loaded summarizer.

        """

        pass
