import tensorflow as tf
from typing import Union, Callable
from headliner.model.summarizer import Summarizer
from headliner.model.summarizer_attention import SummarizerAttention


class ValidationCallback(tf.keras.callbacks.Callback):
    """
    Callback for validation loss.
    """

    def __init__(self,
                 summarizer: Union[Summarizer, SummarizerAttention],
                 val_dataset: tf.data.Dataset,
                 loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 batch_size: int) -> None:
        """
        Initializes the Callback.

        Args:
            summarizer: Summarizer to validate.
            val_dataset: Validation dataset to validate the model on.
            loss_function: Loss function to apply to calculate the validation score.
            batch_size: Batch size of the validation dataset, needed for initializing the model.
        """

        super().__init__()
        self.batch_size = batch_size
        self.summarizer = summarizer
        self.loss_function = loss_function
        self.val_dataset = val_dataset

    def on_epoch_end(self, batch, logs=None) -> None:
        if logs is None:
            logs = {}
        val_loss, count_batches_val = 0, 0
        for test_source_seq, test_target_seq in self.val_dataset.take(-1):
            val_loss_batch = self.summarizer.train_step(source_seq=test_source_seq,
                                                        target_seq=test_target_seq,
                                                        loss_function=self.loss_function,
                                                        apply_gradients=False)
            val_loss += val_loss_batch
            count_batches_val += 1
        if count_batches_val == 0:
            raise ValueError('Tried to validate on empty validation dataset, possibly due to batch size '
                             'exceeding validation data size.')
        logs['loss_val'] = float(val_loss / count_batches_val)
