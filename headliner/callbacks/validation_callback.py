from typing import Callable

import tensorflow as tf

from headliner.model.summarizer import Summarizer


class ValidationCallback(tf.keras.callbacks.Callback):
    """
    Callback for validation loss.
    """

    def __init__(self,
                 summarizer: Summarizer,
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
        self.train_step = summarizer.new_train_step(self.loss_function,
                                                    self.batch_size,
                                                    apply_gradients=False)

    def on_epoch_end(self, batch, logs=None) -> None:
        if logs is None:
            logs = {}
        val_loss, count_batches_val = 0, 0
        for batch in self.val_dataset.take(-1):
            val_loss_batch = self.train_step(*batch)
            val_loss += val_loss_batch
            count_batches_val += 1
        if count_batches_val == 0:
            raise ValueError('Tried to validate on empty validation dataset, possibly due to batch size '
                             'exceeding validation data size.')
        logs['loss_val'] = float(val_loss / count_batches_val)
