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
        en_initial_states = self.summarizer.encoder.init_states(self.batch_size)
        val_loss, count_batches_val = 0, 0
        for test_source_seq, test_target_seq in self.val_dataset.take(-1):
            val_loss_batch = self.summarizer.train_step(source_seq=test_source_seq,
                                                        target_seq=test_target_seq,
                                                        en_initial_states=en_initial_states,
                                                        loss_function=self.loss_function,
                                                        apply_gradients=False)
            val_loss += val_loss_batch
            count_batches_val += 1
        logs['loss_val'] = float(val_loss / count_batches_val)
