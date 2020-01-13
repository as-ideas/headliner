import tensorflow as tf


class TensorboardCallback(tf.keras.callbacks.Callback):
    """
    Callback for validation loss.
    """

    def __init__(self,
                 log_dir: str) -> None:
        """
        Initializes the Callback.

        Args:
            log_dir: Tensorboard log directory to write to.
        """

        super().__init__()
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, batch, logs=None) -> None:
        if logs is not None:
            for key, val in logs.items():
                with self.summary_writer.as_default():
                    tf.summary.scalar(key, val, step=batch)
