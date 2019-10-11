import tensorflow as tf

from headliner.model.summarizer import Summarizer


class ModelCheckpointCallback(tf.keras.callbacks.Callback):
    """
    Callback for checkpointing summarizer models.
    """

    def __init__(self,
                 file_path: str,
                 summarizer: Summarizer,
                 monitor='loss_val',
                 mode='min') -> None:

        """
        Initializes the Callback.

        Args:
            file_path: Path for saving the model (a directory). If existing, the model will be overwritten.
            summarizer: Summarizer to checkpoint.
            monitor: Name of the score monitor for improvements.
            mode: If set to 'min' a decrease of the monitored score is seen as an improvement, otherwise an increase.
        """

        super().__init__()
        self.file_path = file_path
        self.summarizer = summarizer
        self.monitor = monitor
        self.mode = mode
        self.best_score = None

    def on_epoch_end(self, batch, logs=None) -> None:
        if logs is None:
            logs = {}
        if self.file_path is None:
            return
        score = logs[self.monitor]
        score_is_better = False
        if self.best_score is None:
            score_is_better = True
        else:
            if self.mode == 'min' and score < self.best_score:
                score_is_better = True
            if self.mode == 'max' and score > self.best_score:
                score_is_better = True
        if score_is_better:
            print('{score_name} improved from {prev} to {current}, '
                  'saving summarizer to {path}'.format(score_name=self.monitor,
                                                       prev=self.best_score,
                                                       current=score,
                                                       path=self.file_path))
            self.best_score = score
            self.summarizer.save(self.file_path)
        else:
            print('{score_name} did not improve.'.format(score_name=self.monitor))
