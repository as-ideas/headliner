import shutil
import tempfile
import unittest
import os
from unittest.mock import Mock, call

from headliner.callbacks.model_checkpoint_callback import ModelCheckpointCallback


class TestModelCheckpointCallback(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix='TestModelCheckpointingCallback')

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_on_epoch_end(self):
        mock_summarizer = Mock()
        model_save_path = os.path.join(self.temp_dir, 'summarizer_save')
        model_checkoint_callback = ModelCheckpointCallback(file_path=model_save_path,
                                                           summarizer=mock_summarizer,
                                                           monitor='loss_val',
                                                           mode='min')

        logs = {'loss_val': 10}
        model_checkoint_callback.on_epoch_end(0, logs=logs)
        mock_summarizer.save.assert_called_with(model_save_path)
        logs = {'loss_val': 20}
        model_checkoint_callback.on_epoch_end(0, logs=logs)
        mock_summarizer.save.assert_called_with(model_save_path)
        logs = {'loss_val': 5}
        model_checkoint_callback.on_epoch_end(0, logs=logs)
        mock_summarizer.save.assert_has_calls([call(model_save_path), call(model_save_path)])
