import datetime
import logging
import os
import tempfile
from collections import Counter
from typing import Tuple, List, Iterable, Callable, Dict, Union

import tensorflow as tf
import yaml

from headliner.callbacks.evaluation_callback import EvaluationCallback
from headliner.callbacks.model_checkpoint_callback import ModelCheckpointCallback
from headliner.callbacks.tensorboard_callback import TensorboardCallback
from headliner.callbacks.validation_callback import ValidationCallback
from headliner.embeddings import read_embedding, embedding_to_matrix
from headliner.evaluation.scorer import Scorer
from headliner.losses import masked_crossentropy
from headliner.model.bert_summarizer import BertSummarizer
from headliner.model.summarizer import Summarizer
from headliner.preprocessing.bucket_generator import BucketGenerator
from headliner.preprocessing.dataset_generator import DatasetGenerator
from headliner.preprocessing.keras_tokenizer import KerasTokenizer
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.preprocessing.vectorizer import Vectorizer
from headliner.utils.logger import get_logger

START_TOKEN = '<start>'
END_TOKEN = '<end>'
OOV_TOKEN = '<unk>'


class Trainer:

    def __init__(self,
                 max_input_len=None,
                 max_output_len=None,
                 batch_size=16,
                 max_vocab_size_encoder=200000,
                 max_vocab_size_decoder=200000,
                 embedding_path_encoder=None,
                 embedding_path_decoder=None,
                 steps_per_epoch=500,
                 tensorboard_dir=None,
                 model_save_path=None,
                 shuffle_buffer_size=100000,
                 use_bucketing=False,
                 bucketing_buffer_size_batches=10000,
                 bucketing_batches_to_bucket=100,
                 logging_level=logging.INFO,
                 num_print_predictions=5,
                 steps_to_log=10,
                 preprocessor: Union[Preprocessor, None] = None) -> None:
        """
        Initializes the trainer.

        Args:
            max_input_len (output): Maximum length of input sequences, longer sequences will be truncated.
            max_output_len (output): Maximum length of output sequences, longer sequences will be truncated.
            batch_size: Size of mini-batches for stochastic gradient descent.
            max_vocab_size_encoder: Maximum number of unique tokens to consider for encoder embeddings.
            max_vocab_size_decoder: Maximum number of unique tokens to consider for decoder embeddings.
            embedding_path_encoder: Path to embedding file for the encoder.
            embedding_path_decoder: Path to embedding file for the decoder.
            steps_per_epoch: Number of steps to train until callbacks are invoked.
            tensorboard_dir: Directory for saving tensorboard logs.
            model_save_path: Directory for saving the best model.
            shuffle_buffer_size: Size of the buffer for shuffling the files before batching.
            use_bucketing: Whether to bucket the sequences by length to reduce the amount of padding.
            bucketing_buffer_size_batches: Number of batches to buffer when bucketing sequences.
            bucketing_batches_to_bucket: Number of buffered batches from which sequences are collected for bucketing.
            logging_level: Level of logging to use, e.g. logging.INFO or logging.DEBUG.
            num_print_predictions: Number of sample predictions to print in each evaluation.
            steps_to_log: Number of steps to wait for logging output.
            preprocessor (optional): custom preprocessor, if None a standard preprocessor will be created.
        """

        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.batch_size = batch_size
        self.max_vocab_size_encoder = max_vocab_size_encoder
        self.max_vocab_size_decoder = max_vocab_size_decoder
        self.bucketing_buffer_size_batches = bucketing_buffer_size_batches
        self.bucketing_batches_to_bucket = bucketing_batches_to_bucket
        self.embedding_path_encoder = embedding_path_encoder
        self.embedding_path_decoder = embedding_path_decoder
        self.steps_per_epoch = steps_per_epoch
        self.tensorboard_dir = tensorboard_dir
        self.model_save_path = model_save_path
        self.loss_function = masked_crossentropy
        self.use_bucketing = use_bucketing
        self.shuffle_buffer_size = None if use_bucketing else shuffle_buffer_size

        self.bucket_generator = None
        if use_bucketing:
            self.bucket_generator = BucketGenerator(element_length_function=lambda vecs: len(vecs[0]),
                                                    batch_size=self.batch_size,
                                                    buffer_size_batches=self.bucketing_buffer_size_batches,
                                                    batches_to_bucket=self.bucketing_batches_to_bucket,
                                                    shuffle=True,
                                                    seed=42)
        self.logger = get_logger(__name__)
        self.logger.setLevel(logging_level)
        self.num_print_predictions = num_print_predictions
        self.steps_to_log = steps_to_log
        self.preprocessor = preprocessor or Preprocessor(start_token=START_TOKEN, end_token=END_TOKEN)

    @classmethod
    def from_config(cls, file_path, **kwargs):
        with open(file_path, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f)
            batch_size = cfg['batch_size']
            max_vocab_size_encoder = cfg['max_vocab_size_encoder']
            max_vocab_size_decoder = cfg['max_vocab_size_decoder']
            glove_path_encoder = cfg['embedding_path_encoder']
            glove_path_decoder = cfg['embedding_path_decoder']
            steps_per_epoch = cfg['steps_per_epoch']
            tensorboard_dir = cfg['tensorboard_dir']
            model_save_path = cfg['model_save_path']
            use_bucketing = cfg['use_bucketing']
            shuffle_buffer_size = cfg['shuffle_buffer_size']
            bucketing_buffer_size_batches = cfg['bucketing_buffer_size_batches']
            bucketing_batches_to_bucket = cfg['bucketing_batches_to_bucket']
            steps_to_log = cfg['steps_to_log']
            logging_level = logging.INFO
            logging_level_string = cfg['logging_level']
            max_input_len = cfg['max_input_len']
            max_output_len = cfg['max_output_len']
            if logging_level_string == 'debug':
                logging_level = logging.DEBUG
            elif logging_level_string == 'error':
                logging_level = logging.ERROR
            return Trainer(batch_size=batch_size,
                           max_vocab_size_encoder=max_vocab_size_encoder,
                           max_vocab_size_decoder=max_vocab_size_decoder,
                           embedding_path_encoder=glove_path_encoder,
                           embedding_path_decoder=glove_path_decoder,
                           steps_per_epoch=steps_per_epoch,
                           tensorboard_dir=tensorboard_dir,
                           model_save_path=model_save_path,
                           use_bucketing=use_bucketing,
                           shuffle_buffer_size=shuffle_buffer_size,
                           bucketing_buffer_size_batches=bucketing_buffer_size_batches,
                           bucketing_batches_to_bucket=bucketing_batches_to_bucket,
                           logging_level=logging_level,
                           steps_to_log=steps_to_log,
                           max_input_len=max_input_len,
                           max_output_len=max_output_len,
                           **kwargs)

    def train(self,
              summarizer: Summarizer,
              train_data: Iterable[Tuple[str, str]],
              val_data: Iterable[Tuple[str, str]] = None,
              num_epochs=2500,
              scorers: Dict[str, Scorer] = None,
              callbacks: List[tf.keras.callbacks.Callback] = None) -> None:
        """
        Trains a summarizer or resumes training of a previously initialized summarizer.

        Args:
            summarizer: Model to train, can be either a freshly created model or a loaded model.
            train_data: Data to train the model on.
            val_data (optional): Validation data.
            num_epochs: Number of epochs to train.
            scorers (optional): Dictionary with {score_name, scorer} to add validation scores to the logs.
            callbacks (optional): Additional custom callbacks.
        """
        if summarizer.preprocessor is None or summarizer.vectorizer is None:
            self.logger.info('training a bare model, preprocessing data to init model...')
            self._init_model(summarizer, train_data)
        else:
            self.logger.info('training an already initialized model...')
        vectorize_train = self._vectorize_data(preprocessor=summarizer.preprocessor,
                                               vectorizer=summarizer.vectorizer,
                                               bucket_generator=self.bucket_generator)
        vectorize_val = self._vectorize_data(preprocessor=summarizer.preprocessor,
                                             vectorizer=summarizer.vectorizer,
                                             bucket_generator=None)
        train_gen, val_gen = self._create_dataset_generators(summarizer)
        train_dataset = train_gen(lambda: vectorize_train(train_data))
        val_dataset = val_gen(lambda: vectorize_val(val_data))

        train_callbacks = callbacks or []
        if val_data is not None:
            train_callbacks.extend([
                EvaluationCallback(summarizer=summarizer,
                                   scorers=scorers or {},
                                   val_data=val_data,
                                   print_num_examples=self.num_print_predictions),
                ValidationCallback(summarizer=summarizer,
                                   val_dataset=val_dataset,
                                   loss_function=self.loss_function,
                                   batch_size=self.batch_size),
            ])
        loss_monitor = 'loss_val' if val_data is not None else 'loss'
        train_callbacks.append(
            ModelCheckpointCallback(file_path=self.model_save_path,
                                    summarizer=summarizer,
                                    monitor=loss_monitor,
                                    mode='min'))

        if self.tensorboard_dir is not None:
            tb_callback = TensorboardCallback(log_dir=self.tensorboard_dir)
            train_callbacks.append(tb_callback)
        logs = {}
        epoch_count, batch_count, train_losses = 0, 0, []
        train_step = summarizer.new_train_step(self.loss_function,
                                               self.batch_size,
                                               apply_gradients=True)
        while epoch_count < num_epochs:
            for train_batch in train_dataset.take(-1):
                batch_count += 1
                current_loss = train_step(*train_batch)
                train_losses.append(current_loss)
                logs['loss'] = float(sum(train_losses)) / len(train_losses)
                if batch_count % self.steps_to_log == 0:
                    self.logger.info('epoch {epoch}, batch {batch}, '
                                     'logs: {logs}'.format(epoch=epoch_count,
                                                           batch=batch_count,
                                                           logs=logs))
                if batch_count % self.steps_per_epoch == 0:
                    train_losses.clear()
                    for callback in train_callbacks:
                        callback.on_epoch_end(epoch_count, logs=logs)
                    epoch_count += 1
                    if epoch_count >= num_epochs:
                        break

            self.logger.info('finished iterating over dataset, total batches: {}'.format(batch_count))
            if batch_count == 0:
                raise ValueError('Iterating over the dataset yielded zero batches!')

    def _init_model(self,
                    summarizer: Summarizer,
                    train_data: Iterable[Tuple[str, str]]) -> None:

        tokenizer_encoder, tokenizer_decoder = self._create_tokenizers(train_data)
        self.logger.info('vocab encoder: {vocab_enc}, vocab decoder: {vocab_dec}'.format(
            vocab_enc=tokenizer_encoder.vocab_size, vocab_dec=tokenizer_decoder.vocab_size))
        vectorizer = Vectorizer(tokenizer_encoder,
                                tokenizer_decoder,
                                max_input_len=self.max_input_len,
                                max_output_len=self.max_output_len)
        embedding_weights_encoder, embedding_weights_decoder = None, None

        if self.embedding_path_encoder is not None:
            self.logger.info('loading encoder embedding from {}'.format(self.embedding_path_encoder))
            embedding = read_embedding(self.embedding_path_encoder, summarizer.embedding_size)
            embedding_weights_encoder = embedding_to_matrix(embedding=embedding,
                                                            token_index=tokenizer_encoder.token_index,
                                                            embedding_dim=summarizer.embedding_size)
        if self.embedding_path_decoder is not None:
            self.logger.info('loading decoder embedding from {}'.format(self.embedding_path_decoder))
            embedding = read_embedding(self.embedding_path_decoder, summarizer.embedding_size)
            embedding_weights_decoder = embedding_to_matrix(embedding=embedding,
                                                            token_index=tokenizer_decoder.token_index,
                                                            embedding_dim=summarizer.embedding_size)
        summarizer.init_model(preprocessor=self.preprocessor,
                              vectorizer=vectorizer,
                              embedding_weights_encoder=embedding_weights_encoder,
                              embedding_weights_decoder=embedding_weights_decoder)

    def _vectorize_data(self,
                        preprocessor: Preprocessor,
                        vectorizer: Vectorizer,
                        bucket_generator: BucketGenerator = None) \
            -> Callable[[Iterable[Tuple[str, str]]],
                        Iterable[Tuple[List[int], List[int]]]]:

        def vectorize(raw_data: Iterable[Tuple[str, str]]):
            data_preprocessed = (preprocessor(d) for d in raw_data)
            data_vectorized = (vectorizer(d) for d in data_preprocessed)
            if bucket_generator is None:
                return data_vectorized
            else:
                return bucket_generator(data_vectorized)

        return vectorize

    def _create_tokenizers(self,
                           train_data: Iterable[Tuple[str, str]]
                           ) -> Tuple[KerasTokenizer, KerasTokenizer]:

        self.logger.info('fitting tokenizers...')
        counter_encoder = Counter()
        counter_decoder = Counter()
        train_preprocessed = (self.preprocessor(d) for d in train_data)
        for text_encoder, text_decoder in train_preprocessed:
            counter_encoder.update(text_encoder.split())
            counter_decoder.update(text_decoder.split())
        tokens_encoder = {token_count[0] for token_count
                          in counter_encoder.most_common(self.max_vocab_size_encoder)}
        tokens_decoder = {token_count[0] for token_count
                          in counter_decoder.most_common(self.max_vocab_size_decoder)}
        tokens_encoder.update({self.preprocessor.start_token, self.preprocessor.end_token})
        tokens_decoder.update({self.preprocessor.start_token, self.preprocessor.end_token})
        tokenizer_encoder = KerasTokenizer(oov_token=OOV_TOKEN, lower=False, filters='')
        tokenizer_decoder = KerasTokenizer(oov_token=OOV_TOKEN, lower=False, filters='')
        tokenizer_encoder.fit(sorted(list(tokens_encoder)))
        tokenizer_decoder.fit(sorted(list(tokens_decoder)))
        return tokenizer_encoder, tokenizer_decoder

    def _create_dataset_generators(self, summarizer):
        data_rank = 3 if isinstance(summarizer, BertSummarizer) else 2
        train_gen = DatasetGenerator(batch_size=self.batch_size,
                                     shuffle_buffer_size=self.shuffle_buffer_size,
                                     rank=data_rank)
        val_gen = DatasetGenerator(batch_size=self.batch_size,
                                   shuffle_buffer_size=None,
                                   rank=data_rank)
        return train_gen, val_gen
