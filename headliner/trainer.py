import logging
import datetime
import yaml
from collections import Counter
from typing import Tuple, List, Iterable, Union, Callable, Dict
from keras_preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import TensorBoard, Callback
from headliner.callbacks.evaluation_callback import EvaluationCallback
from headliner.callbacks.model_checkpoint_callback import ModelCheckpointCallback
from headliner.callbacks.validation_callback import ValidationCallback
from headliner.embeddings import read_glove, embedding_to_matrix
from headliner.evaluation.scorer import Scorer
from headliner.losses import masked_crossentropy
from headliner.model.summarizer import Summarizer
from headliner.model.summarizer_attention import SummarizerAttention
from headliner.preprocessing.dataset_generator import DatasetGenerator
from headliner.preprocessing.bucket_generator import BucketGenerator
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.preprocessing.vectorizer import Vectorizer
from headliner.utils.logger import get_logger

START_TOKEN = '<start>'
END_TOKEN = '<end>'
OOV_TOKEN = '<unk>'


class Trainer:

    def __init__(self,
                 batch_size=16,
                 max_vocab_size=200000,
                 glove_path=None,
                 steps_per_epoch=500,
                 tensorboard_dir='/tmp/train_tens_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                 model_save_path='/tmp/summarizer_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
                 bucketing_buffer_size_batches=10000,
                 bucketing_batches_to_bucket=100,
                 logging_level=logging.INFO,
                 num_print_predictions=5,
                 steps_to_log=10,
                 preprocessor=None,
                 vectorizer=None) -> None:
        """
        Initializes the trainer.

        Args:
            batch_size: Size of mini-batches for stochastic gradient descent.
            max_vocab_size: Maximum number of unique tokens to consider for embeddings.
            glove_path: Path to glove embedding file.
            steps_per_epoch: Number of steps to train until callbacks are invoked.
            tensorboard_dir: Directory for saving tensorboard logs.
            model_save_path: Directory for saving the best model.
            bucketing_buffer_size_batches: Number of batches to buffer when bucketing sequences.
            bucketing_batches_to_bucket: Number of buffered batches from which sequences are collected for bucketing.
            logging_level: Level of logging to use, e.g. logging.INFO or logging.DEBUG.
            num_print_predictions: Number of sample predictions to print in each evaluation.
            steps_to_log: Number of steps to wait for logging output.
            preprocessor (optional): custom preprocessor, if None a standard preprocessor will be created.
            vectorizer (optional): custom vectorizer, if None a standard vectorizer will be created.
        """

        self.batch_size = batch_size
        self.max_vocab_size = max_vocab_size
        self.bucketing_buffer_size_batches = bucketing_buffer_size_batches
        self.bucketing_batches_to_bucket = bucketing_batches_to_bucket
        self.glove_path = glove_path
        self.steps_per_epoch = steps_per_epoch
        self.tensorboard_dir = tensorboard_dir
        self.model_save_path = model_save_path
        self.loss_function = masked_crossentropy
        self.dataset_generator = DatasetGenerator(self.batch_size)
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
        self.vectorizer = vectorizer

    @classmethod
    def from_config(cls, file_path, **kwargs):
        with open(file_path, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f)
            batch_size = cfg['batch_size']
            max_vocab_size = cfg['max_vocab_size']
            glove_path = cfg['glove_path']
            steps_per_epoch = cfg['steps_per_epoch']
            tensorboard_dir = cfg['tensorboard_dir']
            model_save_path = cfg['model_save_path']
            bucketing_buffer_size_batches = cfg['bucketing_buffer_size_batches']
            bucketing_batches_to_bucket = cfg['bucketing_batches_to_bucket']
            steps_to_log = cfg['steps_to_log']
            logging_level = logging.INFO
            logging_level_string = cfg['logging_level']
            if logging_level_string == 'debug':
                logging_level = logging.DEBUG
            elif logging_level_string == 'error':
                logging_level = logging.ERROR
            return Trainer(batch_size=batch_size,
                           max_vocab_size=max_vocab_size,
                           glove_path=glove_path,
                           steps_per_epoch=steps_per_epoch,
                           tensorboard_dir=tensorboard_dir,
                           model_save_path=model_save_path,
                           bucketing_buffer_size_batches=bucketing_buffer_size_batches,
                           bucketing_batches_to_bucket=bucketing_batches_to_bucket,
                           logging_level=logging_level,
                           steps_to_log=steps_to_log,
                           **kwargs)

    def train(self,
              summarizer: Union[Summarizer, SummarizerAttention],
              train_data: Iterable[Tuple[str, str]],
              val_data: Iterable[Tuple[str, str]] = None,
              num_epochs=2500,
              scorers: Dict[str, Scorer] = None,
              callbacks: List[Callback] = None) -> None:
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
            self.logger.info('training a bare model, initializing preprocessing...')
            self._init_model(summarizer, train_data)
        else:
            self.logger.info('training an already initialized model...')
        vectorize_train = self._vectorize_data(preprocessor=summarizer.preprocessor,
                                               vectorizer=summarizer.vectorizer,
                                               bucket_generator=self.bucket_generator)
        vectorize_val = self._vectorize_data(preprocessor=summarizer.preprocessor,
                                             vectorizer=summarizer.vectorizer,
                                             bucket_generator=None)
        train_dataset = self.dataset_generator(lambda: vectorize_train(train_data))
        val_dataset = self.dataset_generator(lambda: vectorize_val(val_data))

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
                ModelCheckpointCallback(file_path=self.model_save_path,
                                        summarizer=summarizer,
                                        monitor='loss_val',
                                        mode='min'),
                TensorBoard(log_dir=self.tensorboard_dir,
                            update_freq='epoch')
            ])

        logs = {}
        epoch_count, batch_count = 0, 0
        while epoch_count < num_epochs:
            for train_source_seq, train_target_seq in train_dataset.take(-1):
                batch_count += 1
                train_loss = summarizer.train_step(source_seq=train_source_seq,
                                                   target_seq=train_target_seq,
                                                   loss_function=self.loss_function)
                logs['loss'] = float(train_loss)
                if batch_count % self.steps_to_log == 0:
                    self.logger.info('epoch {epoch}, batch {batch}, logs: {logs}'.format(epoch=epoch_count,
                                                                                         batch=batch_count,
                                                                                         logs=logs))
                if batch_count % self.steps_per_epoch == 0:
                    for callback in train_callbacks:
                        callback.on_epoch_end(epoch_count, logs=logs)
                    epoch_count += 1
                    if epoch_count >= num_epochs:
                        break

            self.logger.info('finished iterating over dataset, total batches: {}'.format(batch_count))
            if batch_count == 0:
                raise ValueError('Iterating over the dataset yielded zero batches!')

    def _init_model(self,
                    summarizer: Union[Summarizer, SummarizerAttention],
                    train_data: Iterable[Tuple[str, str]]) -> None:

        if self.vectorizer is not None:
            summarizer.init_model(preprocessor=self.preprocessor,
                                  vectorizer=self.vectorizer,
                                  embedding_weights_encoder=None,
                                  embedding_weights_decoder=None)
        else:
            tokenizer_encoder, tokenizer_decoder = self._create_tokenizers(train_data)
            self.logger.info('vocab encoder: {vocab_enc}, vocab decoder: {vocab_dec}, start training loop...'.format(
                vocab_enc=len(tokenizer_encoder.word_index), vocab_dec=len(tokenizer_decoder.word_index)))
            vectorizer = Vectorizer(tokenizer_encoder, tokenizer_decoder)
            embedding_weights_encoder, embedding_weights_decoder = None, None
            if self.glove_path is not None:
                print('loading embedding from {}'.format(self.glove_path))
                embedding = read_glove(self.glove_path, summarizer.embedding_size)
                embedding_weights_encoder = embedding_to_matrix(embedding=embedding,
                                                                token_index=tokenizer_encoder.word_index,
                                                                embedding_dim=summarizer.embedding_size)
                embedding_weights_decoder = embedding_to_matrix(embedding=embedding,
                                                                token_index=tokenizer_decoder.word_index,
                                                                embedding_dim=summarizer.embedding_size)
                unknown_tokens_encoder = tokenizer_encoder.word_index.keys() - embedding.keys()
                unknown_tokens_decoder = tokenizer_decoder.word_index.keys() - embedding.keys()
                print('unknown vocab encoder: {vocab_enc}, decoder: {vocab_dec}'.format(
                    vocab_enc=len(unknown_tokens_encoder), vocab_dec=len(unknown_tokens_decoder)))
            summarizer.init_model(preprocessor=self.preprocessor,
                                  vectorizer=vectorizer,
                                  embedding_weights_encoder=embedding_weights_encoder,
                                  embedding_weights_decoder=embedding_weights_decoder)

    def _vectorize_data(self,
                        preprocessor: Preprocessor,
                        vectorizer: Vectorizer,
                        bucket_generator: BucketGenerator = None) -> Callable[[Iterable[Tuple[str, str]]],
                                                                              Iterable[Tuple[List[int], List[int]]]]:

        def vectorize(raw_data: Iterable[Tuple[str, str]]):
            data_preprocessed = (preprocessor(d) for d in raw_data)
            data_vectorized = (vectorizer(d) for d in data_preprocessed)
            if bucket_generator is None:
                return data_vectorized
            else:
                return self.bucket_generator(data_vectorized)

        return vectorize

    def _create_tokenizers(self,
                           train_data: Iterable[Tuple[str, str]]
                           ) -> Tuple[Tokenizer, Tokenizer]:

        counter_encoder = Counter()
        counter_decoder = Counter()
        train_text_encoder = (self.preprocessor(d)[0] for d in train_data)
        train_text_decoder = (self.preprocessor(d)[1] for d in train_data)
        for text_encoder in train_text_encoder:
            counter_encoder.update(text_encoder.split())
        for text_decoder in train_text_decoder:
            counter_decoder.update(text_decoder.split())
        tokens_encoder = {token_count[0] for token_count in counter_encoder.most_common(self.max_vocab_size)}
        tokens_decoder = {token_count[0] for token_count in counter_decoder.most_common(self.max_vocab_size)}
        tokens_encoder.update({self.preprocessor.start_token, self.preprocessor.end_token})
        tokens_decoder.update({self.preprocessor.start_token, self.preprocessor.end_token})
        tokenizer_encoder = Tokenizer(oov_token=OOV_TOKEN, filters='', lower=False)
        tokenizer_decoder = Tokenizer(oov_token=OOV_TOKEN, filters='', lower=False)
        tokenizer_encoder.fit_on_texts(sorted(list(tokens_encoder)))
        tokenizer_decoder.fit_on_texts(sorted(list(tokens_decoder)))
        return tokenizer_encoder, tokenizer_decoder
