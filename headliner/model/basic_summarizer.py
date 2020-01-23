import os
import pickle
from typing import Callable, Dict, Union

import numpy as np
import tensorflow as tf

from headliner.model.basic_model import Encoder, Decoder
from headliner.model.summarizer import Summarizer
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.preprocessing.vectorizer import Vectorizer


class BasicSummarizer(Summarizer):

    def __init__(self, lstm_size=50, max_prediction_len=20, embedding_size=50, embedding_encoder_trainable=True,
                 embedding_decoder_trainable=True):

        super().__init__()
        self.lstm_size = lstm_size
        self.max_prediction_len = max_prediction_len
        self.embedding_size = embedding_size
        self.embedding_encoder_trainable = embedding_encoder_trainable
        self.embedding_decoder_trainable = embedding_decoder_trainable
        self.optimizer = BasicSummarizer._new_optimizer()
        self.encoder = None
        self.decoder = None
        self.embedding_shape_in = None
        self.embedding_shape_out = None

    def init_model(self,
                   preprocessor: Preprocessor,
                   vectorizer: Vectorizer,
                   embedding_weights_encoder=None,
                   embedding_weights_decoder=None) -> None:
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.embedding_shape_in = (self.vectorizer.encoding_dim, self.embedding_size)
        self.embedding_shape_out = (self.vectorizer.decoding_dim, self.embedding_size)
        self.encoder = Encoder(self.embedding_shape_in,
                               self.lstm_size,
                               embedding_trainable=self.embedding_encoder_trainable,
                               embedding_weights=embedding_weights_encoder)
        self.decoder = Decoder(self.embedding_shape_out,
                               self.lstm_size,
                               embedding_trainable=self.embedding_decoder_trainable,
                               embedding_weights=embedding_weights_decoder)
        self.encoder.compile(optimizer=self.optimizer)
        self.decoder.compile(optimizer=self.optimizer)

    def __getstate__(self):
        """ Prevents pickle from serializing encoder and decoder. """
        state = self.__dict__.copy()
        del state['encoder']
        del state['decoder']
        del state['optimizer']
        return state

    def predict(self, text: str) -> str:
        return self.predict_vectors(text, '')['predicted_text']

    def predict_vectors(self, input_text: str, target_text: str) -> Dict[str, Union[str, np.array]]:
        text_preprocessed = self.preprocessor((input_text, target_text))
        en_inputs, _ = self.vectorizer(text_preprocessed)
        en_initial_states = self.encoder.init_states(1)
        en_outputs = self.encoder(tf.constant([en_inputs]), en_initial_states)
        start_end_seq = self.vectorizer.encode_output(
            ' '.join([self.preprocessor.start_token, self.preprocessor.end_token]))
        de_start_index, de_end_index = start_end_seq[:1], start_end_seq[-1:]
        de_input = tf.constant([de_start_index])
        de_state_h, de_state_c = en_outputs[1:]
        output = {'preprocessed_text': text_preprocessed,
                  'logits': [],
                  'alignment': [],
                  'predicted_sequence': []}
        for _ in range(self.max_prediction_len):
            de_output, de_state_h, de_state_c = self.decoder(de_input, (de_state_h, de_state_c))
            de_input = tf.argmax(de_output, -1)
            pred_token_index = de_input.numpy()[0][0]
            if pred_token_index != 0:
                output['logits'].append(np.squeeze(de_output.numpy()))
                output['predicted_sequence'].append(pred_token_index)
                if pred_token_index == de_end_index:
                    break
        output['predicted_text'] = self.vectorizer.decode_output(output['predicted_sequence'])
        return output

    def new_train_step(self,
                       loss_function: Callable[[tf.Tensor], tf.Tensor],
                       batch_size: int,
                       apply_gradients=True) -> Callable[[tf.Tensor, tf.Tensor], float]:

        train_step_signature = [
            tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
        ]
        encoder = self.encoder
        decoder = self.decoder
        optimizer = self.optimizer

        @tf.function(input_signature=train_step_signature)
        def train_step(source_seq: tf.Tensor,
                       target_seq: tf.Tensor) -> float:
            en_initial_states = self.encoder.init_states(source_seq.get_shape()[0])
            with tf.GradientTape() as tape:
                en_outputs = encoder(source_seq, en_initial_states)
                en_states = en_outputs[1:]
                de_states = en_states
                de_outputs = decoder(target_seq[:, :-1], de_states)
                logits = de_outputs[0]
                loss = loss_function(target_seq[:, 1:], logits)
            if apply_gradients is True:
                variables = encoder.trainable_variables + decoder.trainable_variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
            return float(loss)

        return train_step

    def save(self, out_path):
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        summarizer_path = os.path.join(out_path, 'summarizer.pkl')
        encoder_path = os.path.join(out_path, 'encoder')
        decoder_path = os.path.join(out_path, 'decoder')
        with open(summarizer_path, 'wb+') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.encoder.save_weights(encoder_path, save_format='tf')
        self.decoder.save_weights(decoder_path, save_format='tf')

    @staticmethod
    def load(in_path):
        summarizer_path = os.path.join(in_path, 'summarizer.pkl')
        encoder_path = os.path.join(in_path, 'encoder')
        decoder_path = os.path.join(in_path, 'decoder')
        with open(summarizer_path, 'rb') as handle:
            summarizer = pickle.load(handle)
        summarizer.encoder = Encoder(summarizer.embedding_shape_in,
                                     summarizer.lstm_size,
                                     embedding_trainable=summarizer.embedding_encoder_trainable)
        summarizer.decoder = Decoder(summarizer.embedding_shape_out,
                                     summarizer.lstm_size,
                                     embedding_trainable=summarizer.embedding_decoder_trainable)
        optimizer = BasicSummarizer._new_optimizer()
        summarizer.encoder.compile(optimizer=optimizer)
        summarizer.decoder.compile(optimizer=optimizer)
        summarizer.encoder.load_weights(encoder_path)
        summarizer.decoder.load_weights(decoder_path)
        summarizer.optimizer = summarizer.encoder.optimizer
        return summarizer

    @staticmethod
    def _new_optimizer() -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.Adam()
