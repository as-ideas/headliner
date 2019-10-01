import os
import pickle
import numpy as np
import tensorflow as tf
from typing import Tuple, Callable, Dict, Union
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.preprocessing.vectorizer import Vectorizer


class Encoder(tf.keras.Model):

    def __init__(self,
                 embedding_shape: Tuple[int, int],
                 lstm_size=50,
                 embedding_weights=None) -> None:
        super(Encoder, self).__init__()
        vocab_size, vec_dim = embedding_shape
        if embedding_weights is not None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, vec_dim, weights=[embedding_weights],
                                                       trainable=False)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, vec_dim, trainable=True)
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True, go_backwards=False)
        self.lstm_size = lstm_size

    def call(self,
             sequence: tf.Tensor,
             states: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)
        return output, state_h, state_c

    def init_states(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.zeros([batch_size, self.lstm_size]), \
               tf.zeros([batch_size, self.lstm_size])


class LuongAttention(tf.keras.Model):

    def __init__(self, rnn_size):
        super(LuongAttention, self).__init__()
        self.wa = tf.keras.layers.Dense(rnn_size)

    def call(self, decoder_output, encoder_output):
        score = tf.matmul(decoder_output, self.wa(encoder_output), transpose_b=True)
        alignment = tf.nn.softmax(score, axis=2)
        context = tf.matmul(alignment, encoder_output)
        return context, alignment


class Decoder(tf.keras.Model):

    def __init__(self,
                 embedding_shape: Tuple[int, int],
                 lstm_size=50,
                 embedding_weights=None) -> None:
        super(Decoder, self).__init__()
        self.lstm_size = lstm_size
        vocab_size, vec_dim = embedding_shape
        if embedding_weights is not None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, vec_dim, weights=[embedding_weights],
                                                       trainable=False)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size, vec_dim, trainable=True)
        self.lstm_size = lstm_size
        self.attention = LuongAttention(lstm_size)
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True)
        self.wc = tf.keras.layers.Dense(lstm_size, activation='tanh')
        self.ws = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, state, encoder_output):
        embed = self.embedding(sequence)
        lstm_out, state_h, state_c = self.lstm(embed, initial_state=state)
        context, alignment = self.attention(lstm_out, encoder_output)
        lstm_out = tf.concat([tf.squeeze(context, 1), tf.squeeze(lstm_out, 1)], 1)
        lstm_out = self.wc(lstm_out)
        logits = self.ws(lstm_out)
        return logits, state_h, state_c, alignment

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.lstm_size]),
                tf.zeros([batch_size, self.lstm_size]))


class SummarizerAttention:

    def __init__(self,
                 lstm_size=50,
                 max_head_len=20,
                 embedding_size=50):
        self.lstm_size = lstm_size
        self.max_head_len = max_head_len
        self.embedding_size = embedding_size
        self.preprocessor = None
        self.vectorizer = None
        self.encoder = None
        self.decoder = None
        self.optimizer = None
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
        self.encoder = Encoder(self.embedding_shape_in, self.lstm_size, embedding_weights=embedding_weights_encoder)
        self.decoder = Decoder(self.embedding_shape_out, self.lstm_size, embedding_weights=embedding_weights_decoder)
        self.optimizer = SummarizerAttention._new_optimizer()
        self.encoder.compile(optimizer=self.optimizer)
        self.decoder.compile(optimizer=self.optimizer)

    def __getstate__(self):
        """ Prevents pickle from serializing encoder and decoder """
        state = self.__dict__.copy()
        del state['encoder']
        del state['decoder']
        del state['optimizer']
        return state

    def predict(self, text: str) -> str:
        """
        Predicts summary of an input text.
        """

        return self.predict_vectors(text, '')['predicted_text']

    def predict_vectors(self, input_text: str, target_text: str) -> Dict[str, Union[str, np.array]]:
        """
        Predicts summary of an input text and outputs information needed for evaluation:
        output logits, input tokens, output tokens, predicted tokens, preprocessed text,
        attention alignment.
        """

        text_preprocessed = self.preprocessor((input_text, target_text))
        en_inputs, de_inputs = self.vectorizer(text_preprocessed)
        en_initial_states = self.encoder.init_states(1)
        en_outputs = self.encoder(tf.constant([en_inputs]), en_initial_states)
        _, de_start_index = self.vectorizer(('', self.preprocessor.start_token))
        _, de_end_index = self.vectorizer(('', self.preprocessor.end_token))
        de_input = tf.constant([de_start_index])
        de_state_h, de_state_c = en_outputs[1:]
        output = {'preprocessed_text': text_preprocessed,
                  'logits': [],
                  'alignment': [],
                  'predicted_sequence': []}
        for _ in range(self.max_head_len):
            de_output, de_state_h, de_state_c, alignment = self.decoder(de_input, (de_state_h, de_state_c),
                                                                        en_outputs[0])
            de_input = tf.expand_dims(tf.argmax(de_output, -1), 0)
            pred_token_index = de_input.numpy()[0][0]
            if pred_token_index != 0:
                output['logits'].append(np.squeeze(de_output.numpy()))
                output['alignment'].append(np.squeeze(alignment.numpy()))
                output['predicted_sequence'].append(pred_token_index)
                if pred_token_index == de_end_index:
                    break
        output['predicted_text'] = self.vectorizer.decode_output(output['predicted_sequence'])
        return output

    def train_step(self,
                   source_seq: tf.Tensor,
                   target_seq: tf.Tensor,
                   loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                   apply_gradients=True) -> float:

        loss = 0
        en_initial_states = self.encoder.init_states(source_seq.get_shape()[0])
        with tf.GradientTape() as tape:
            en_outputs = self.encoder(source_seq, en_initial_states)
            en_states = en_outputs[1:]
            de_state_h, de_state_c = en_states
            for i in range(target_seq.shape[1] - 1):
                decoder_in = tf.expand_dims(target_seq[:, i], 1)
                logit, de_state_h, de_state_c, _ = self.decoder(
                    decoder_in, (de_state_h, de_state_c), en_outputs[0])
                loss += loss_function(target_seq[:, i + 1], logit)
        if apply_gradients is True:
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
        return loss / (target_seq.shape[1] - 1)

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
                                     summarizer.lstm_size)
        summarizer.decoder = Decoder(summarizer.embedding_shape_out,
                                     summarizer.lstm_size)
        optimizer = SummarizerAttention._new_optimizer()
        summarizer.encoder.compile(optimizer=optimizer)
        summarizer.decoder.compile(optimizer=optimizer)
        summarizer.encoder.load_weights(encoder_path)
        summarizer.decoder.load_weights(decoder_path)
        summarizer.optimizer = summarizer.encoder.optimizer
        return summarizer

    @staticmethod
    def _new_optimizer():
        return tf.keras.optimizers.Adam()
