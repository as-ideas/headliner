from typing import Tuple

import tensorflow as tf


class Encoder(tf.keras.Model):

    def __init__(self,
                 embedding_shape: Tuple[int, int],
                 lstm_size=50,
                 embedding_trainable=True,
                 embedding_weights=None) -> None:
        super(Encoder, self).__init__()
        vocab_size, vec_dim = embedding_shape
        weights = None if embedding_weights is None else [embedding_weights]
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   vec_dim,
                                                   weights=weights,
                                                   trainable=embedding_trainable)
        self.lstm = tf.keras.layers.LSTM(lstm_size,
                                         return_sequences=True,
                                         return_state=True)
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
                 embedding_trainable=True,
                 embedding_weights=None) -> None:
        super(Decoder, self).__init__()
        self.lstm_size = lstm_size
        vocab_size, vec_dim = embedding_shape
        weights = None if embedding_weights is None else [embedding_weights]
        self.embedding = tf.keras.layers.Embedding(vocab_size, vec_dim,
                                                   weights=weights,
                                                   trainable=embedding_trainable)
        self.lstm_size = lstm_size
        self.attention = LuongAttention(lstm_size)
        self.lstm = tf.keras.layers.LSTM(lstm_size,
                                         return_sequences=True,
                                         return_state=True)
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

