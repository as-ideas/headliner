from typing import Tuple

import tensorflow as tf


class Encoder(tf.keras.Model):

    def __init__(self,
                 embedding_shape: Tuple[int, int],
                 lstm_size=50,
                 embedding_weights=None,
                 embedding_trainable=True) -> None:
        super(Encoder, self).__init__()
        vocab_size, vec_dim = embedding_shape
        weights = None if embedding_weights is None else [embedding_weights]
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   vec_dim,
                                                   weights=weights,
                                                   trainable=embedding_trainable)
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True, go_backwards=True)
        self.lstm_size = lstm_size

    def call(self,
             sequence: tf.Tensor,
             states: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        embed = self.embedding(sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)
        return output, state_h, state_c

    def init_states(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        return tf.zeros([batch_size, self.lstm_size]), tf.zeros([batch_size, self.lstm_size])


class Decoder(tf.keras.Model):

    def __init__(self,
                 embedding_shape: Tuple[int, int],
                 lstm_size=50,
                 embedding_weights=None,
                 embedding_trainable=True) -> None:
        super(Decoder, self).__init__()
        self.lstm_size = lstm_size
        vocab_size, vec_dim = embedding_shape
        weights = None if embedding_weights is None else [embedding_weights]
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   vec_dim,
                                                   weights=weights,
                                                   trainable=embedding_trainable)
        self.lstm = tf.keras.layers.LSTM(lstm_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence: tf.Tensor, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        embed = self.embedding(sequence)
        lstm_out, state_h, state_c = self.lstm(embed, state)
        logits = self.dense(lstm_out)
        return logits, state_h, state_c
