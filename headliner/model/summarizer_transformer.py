import os
import pickle
from typing import Callable
from typing import Tuple, Dict, Union

import numpy as np
import tensorflow as tf

from headliner.model.summarizer import Summarizer
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.preprocessing.vectorizer import Vectorizer


def get_angles(pos, i, embedding_size):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_size))
    return pos * angle_rates


def positional_encoding(position, embedding_size):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(embedding_size)[np.newaxis, :],
                            embedding_size)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


def point_wise_feed_forward_network(embedding_size, feed_forward_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(feed_forward_dim, activation='relu'),
        tf.keras.layers.Dense(embedding_size)
    ])


def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self,
                 embedding_size: int,
                 num_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        assert embedding_size % self.num_heads == 0
        self.depth = embedding_size // self.num_heads
        self.wq = tf.keras.layers.Dense(embedding_size)
        self.wk = tf.keras.layers.Dense(embedding_size)
        self.wv = tf.keras.layers.Dense(embedding_size)
        self.dense = tf.keras.layers.Dense(embedding_size)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embedding_size))
        output = self.dense(concat_attention)
        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self,
                 embedding_size: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 dropout_rate=0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(embedding_size, num_heads)
        self.ffn = point_wise_feed_forward_network(embedding_size, feed_forward_dim)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, embedding_size)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, embedding_size)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, embedding_size)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, embedding_size)
        return out2


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self,
                 embedding_size: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 dropout_rate=0.1) -> None:
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(embedding_size, num_heads)
        self.mha2 = MultiHeadAttention(embedding_size, num_heads)
        self.ffn = point_wise_feed_forward_network(embedding_size, feed_forward_dim)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self,
             x,
             enc_output,
             training,
             look_ahead_mask,
             padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 embedding_shape: Tuple[int, int],
                 embedding_trainable=True,
                 embedding_weights=None,
                 dropout_rate=0.1) -> None:
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        vocab_size, vec_dim = embedding_shape
        weights = None if embedding_weights is None else [embedding_weights]
        self.embedding_size = vec_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   vec_dim,
                                                   weights=weights,
                                                   trainable=embedding_trainable)
        self.pos_encoding = positional_encoding(vocab_size, self.embedding_size)
        self.enc_layers = [EncoderLayer(vec_dim, num_heads, feed_forward_dim, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 embedding_shape: Tuple[int, int],
                 embedding_trainable=True,
                 embedding_weights=None,
                 dropout_rate=0.1) -> None:
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        vocab_size, vec_dim = embedding_shape
        weights = None if embedding_weights is None else [embedding_weights]
        self.embedding_size = vec_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                   vec_dim,
                                                   weights=weights,
                                                   trainable=embedding_trainable)
        self.pos_encoding = positional_encoding(vocab_size, vec_dim)
        self.dec_layers = [DecoderLayer(vec_dim, num_heads, feed_forward_dim, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self,
             x,
             enc_output,
             training,
             look_ahead_mask,
             padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


class Transformer(tf.keras.Model):

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 embedding_shape_encoder: Tuple[int, int],
                 embedding_shape_decoder: Tuple[int, int],
                 embedding_encoder_trainable=True,
                 embedding_decoder_trainable=True,
                 embedding_weights_encoder=None,
                 embedding_weights_decoder=None,
                 dropout_rate=0.1) -> None:
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers,
                               num_heads,
                               feed_forward_dim,
                               embedding_shape_encoder,
                               embedding_trainable=embedding_encoder_trainable,
                               embedding_weights=embedding_weights_encoder,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers,
                               num_heads,
                               feed_forward_dim,
                               embedding_shape_decoder,
                               embedding_trainable=embedding_decoder_trainable,
                               embedding_weights=embedding_weights_decoder,
                               dropout_rate=dropout_rate)

        self.final_layer = tf.keras.layers.Dense(embedding_shape_decoder[0])

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)

        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights


class SummarizerTransformer(Summarizer):

    def __init__(self,
                 max_prediction_len=20,
                 num_layers=1,
                 num_heads=2,
                 feed_forward_dim=512,
                 dropout_rate=0,
                 embedding_size=128,
                 embedding_encoder_trainable=True,
                 embedding_decoder_trainable=True):

        super().__init__()
        self.max_prediction_len = max_prediction_len
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.feed_forward_dim = feed_forward_dim
        self.embedding_encoder_trainable = embedding_encoder_trainable
        self.embedding_decoder_trainable = embedding_decoder_trainable
        self.transformer = None
        self.optimizer = None
        self.embedding_shape_in = None
        self.embedding_shape_out = None

    def __getstate__(self):
        """ Prevents pickle from serializing the transformer and optimizer """
        state = self.__dict__.copy()
        del state['transformer']
        del state['optimizer']
        return state

    def init_model(self,
                   preprocessor: Preprocessor,
                   vectorizer: Vectorizer,
                   embedding_weights_encoder=None,
                   embedding_weights_decoder=None
                   ) -> None:
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.embedding_shape_in = (self.vectorizer.encoding_dim, self.embedding_size)
        self.embedding_shape_out = (self.vectorizer.decoding_dim, self.embedding_size)
        self.transformer = Transformer(num_layers=self.num_layers,
                                       num_heads=self.num_heads,
                                       feed_forward_dim=self.feed_forward_dim,
                                       embedding_shape_encoder=(self.vectorizer.encoding_dim, self.embedding_size),
                                       embedding_shape_decoder=(self.vectorizer.decoding_dim, self.embedding_size),
                                       embedding_encoder_trainable=self.embedding_encoder_trainable,
                                       embedding_decoder_trainable=self.embedding_decoder_trainable,
                                       embedding_weights_encoder=embedding_weights_encoder,
                                       embedding_weights_decoder=embedding_weights_decoder,
                                       dropout_rate=self.dropout_rate)
        self.optimizer = self.new_optimizer()
        self.transformer.compile(optimizer=self.optimizer)

    def new_train_step(self,
                       loss_function: Callable[[tf.Tensor], tf.Tensor],
                       batch_size: int,
                       apply_gradients=True):

        transformer = self.transformer
        optimizer = self.optimizer

        train_step_signature = [
            tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
            with tf.GradientTape() as tape:
                predictions, _ = transformer(inp, tar_inp,
                                             True,
                                             enc_padding_mask,
                                             combined_mask,
                                             dec_padding_mask)
                loss = loss_function(tar_real, predictions)
            if apply_gradients:
                gradients = tape.gradient(loss, transformer.trainable_variables)
                optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
            return loss

        return train_step

    def predict(self, text: str) -> str:
        return self.predict_vectors(text, '')['predicted_text']

    def predict_vectors(self, input_text: str, target_text: str) -> Dict[str, Union[str, np.array]]:
        text_preprocessed = self.preprocessor((input_text, target_text))
        inp_sentence, output_sentence = self.vectorizer(text_preprocessed)
        encoder_input = tf.expand_dims(inp_sentence, 0)
        start_end_seq = self.vectorizer.encode_output(
            ' '.join([self.preprocessor.start_token, self.preprocessor.end_token]))
        de_start_index, de_end_index = start_end_seq[:1], start_end_seq[-1:]
        decoder_output = tf.expand_dims(de_start_index, 0)
        output = {'preprocessed_text': text_preprocessed,
                  'logits': [],
                  'attention_weights': [],
                  'predicted_sequence': []}
        for i in range(self.max_prediction_len):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, decoder_output)
            predictions, attention_weights = self.transformer(encoder_input,
                                                              decoder_output,
                                                              False,
                                                              enc_padding_mask,
                                                              combined_mask,
                                                              dec_padding_mask)

            predictions = predictions[:, -1:, :]
            pred_token_index = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            decoder_output = tf.concat([decoder_output, pred_token_index], axis=-1)
            if pred_token_index != 0:
                output['logits'].append(np.squeeze(predictions.numpy()))
                output['attention_weights'] = attention_weights
                output['predicted_sequence'].append(int(pred_token_index))
                if pred_token_index == de_end_index:
                    break
        output['predicted_text'] = self.vectorizer.decode_output(output['predicted_sequence'])
        return output

    def save(self, out_path: str) -> None:
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        summarizer_path = os.path.join(out_path, 'summarizer.pkl')
        transformer_path = os.path.join(out_path, 'transformer')
        with open(summarizer_path, 'wb+') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.transformer.save_weights(transformer_path, save_format='tf')

    @staticmethod
    def load(in_path: str):
        summarizer_path = os.path.join(in_path, 'summarizer.pkl')
        transformer_path = os.path.join(in_path, 'transformer')
        with open(summarizer_path, 'rb') as handle:
            summarizer = pickle.load(handle)
        summarizer.transformer = Transformer(num_layers=summarizer.num_layers,
                                             num_heads=summarizer.num_heads,
                                             feed_forward_dim=summarizer.feed_forward_dim,
                                             embedding_shape_encoder=(summarizer.vectorizer.encoding_dim,
                                                                      summarizer.embedding_size),
                                             embedding_shape_decoder=(summarizer.vectorizer.decoding_dim,
                                                                      summarizer.embedding_size),
                                             embedding_encoder_trainable=summarizer.embedding_encoder_trainable,
                                             embedding_decoder_trainable=summarizer.embedding_decoder_trainable,
                                             dropout_rate=summarizer.dropout_rate)
        optimizer = SummarizerTransformer.new_optimizer()
        summarizer.transformer.compile(optimizer=optimizer)
        summarizer.transformer.load_weights(transformer_path)
        summarizer.optimizer = summarizer.transformer.optimizer
        return summarizer

    @staticmethod
    def new_optimizer() -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.Adam()
