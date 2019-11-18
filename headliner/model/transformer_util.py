import numpy as np
import tensorflow as tf


def get_angles(pos, i, embedding_size):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_size))
    return pos * angle_rates


def positional_encoding(max_len, embedding_size):
    angle_rads = get_angles(np.arange(max_len)[:, np.newaxis],
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
