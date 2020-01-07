from typing import Tuple

from transformers import TFBertModel

from headliner.model.transformer_util import *


class Encoder(tf.keras.Model):

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 embedding_shape: Tuple[int, int],
                 bert_embedding_name=None,
                 embedding_trainable=True,
                 embedding_weights=None,
                 dropout_rate=0.1,
                 max_seq_len=10000) -> None:
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        vocab_size, vec_dim = embedding_shape
        self.embedding_size = vec_dim
        self.bert_embedding_name = bert_embedding_name
        if bert_embedding_name is not None:
            self.embedding = TFBertModel.from_pretrained(bert_embedding_name)
            self.embedding.trainable = embedding_trainable
        else:
            weights = None if embedding_weights is None else [embedding_weights]
            self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                       vec_dim,
                                                       weights=weights,
                                                       trainable=embedding_trainable)
        self.pos_encoding = positional_encoding(max_seq_len, self.embedding_size)
        self.enc_layers = [EncoderLayer(vec_dim, num_heads, feed_forward_dim, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        if self.bert_embedding_name is not None:
            x = self.embedding(x)[0]
        else:
            x = self.embedding(x)
            x *= tf.math.sqrt(tf.cast(self.embedding_size, tf.float32))
            x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x


class Decoder(tf.keras.Model):

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 embedding_shape: Tuple[int, int],
                 bert_embedding_name=None,
                 embedding_trainable=True,
                 embedding_weights=None,
                 dropout_rate=0.1,
                 max_seq_len=10000) -> None:
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        vocab_size, vec_dim = embedding_shape
        self.embedding_size = vec_dim
        self.bert_embedding_name = bert_embedding_name
        if bert_embedding_name is not None:
            self.embedding = TFBertModel.from_pretrained(bert_embedding_name)
            self.embedding.trainable = embedding_trainable
        else:
            weights = None if embedding_weights is None else [embedding_weights]
            self.embedding = tf.keras.layers.Embedding(vocab_size,
                                                       vec_dim,
                                                       weights=weights,
                                                       trainable=embedding_trainable)
        self.pos_encoding = positional_encoding(max_seq_len, vec_dim)
        self.dec_layers = [DecoderLayer(vec_dim, num_heads, feed_forward_dim, dropout_rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.final_layer = tf.keras.layers.Dense(embedding_shape[0])

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
        x = self.final_layer(x)
        return x, attention_weights


class Transformer(tf.keras.Model):

    def __init__(self,
                 num_layers_encoder: int,
                 num_layers_decoder: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 embedding_shape_encoder: Tuple[int, int],
                 embedding_shape_decoder: Tuple[int, int],
                 bert_embedding_encoder=None,
                 bert_embedding_decoder=None,
                 embedding_encoder_trainable=True,
                 embedding_decoder_trainable=True,
                 embedding_weights_encoder=None,
                 embedding_weights_decoder=None,
                 dropout_rate=0.1,
                 max_seq_len=10000) -> None:
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers_encoder,
                               num_heads,
                               feed_forward_dim,
                               embedding_shape_encoder,
                               bert_embedding_name=bert_embedding_encoder,
                               embedding_trainable=embedding_encoder_trainable,
                               embedding_weights=embedding_weights_encoder,
                               dropout_rate=dropout_rate,
                               max_seq_len=max_seq_len)

        self.decoder = Decoder(num_layers_decoder,
                               num_heads,
                               feed_forward_dim,
                               embedding_shape_decoder,
                               bert_embedding_name=bert_embedding_decoder,
                               embedding_trainable=embedding_decoder_trainable,
                               embedding_weights=embedding_weights_decoder,
                               dropout_rate=dropout_rate,
                               max_seq_len=max_seq_len)



    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)

        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        return dec_output, attention_weights
