from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pickle
import time
from typing import List
from typing import Tuple, Dict, Union

import numpy as np
import tensorflow as tf
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from headliner.losses import masked_crossentropy
from headliner.preprocessing.dataset_generator import DatasetGenerator
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.preprocessing.vectorizer import Vectorizer


def read_data(file_path: str) -> List[Tuple[str, str]]:
    data_out = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            x, y = l.strip().split('\t')
            data_out.append((x, y))
        return data_out


def get_angles(pos, i, model_dim):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(model_dim))
    return pos * angle_rates


def positional_encoding(position, model_dim):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(model_dim)[np.newaxis, :],
                            model_dim)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def point_wise_feed_forward_network(model_dim, feed_forward_dim):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(feed_forward_dim, activation='relu'),  # (batch_size, seq_len, feed_forward_dim)
      tf.keras.layers.Dense(model_dim)  # (batch_size, seq_len, model_dim)
  ])


def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        assert model_dim % self.num_heads == 0
        self.depth = model_dim // self.num_heads
        self.wq = tf.keras.layers.Dense(model_dim)
        self.wk = tf.keras.layers.Dense(model_dim)
        self.wv = tf.keras.layers.Dense(model_dim)
        self.dense = tf.keras.layers.Dense(model_dim)

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
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.model_dim))
        output = self.dense(concat_attention)
        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, model_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(model_dim, num_heads)
        self.ffn = point_wise_feed_forward_network(model_dim, feed_forward_dim)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, model_dim)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, model_dim)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, model_dim)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, model_dim)
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    
    def __init__(self, model_dim, num_heads, feed_forward_dim, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(model_dim, num_heads)
        self.mha2 = MultiHeadAttention(model_dim, num_heads)
        self.ffn = point_wise_feed_forward_network(model_dim, feed_forward_dim)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x,
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
                 num_layers,
                 model_dim, num_heads,
                 feed_forward_dim,
                 input_vocab_size,
                 rate=0.1):
        super(Encoder, self).__init__()

        self.model_dim = model_dim
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, model_dim)
        self.pos_encoding = positional_encoding(input_vocab_size, self.model_dim)
        self.enc_layers = [EncoderLayer(model_dim, num_heads, feed_forward_dim, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers,
                 model_dim,
                 num_heads,
                 feed_forward_dim,
                 target_vocab_size,
                 dropout_rate=0.1):

        super(Decoder, self).__init__()

        self.model_dim = model_dim
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, model_dim)
        self.pos_encoding = positional_encoding(target_vocab_size, model_dim)
        self.dec_layers = [DecoderLayer(model_dim, num_heads, feed_forward_dim, dropout_rate)
                          for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, model_dim)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


class Transformer(tf.keras.Model):

    def __init__(self, num_layers, model_dim, num_heads, feed_forward_dim, input_vocab_size,
                 target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers,
                               model_dim,
                               num_heads,
                               feed_forward_dim,
                               input_vocab_size,
                               rate)

        self.decoder = Decoder(num_layers,
                               model_dim,
                               num_heads,
                               feed_forward_dim,
                               target_vocab_size,
                               rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, model_dim)

        # dec_output.shape == (batch_size, tar_seq_len, model_dim)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights



tf.random.set_seed(42)
np.random.seed(42)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, model_dim, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.model_dim = model_dim
        self.model_dim = tf.cast(self.model_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.model_dim) * tf.math.minimum(arg1, arg2)



class SummarizerTransformer:

    def __init__(self,
                 max_prediction_len=20,
                 model_dim=128,
                 num_layers=1,
                 num_heads=2,
                 feed_forward_dim=512,
                 dropout_rate=0,
                 embedding_size=50,
                 embedding_encoder_trainable=True,
                 embedding_decoder_trainable=True):

        self.max_prediction_len = max_prediction_len
        self.model_dim = model_dim
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.dropout_rate = dropout_rate
        self.feed_forward_dim = feed_forward_dim
        self.embedding_size = embedding_size
        self.embedding_encoder_trainable = embedding_encoder_trainable
        self.embedding_decoder_trainable = embedding_decoder_trainable
        self.preprocessor = None
        self.vectorizer = None
        self.encoder = None
        self.decoder = None
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
        self.transformer = Transformer(self.num_layers,
                                       self.model_dim,
                                       self.num_heads,
                                       self.feed_forward_dim,
                                       self.vectorizer.encoding_dim,
                                       self.vectorizer.decoding_dim,
                                       self.dropout_rate)
        self.optimizer = self.new_optimizer()
        self.transformer.compile(optimizer=self.optimizer)

    def new_train_step(self, loss_function, batch_size, apply_gradients=True):

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

        text_prerpcessed = self.preprocessor((input_text, target_text))
        inp_sentence, output_sentence = self.vectorizer(text_prerpcessed)
        encoder_input = tf.expand_dims(inp_sentence, 0)
        decoder_input = self.vectorizer.encode_output(self.preprocessor.start_token)
        decoder_output = tf.expand_dims(decoder_input, 0)
        output = {'preprocessed_text': text_prerpcessed,
                  'logits': [],
                  'alignment': [],
                  'predicted_sequence': []}
        for i in range(self.max_prediction_len):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                encoder_input, decoder_output)
            predictions, alignment = self.transformer(encoder_input,
                                                              decoder_output,
                                                              False,
                                                              enc_padding_mask,
                                                              combined_mask,
                                                              dec_padding_mask)

            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            pred_token_index = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            end_index = self.vectorizer.encode_output(self.preprocessor.end_token)
            decoder_output = tf.concat([decoder_output, pred_token_index], axis=-1)
            if pred_token_index != 0:
                output['logits'].append(np.squeeze(predictions.numpy()))
                output['alignment'].append(alignment)
                output['predicted_sequence'].append(int(pred_token_index))
                if pred_token_index == end_index:
                    break
        output['predicted_text'] = self.vectorizer.decode_output(output['predicted_sequence'])
        return output

    def save(self, out_path):
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        summarizer_path = os.path.join(out_path, 'summarizer.pkl')
        transformer_path = os.path.join(out_path, 'transformer')
        with open(summarizer_path, 'wb+') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.transformer.save_weights(transformer_path, save_format='tf')

    @staticmethod
    def load(in_path):
        summarizer_path = os.path.join(in_path, 'summarizer.pkl')
        transformer_path = os.path.join(in_path, 'transformer')
        with open(summarizer_path, 'rb') as handle:
            summarizer = pickle.load(handle)
        summarizer.transformer = Transformer(summarizer.num_layers,
                                             summarizer.model_dim,
                                             summarizer.num_heads,
                                             summarizer.feed_forward_dim,
                                             summarizer.vectorizer.encoding_dim,
                                             summarizer.vectorizer.decoding_dim,
                                             summarizer.dropout_rate)
        optimizer = summarizer.new_optimizer()
        summarizer.transformer.compile(optimizer=optimizer)
        summarizer.transformer.load_weights(transformer_path)
        summarizer.optimizer = summarizer.transformer.optimizer
        return summarizer

    def new_optimizer(self):
        learning_rate = CustomSchedule(self.model_dim)
        return tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


if __name__ == '__main__':

    data_raw = read_data('/Users/cschaefe/datasets/en_ger.txt')[:10000]
    train_data, val_data = train_test_split(data_raw, test_size=100, shuffle=True, random_state=42)
    preprocessor = Preprocessor()
    train_text_encoder = (preprocessor(d)[0] for d in train_data)
    train_text_decoder = (preprocessor(d)[1] for d in train_data)
    tokenizer_encoder = Tokenizer(oov_token='<oov>', filters='', lower=False)
    tokenizer_decoder = Tokenizer(oov_token='<oov>', filters='', lower=False)
    tokenizer_encoder.fit_on_texts(train_text_encoder)
    tokenizer_decoder.fit_on_texts(train_text_decoder)
    vectorizer = Vectorizer(tokenizer_encoder, tokenizer_decoder)

    BUFFER_SIZE = 20000
    BATCH_SIZE = 16
    MAX_LENGTH = 10
    batch_generator = DatasetGenerator(BATCH_SIZE)
    data_prep_train = [preprocessor(d) for d in train_data]
    data_vecs_train = [vectorizer(d) for d in data_prep_train]
    data_prep_val = [preprocessor(d) for d in val_data]
    data_vecs_val = [vectorizer(d) for d in data_prep_val]
    train_dataset = batch_generator(lambda: data_vecs_train)
    val_dataset = batch_generator(lambda: data_vecs_val)
    input_vocab_size = len(tokenizer_encoder.word_index) + 1
    target_vocab_size = len(tokenizer_decoder.word_index) + 1

    summarizer_transformer = SummarizerTransformer(max_prediction_len=MAX_LENGTH)
    summarizer_transformer.init_model(preprocessor, vectorizer)
    loss_function = masked_crossentropy
    train_step = summarizer_transformer.new_train_step(loss_function, BATCH_SIZE)
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    for epoch in range(20):
        start = time.time()

        train_loss.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            loss = train_step(inp, tar)
            train_loss(loss)

            if batch % 50 == 0:
                if batch == 50:
                    res = float(train_loss.result())
                    assert abs(res - 6.362922191619873) < 1e-6, 'train loss result does not match!'
                    attention_weights = summarizer_transformer.predict_vectors(val_data[0][0], '')['alignment'][-1]
                    sum_weight = np.sum(np.squeeze(attention_weights['decoder_layer1_block1'].numpy()), axis=1)
                    assert abs(sum_weight[0][0] - 3.0068233) < 1e-6, 'attention weight result does not match!'

                print()
                for l in range(5):
                    pred_vecs = summarizer_transformer.predict_vectors(val_data[l][0], val_data[l][1])
                    print('pred text vec: ' + pred_vecs['predicted_text'])

                print('Epoch {} Batch {} Loss {:.10f}'.format(epoch + 1, batch, train_loss.result()))


        print('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
