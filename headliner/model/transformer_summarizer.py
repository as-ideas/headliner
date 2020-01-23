import os
import pickle
from typing import Callable
from typing import Dict, Union

import numpy as np
import tensorflow as tf

from headliner.model.summarizer import Summarizer
from headliner.model.transformer_model import Transformer, create_masks
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.preprocessing.vectorizer import Vectorizer


class TransformerSummarizer(Summarizer):

    def __init__(self,
                 max_prediction_len=20,
                 num_layers=1,
                 num_heads=2,
                 feed_forward_dim=512,
                 dropout_rate=0,
                 embedding_size=128,
                 embedding_encoder_trainable=True,
                 embedding_decoder_trainable=True,
                 max_sequence_len=10000):

        super().__init__()
        self.max_prediction_len = max_prediction_len
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.feed_forward_dim = feed_forward_dim
        self.embedding_encoder_trainable = embedding_encoder_trainable
        self.embedding_decoder_trainable = embedding_decoder_trainable
        self.optimizer = TransformerSummarizer.new_optimizer()
        self.transformer = None
        self.embedding_shape_in = None
        self.embedding_shape_out = None
        self.max_sequence_len = max_sequence_len

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
                   embedding_weights_decoder=None) -> None:
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.embedding_shape_in = (self.vectorizer.encoding_dim, self.embedding_size)
        self.embedding_shape_out = (self.vectorizer.decoding_dim, self.embedding_size)
        self.transformer = Transformer(num_layers=self.num_layers,
                                       num_heads=self.num_heads,
                                       feed_forward_dim=self.feed_forward_dim,
                                       embedding_shape_encoder=self.embedding_shape_in,
                                       embedding_shape_decoder=self.embedding_shape_out,
                                       embedding_encoder_trainable=self.embedding_encoder_trainable,
                                       embedding_decoder_trainable=self.embedding_decoder_trainable,
                                       embedding_weights_encoder=embedding_weights_encoder,
                                       embedding_weights_decoder=embedding_weights_decoder,
                                       dropout_rate=self.dropout_rate,
                                       max_sequence_len=self.max_sequence_len)
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
        en_inputs, _ = self.vectorizer(text_preprocessed)
        en_inputs = tf.expand_dims(en_inputs, 0)
        start_end_seq = self.vectorizer.encode_output(
            ' '.join([self.preprocessor.start_token, self.preprocessor.end_token]))
        de_start_index, de_end_index = start_end_seq[:1], start_end_seq[-1:]
        decoder_output = tf.expand_dims(de_start_index, 0)
        output = {'preprocessed_text': text_preprocessed,
                  'logits': [],
                  'attention_weights': [],
                  'predicted_sequence': []}
        for _ in range(self.max_prediction_len):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                en_inputs, decoder_output)
            predictions, attention_weights = self.transformer(en_inputs,
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
                                             embedding_shape_encoder=summarizer.embedding_shape_in,
                                             embedding_shape_decoder=summarizer.embedding_shape_out,
                                             embedding_encoder_trainable=summarizer.embedding_encoder_trainable,
                                             embedding_decoder_trainable=summarizer.embedding_decoder_trainable,
                                             dropout_rate=summarizer.dropout_rate)
        optimizer = TransformerSummarizer.new_optimizer()
        summarizer.transformer.compile(optimizer=optimizer)
        summarizer.transformer.load_weights(transformer_path)
        summarizer.optimizer = summarizer.transformer.optimizer
        return summarizer

    @staticmethod
    def new_optimizer() -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.Adam()
