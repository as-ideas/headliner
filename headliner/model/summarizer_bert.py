import os
import pickle
from typing import Callable
from typing import Dict, Union

import numpy as np
import tensorflow as tf
from headliner.utils.logger import get_logger

from headliner.model.model_bert import Transformer, create_masks
from headliner.model.summarizer import Summarizer
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.preprocessing.vectorizer import Vectorizer


class SummarizerBert(Summarizer):

    def __init__(self,
                 max_prediction_len=20,
                 num_layers_encoder=1,
                 num_layers_decoder=1,
                 num_heads=2,
                 feed_forward_dim=512,
                 dropout_rate=0,
                 embedding_size_encoder=768,
                 embedding_size_decoder=64,
                 bert_embedding_encoder=None,
                 bert_embedding_decoder=None,
                 embedding_encoder_trainable=True,
                 embedding_decoder_trainable=True,
                 max_sequence_len=10000):

        super().__init__()
        self.max_prediction_len = max_prediction_len
        self.embedding_size_encoder = embedding_size_encoder
        self.embedding_size_decoder = embedding_size_decoder
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.feed_forward_dim = feed_forward_dim
        self.embedding_encoder_trainable = embedding_encoder_trainable
        self.embedding_decoder_trainable = embedding_decoder_trainable
        self.bert_embedding_encoder = bert_embedding_encoder
        self.bert_embedding_decoder = bert_embedding_decoder
        self.optimizer_encoder = SummarizerBert.new_optimizer_encoder()
        self.optimizer_decoder = SummarizerBert.new_optimizer_decoder()
        self.transformer = None
        self.embedding_shape_in = None
        self.embedding_shape_out = None
        self.max_sequence_len = max_sequence_len
        self.logger = get_logger(__name__)

    def __getstate__(self):
        """ Prevents pickle from serializing the transformer and optimizer """
        state = self.__dict__.copy()
        del state['transformer']
        del state['logger']
        del state['optimizer_encoder']
        del state['optimizer_decoder']
        return state

    def init_model(self,
                   preprocessor: Preprocessor,
                   vectorizer: Vectorizer,
                   embedding_weights_encoder=None,
                   embedding_weights_decoder=None
                   ) -> None:
        self.preprocessor = preprocessor
        self.vectorizer = vectorizer
        self.embedding_shape_in = (self.vectorizer.encoding_dim, self.embedding_size_encoder)
        self.embedding_shape_out = (self.vectorizer.decoding_dim, self.embedding_size_decoder)
        self.transformer = Transformer(num_layers_encoder=self.num_layers_encoder,
                                       num_layers_decoder=self.num_layers_decoder,
                                       num_heads=self.num_heads,
                                       feed_forward_dim=self.feed_forward_dim,
                                       embedding_shape_encoder=self.embedding_shape_in,
                                       embedding_shape_decoder=self.embedding_shape_out,
                                       bert_embedding_encoder=self.bert_embedding_encoder,
                                       bert_embedding_decoder=self.bert_embedding_decoder,
                                       embedding_encoder_trainable=self.embedding_encoder_trainable,
                                       embedding_decoder_trainable=self.embedding_decoder_trainable,
                                       embedding_weights_encoder=embedding_weights_encoder,
                                       embedding_weights_decoder=embedding_weights_decoder,
                                       dropout_rate=self.dropout_rate,
                                       max_seq_len=self.max_sequence_len)
        self.transformer.compile()

    def new_train_step(self,
                       loss_function: Callable[[tf.Tensor], tf.Tensor],
                       batch_size: int,
                       apply_gradients=True):

        transformer = self.transformer
        optimizer_encoder = self.optimizer_encoder
        optimizer_decoder = self.optimizer_decoder

        train_step_signature = [
            tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
            with tf.GradientTape(persistent=True) as tape:
                predictions, _ = transformer(inp, tar_inp,
                                             True,
                                             enc_padding_mask,
                                             combined_mask,
                                             dec_padding_mask)
                loss = loss_function(tar_real, predictions)
            if apply_gradients:
                transformer.encoder.trainable = False
                gradients = tape.gradient(loss, transformer.trainable_variables)
                optimizer_decoder.apply_gradients(zip(gradients, transformer.trainable_variables))
                transformer.encoder.trainable = True
                gradients_encoder = tape.gradient(loss, transformer.encoder.trainable_variables)
                optimizer_encoder.apply_gradients(zip(gradients_encoder, transformer.encoder.trainable_variables))

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
        optimizer_encoder_path = os.path.join(out_path, 'optimizer_encoder.pkl')
        optimizer_decoder_path = os.path.join(out_path, 'optimizer_decoder.pkl')
        transformer_path = os.path.join(out_path, 'transformer')
        with open(summarizer_path, 'wb+') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # just quick & dirty pickle the optimizer states
        with open(optimizer_encoder_path, 'wb+') as handle:
            pickle.dump(self.optimizer_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(optimizer_decoder_path, 'wb+') as handle:
            pickle.dump(self.optimizer_decoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.transformer.save_weights(transformer_path, save_format='tf')

    @staticmethod
    def load(in_path: str):
        summarizer_path = os.path.join(in_path, 'summarizer.pkl')
        transformer_path = os.path.join(in_path, 'transformer')
        optimizer_encoder_path = os.path.join(in_path, 'optimizer_encoder.pkl')
        optimizer_decoder_path = os.path.join(in_path, 'optimizer_decoder.pkl')
        with open(summarizer_path, 'rb') as handle:
            summarizer = pickle.load(handle)
        summarizer.logger = get_logger(__name__)
        summarizer.transformer = Transformer(num_layers_encoder=summarizer.num_layers_encoder,
                                             num_layers_decoder=summarizer.num_layers_decoder,
                                             num_heads=summarizer.num_heads,
                                             feed_forward_dim=summarizer.feed_forward_dim,
                                             embedding_shape_encoder=summarizer.embedding_shape_in,
                                             embedding_shape_decoder=summarizer.embedding_shape_out,
                                             bert_embedding_encoder=summarizer.bert_embedding_encoder,
                                             bert_embedding_decoder=summarizer.bert_embedding_decoder,
                                             embedding_encoder_trainable=summarizer.embedding_encoder_trainable,
                                             embedding_decoder_trainable=summarizer.embedding_decoder_trainable,
                                             dropout_rate=summarizer.dropout_rate)
        summarizer.transformer.compile()
        summarizer.transformer.load_weights(transformer_path)

        # just quick & dirty unpickle the optimizer states
        try:
            with open(optimizer_encoder_path, 'rb') as handle:
                summarizer.optimizer_encoder = pickle.load(handle)
        except Exception as e:
            summarizer.logger.warn('Warning: Could not load {}, creating a bare optimizer. {}'
                                   .format(optimizer_encoder_path, e))
            summarizer.optimizer_encoder = SummarizerBert.new_optimizer_encoder()
        try:
            with open(optimizer_decoder_path, 'rb') as handle:
                summarizer.optimizer_decoder = pickle.load(handle)
        except Exception as e:
            summarizer.logger.warn('Warning: Could not load {}, creating a bare optimizer. {}'
                                   .format(optimizer_decoder_path, e))
            summarizer.optimizer_decoder = SummarizerBert.new_optimizer_decoder()

        return summarizer

    @staticmethod
    def new_optimizer_decoder(learning_rate_start=0.02) -> tf.keras.optimizers.Optimizer:
        learning_rate = CustomSchedule(warmup_steps=10000, learning_rate_start=learning_rate_start)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999,)
        return optimizer

    @staticmethod
    def new_optimizer_encoder(learning_rate_start=5e-4) -> tf.keras.optimizers.Optimizer:
        learning_rate = CustomSchedule(warmup_steps=20000, learning_rate_start=learning_rate_start)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999,)
        return optimizer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, warmup_steps=10000, learning_rate_start=1e-1):
        super(CustomSchedule, self).__init__()
        self.warmup_steps = warmup_steps
        self.learning_rate_start = learning_rate_start

    def __call__(self, step):
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        return self.learning_rate_start * tf.math.minimum(arg1, arg2)
