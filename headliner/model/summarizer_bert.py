import os
import pickle
from typing import Callable
from typing import Dict, Union

import numpy as np
import tensorflow as tf

from headliner.model.model_bert import Transformer, create_masks
from headliner.model.summarizer import Summarizer
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.preprocessing.vectorizer import Vectorizer


class BeamNode:

    def __init__(self,
                 parent_node,
                 sum_logits,
                 token_index):
        self.parent_node = parent_node
        self.sum_logits = sum_logits
        self.token_index = token_index

    def __repr__(self):
        return 'BeamNode (sum_logits: {}, token_index: {}, parent_node: {})'.format(
            self.sum_logits, self.token_index, self.parent_node)


def length_penalty_factor(seq_len, alpha=2):
    return ((5.0 + (seq_len + 1)) / 6.0) ** alpha


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
                 embedding_decoder_trainable=True):

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

    def __getstate__(self):
        """ Prevents pickle from serializing the transformer and optimizer """
        state = self.__dict__.copy()
        del state['transformer']
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
                                       dropout_rate=self.dropout_rate)
        self.transformer.compile()

    def new_train_step(self,
                       loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                       batch_size: int,
                       apply_gradients=True):

        transformer = self.transformer
        optimizer_encoder = self.optimizer_encoder
        optimizer_decoder = self.optimizer_decoder

        train_step_signature = [
            tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
        ]

        @tf.function(input_signature=train_step_signature)
        def train_step(inp, inp_ind, tar):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

            with tf.GradientTape(persistent=True) as tape:
                predictions, _ = transformer(inp,
                                             inp_ind,
                                             tar_inp,
                                             apply_gradients,
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
        en_inputs, en_input_ids, _ = self.vectorizer(text_preprocessed)
        en_inputs = tf.expand_dims(en_inputs, 0)
        en_input_ids = tf.expand_dims(en_input_ids, 0)
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
                                                              en_input_ids,
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

    def predict_beam_search(self, input_text: str, beam_width=5):
        text_preprocessed = self.preprocessor((input_text, ''))
        en_inputs, en_input_ids, _ = self.vectorizer(text_preprocessed)
        en_inputs = tf.expand_dims(en_inputs, 0)
        en_input_ids = tf.expand_dims(en_input_ids, 0)
        start_end_seq = self.vectorizer.encode_output(
            ' '.join([self.preprocessor.start_token, self.preprocessor.end_token]))
        de_start_index, de_end_index = start_end_seq[:1], start_end_seq[-1:]

        # start beam search with single root node
        last_nodes = [BeamNode(parent_node=None, sum_logits=0, token_index=de_start_index[0])]
        final_nodes = []
        en_inputs_shape = tf.shape(en_inputs)
        en_input_ids_shape = tf.shape(en_input_ids)

        # iteratively predict on batch and add k top predictions
        for time_step in range(0, self.max_prediction_len):

            # do batch prediction with current top paths
            en_inputs = tf.broadcast_to(en_inputs[0], (len(last_nodes), en_inputs_shape[1]))
            en_input_ids = tf.broadcast_to(en_input_ids[0], (len(last_nodes), en_input_ids_shape[1]))

            # construct batch of top k paths for next batch prediction of the model
            current_batch = np.zeros((len(last_nodes), time_step+1))
            for i in range(len(last_nodes)):
                node = last_nodes[i]
                for index in range(time_step, -1, -1):
                    current_batch[i, index] = int(node.token_index)
                    node = node.parent_node

            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                en_inputs, current_batch)
            predictions, attention_weights = self.transformer(en_inputs,
                                                              en_input_ids,
                                                              current_batch,
                                                              False,
                                                              enc_padding_mask,
                                                              combined_mask,
                                                              dec_padding_mask)

            # get top k predictions for each path for k*k path candidates
            current_nodes = []
            for i in range(len(last_nodes)):
                top_k_logits, top_k_indices = tf.math.top_k(predictions[i, -1, :], k=beam_width)
                last_node = last_nodes[i]
                for j in range(beam_width):
                    current_nodes.append(BeamNode(parent_node=last_node,
                                                  sum_logits=top_k_logits[j] + last_node.sum_logits,
                                                  token_index=top_k_indices[j]))

            # select top k from k*k path candidate
            current_nodes.sort(key=lambda node: node.sum_logits)
            last_nodes = []
            for current_node in current_nodes[-beam_width:]:
                # if path is finished, add to final nodes,
                if int(current_node.token_index) == de_end_index[0]:
                    final_nodes.append(current_node)
                else:
                    last_nodes.append(current_node)

            if len(last_nodes) == 0:
                break

        # construct sequences from final nodes by traversing the tree from last node to root node
        final_preds = []
        for i in range(len(final_nodes)):
            node = final_nodes[i]
            sum_logits = float(node.sum_logits)
            token_indices = []
            while True:
                token_indices.append(int(node.token_index))
                if node.parent_node is None:
                    token_indices.reverse()
                    pred = self.vectorizer.decode_output(token_indices)
                    sum_logits = sum_logits / length_penalty_factor(len(token_indices))
                    final_preds.append((sum_logits, pred))
                    break
                node = node.parent_node

        final_preds.sort(key=lambda t: t[0])
        return final_preds

    @staticmethod
    def load(in_path: str):
        summarizer_path = os.path.join(in_path, 'summarizer.pkl')
        transformer_path = os.path.join(in_path, 'transformer')
        with open(summarizer_path, 'rb') as handle:
            summarizer = pickle.load(handle)
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
        optimizer_encoder = SummarizerBert.new_optimizer_encoder()
        optimizer_decoder = SummarizerBert.new_optimizer_decoder()
        summarizer.transformer.compile()
        summarizer.transformer.load_weights(transformer_path)
        summarizer.optimizer_encoder = optimizer_encoder
        summarizer.optimizer_transformer = optimizer_decoder

        return summarizer

    @staticmethod
    def new_optimizer_decoder() -> tf.keras.optimizers.Optimizer:
        learning_rate = CustomSchedule(warmup_steps=10000, learning_rate_start=0.1)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999,)
        return optimizer

    @staticmethod
    def new_optimizer_encoder() -> tf.keras.optimizers.Optimizer:
        learning_rate = CustomSchedule(warmup_steps=20000, learning_rate_start=2e-3)
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
