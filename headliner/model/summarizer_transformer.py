import os
import pickle
from typing import Callable
from typing import Dict, Union

import numpy as np
import tensorflow as tf

from headliner.model.model_transformer import Transformer, create_masks
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


def length_penalty_factor(seq_len, alpha=1):
    return ((5.0 + (seq_len + 1)) / 6.0) ** alpha


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
        self.optimizer = SummarizerTransformer.new_optimizer()
        self.transformer = None
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
                                       embedding_shape_encoder=self.embedding_shape_in,
                                       embedding_shape_decoder=self.embedding_shape_out,
                                       embedding_encoder_trainable=self.embedding_encoder_trainable,
                                       embedding_decoder_trainable=self.embedding_decoder_trainable,
                                       embedding_weights_encoder=embedding_weights_encoder,
                                       embedding_weights_decoder=embedding_weights_decoder,
                                       dropout_rate=self.dropout_rate)
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

    def predict_beam_search(self, input_text: str, beam_width=5):
        text_preprocessed = self.preprocessor((input_text, ''))
        en_inputs, _ = self.vectorizer(text_preprocessed)
        en_inputs = tf.expand_dims(en_inputs, 0)
        start_end_seq = self.vectorizer.encode_output(
            ' '.join([self.preprocessor.start_token, self.preprocessor.end_token]))
        de_start_index, de_end_index = start_end_seq[:1], start_end_seq[-1:]
        decoder_output = tf.expand_dims(de_start_index, 0)

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(en_inputs, decoder_output)
        predictions, _ = self.transformer(en_inputs,
                                          decoder_output,
                                          False,
                                          enc_padding_mask,
                                          combined_mask,
                                          dec_padding_mask)

        # build up beam search tree with root node plus first top k predictions
        root_node = BeamNode(parent_node=None, sum_logits=0, token_index=de_start_index[0])
        top_k_logits, top_k_indices = tf.math.top_k(predictions[0, -1, :], k=beam_width)
        last_nodes = [BeamNode(parent_node=root_node,
                               sum_logits=length_penalty_factor(2) * top_k_logits[i],
                               token_index=top_k_indices[i]) for i in range(beam_width)]
        current_batch = np.zeros((beam_width, 2))
        for i in range(beam_width):
            node = last_nodes[i]
            for index in range(1, -1, -1):
                current_batch[i, index] = int(node.token_index)
                node = node.parent_node
        final_nodes = []
        en_inputs_shape = tf.shape(en_inputs)

        # iteratively predict on batch and add k top predictions
        for time_step in range(2, self.max_prediction_len):

            # do batch prediction with current top k paths
            en_inputs = tf.broadcast_to(en_inputs[0], (len(last_nodes), en_inputs_shape[1]))
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                en_inputs, current_batch)
            predictions, attention_weights = self.transformer(en_inputs,
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
                    norm_logit = length_penalty_factor(time_step) * top_k_logits[j]
                    current_nodes.append(BeamNode(parent_node=last_node,
                                                  sum_logits=norm_logit + last_node.sum_logits,
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

            # construct batch of top k paths for next batch prediction of the model
            current_batch = np.zeros((len(last_nodes), time_step+1))
            for i in range(len(last_nodes)):
                node = last_nodes[i]
                for index in range(time_step, -1, -1):
                    current_batch[i, index] = int(node.token_index)
                    node = node.parent_node

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
                    final_preds.append((sum_logits, pred))
                    break
                node = node.parent_node

        final_preds.sort(key=lambda t: t[0])
        return final_preds

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
        optimizer = SummarizerTransformer.new_optimizer()
        summarizer.transformer.compile(optimizer=optimizer)
        summarizer.transformer.load_weights(transformer_path)
        summarizer.optimizer = summarizer.transformer.optimizer
        return summarizer

    @staticmethod
    def new_optimizer() -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.Adam()
