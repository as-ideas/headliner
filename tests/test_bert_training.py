import unittest

import numpy as np
import tensorflow as tf
from spacy.lang.en import English
from transformers import BertTokenizer

from headliner.losses import masked_crossentropy
from headliner.model.bert_summarizer import BertSummarizer
from headliner.preprocessing.bert_preprocessor import BertPreprocessor
from headliner.preprocessing.bert_vectorizer import BertVectorizer
from headliner.preprocessing.dataset_generator import DatasetGenerator
from headliner.preprocessing.keras_tokenizer import KerasTokenizer


class TestBertTraining(unittest.TestCase):

    def setUp(self) -> None:
        tf.random.set_seed(42)
        np.random.seed(42)
        data = [('I love dogs.', 'Dogs.'),
                ('I love cats.', 'Cats.')]
        tokenizer_encoder = BertTokenizer.from_pretrained('bert-base-uncased')
        self.preprocessor = BertPreprocessor(nlp=English())
        data_prep = [self.preprocessor(d) for d in data]
        tokenizer_decoder = KerasTokenizer(lower=False, filters='')
        tokenizer_decoder.fit([d[1] for d in data_prep])
        self.vectorizer = BertVectorizer(tokenizer_encoder=tokenizer_encoder,
                                         tokenizer_decoder=tokenizer_decoder,
                                         max_output_len=10)
        batch_generator = DatasetGenerator(2, rank=3)
        data_vecs = [self.vectorizer(d) for d in data_prep]
        self.dataset = batch_generator(lambda: data_vecs)
        self.loss_func = masked_crossentropy

    def test_training_summarizer_bert(self):
        bert_summarizer = BertSummarizer(num_heads=4,
                                         num_layers_encoder=0,
                                         num_layers_decoder=1,
                                         feed_forward_dim=20,
                                         embedding_size_encoder=768,
                                         embedding_size_decoder=64,
                                         bert_embedding_encoder='bert-base-uncased',
                                         embedding_encoder_trainable=True,
                                         dropout_rate=0,
                                         max_prediction_len=10)
        bert_summarizer.optimizer_encoder = tf.keras.optimizers.Adam(learning_rate=3e-5)
        bert_summarizer.optimizer_decoder = tf.keras.optimizers.Adam(learning_rate=1e-4)
        bert_summarizer.init_model(preprocessor=self.preprocessor,
                                   vectorizer=self.vectorizer,
                                   embedding_weights_encoder=None,
                                   embedding_weights_decoder=None)

        loss_bert = 0
        train_step = bert_summarizer.new_train_step(loss_function=self.loss_func,
                                                    batch_size=2)
        for e in range(0, 10):
            for token_ids, sent_ids, target_ids in self.dataset.take(-1):
                loss_bert = train_step(token_ids, sent_ids, target_ids)
                print(str(loss_bert))

        self.assertAlmostEqual(0.08860704302787781, float(loss_bert), 6)
        model_output = bert_summarizer.predict_vectors('I love dogs.', '')
        expected_first_logits = np.array([-2.069179, -1.594622,  1.158607,  3.03027, 1.404088])
        np.testing.assert_allclose(expected_first_logits, model_output['logits'][0], atol=1e-6)
        self.assertEqual('[CLS] I love dogs. [SEP]', model_output['preprocessed_text'][0])
        self.assertEqual('Dogs. [SEP]', model_output['predicted_text'])
