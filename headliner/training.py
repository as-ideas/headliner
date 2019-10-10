import json
import logging
from typing import Tuple, List

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow_datasets.core.features.text import SubwordTextEncoder

from headliner.evaluation import BleuScorer
from headliner.model.summarizer_transformer import SummarizerTransformer
from headliner.preprocessing import Preprocessor, Vectorizer
from headliner.trainer import Trainer


def read_data_json(file_path: str,
                   max_sequence_length: int) -> List[Tuple[str, str]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data_out = json.load(f)
        return [d for d in zip(data_out['desc'], data_out['heads']) if len(d[0].split(' ')) <= max_sequence_length]


def read_data(file_path: str) -> List[Tuple[str, str]]:
    data_out = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            x, y = l.strip().split('\t')
            data_out.append((x, y))
        return data_out


if __name__ == '__main__':

    data_raw = read_data('/Users/cschaefe/datasets/en_ger.txt')
    train_data, val_data = train_test_split(data_raw, test_size=200, shuffle=True, random_state=42)
    preprocessor = Preprocessor(start_token='<start>', end_token='<end>')
    train_data_prep = [preprocessor(d) for d in train_data]

    input_texts = [t[0] for t in train_data_prep]
    output_texts = [t[1] for t in train_data_prep]

    tokenizer_encoder = SubwordTextEncoder.build_from_corpus(input_texts,
                                                             target_vocab_size=2**13)
    tokenizer_decoder = SubwordTextEncoder.build_from_corpus(output_texts,
                                                             target_vocab_size=2**13,
                                                             reserved_tokens=[preprocessor.start_token,
                                                                              preprocessor.end_token])
    summarizer = SummarizerTransformer(num_heads=1,
                                       feed_forward_dim=1024,
                                       embedding_size=64,
                                       embedding_encoder_trainable=True,
                                       embedding_decoder_trainable=True,
                                       dropout_rate=0.1,
                                       max_prediction_len=20)

    trainer = Trainer(steps_per_epoch=500,
                      batch_size=64,
                      steps_to_log=5,
                      tensorboard_dir='/tmp/trans_emb',
                      num_print_predictions=10)

    trainer.train(summarizer, train_data, val_data=val_data)

    summarizer = SummarizerTransformer.load('/tmp/summarizer')
    pred = summarizer.predict('Some input new')

    print(pred)







    """
    logging.basicConfig(level=logging.INFO)

    class DataIterator:
        def __iter__(self):
            for i in range(100):
                yield ('You are the stars, earth and sky for me!', 'I love you.')

    data_iter = DataIterator()
    summarizer = SummarizerAttention(lstm_size=16, embedding_size=10)

    trainer = Trainer(batch_size=32, steps_per_epoch=100)
    trainer.train(summarizer, data_iter, num_epochs=3)

    pred_vectors = summarizer.predict_vectors('You are great, but I have other plans.', '')
    print(pred_vectors)
    """



