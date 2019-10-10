import json
import logging
from typing import Tuple, List

import tensorflow as tf
import tensorflow_datasets as tfds
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

    data_raw = read_data('/Users/cschaefe/datasets/en_ger.txt')[:10000]
    train_data, val_data = train_test_split(data_raw, test_size=100, shuffle=True, random_state=42)


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
    encoded_1 = tokenizer_decoder.encode('<start> <end>')
    encoded_2 = tokenizer_decoder.encode('<start> wie <end>')
   # encoded_2 = tokenizer_decoder.encode(preprocessor.start_token)
    encoded_3 = tokenizer_decoder.encode('<start>')


    print('vocab size encoder: {}, decoder: {}'.format(
        tokenizer_encoder.vocab_size, tokenizer_decoder.vocab_size))

    vectorizer = Vectorizer(tokenizer_encoder, tokenizer_decoder)

    out = vectorizer(('<start> wie <end>', '<start> wie <end>'))


    summarizer = SummarizerTransformer(num_heads=1,
                                       feed_forward_dim=1024,
                                       num_layers=1,
                                       embedding_size=64,
                                       dropout_rate=0,
                                       max_prediction_len=50)
    summarizer.init_model(preprocessor, vectorizer)

    trainer = Trainer(steps_per_epoch=500,
                      batch_size=4,
                      model_save_path='/tmp/summarizer_transformer',
                      steps_to_log=50,
                      bucketing_buffer_size_batches=10000,
                      bucketing_batches_to_bucket=10)

    trainer.train(summarizer,
                  train_data,
                  num_epochs=1000,
                  val_data=val_data)

    best_summarizer = SummarizerTransformer.load('/tmp/summarizer_transformer')
    pred_vectors = best_summarizer.predict_vectors('How are you?', '')
    print(pred_vectors['predicted_text'])






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



