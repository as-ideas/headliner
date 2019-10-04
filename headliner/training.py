import json
import logging
import tensorflow as tf
from typing import Tuple, List
from sklearn.model_selection import train_test_split

from headliner.evaluation import BleuScorer
from headliner.model.summarizer_attention import SummarizerAttention
from headliner.model.summarizer_transformer import SummarizerTransformer
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


    tf.get_logger().setLevel(logging.ERROR)

    data_raw = read_data('/Users/cschaefe/datasets/en_ger.txt')[:10000]
    train_data, val_data = train_test_split(data_raw, test_size=100, shuffle=True, random_state=42)
    summarizer = SummarizerAttention(max_prediction_len=12, lstm_size=256, embedding_size=50)

#    summarizer = SummarizerTransformer(max_prediction_len=12)

    trainer = Trainer(steps_per_epoch=500,
                      batch_size=16,
                      steps_to_log=5,
                      max_output_len=10,
                      tensorboard_dir='/tmp/attention')

    trainer.train(summarizer, train_data, val_data=val_data, scorers={'bleu': BleuScorer(weights=(1, 0, 0, 0))})








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



