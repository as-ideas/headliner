import json
import logging
import tensorflow as tf
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from headliner.model.summarizer_attention import SummarizerAttention
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

#    tf.get_logger().setLevel(logging.ERROR)
#    logging.basicConfig(level=logging.INFO)
#    data_raw = read_data_json('/Users/cschaefe/datasets/welt_dedup.json', 2000)
#    train_data, val_data = train_test_split(data_raw, test_size=500, shuffle=True, random_state=42)
#    summarizer = SummarizerAttention(lstm_size=256, embedding_size=50)
#    trainer = Trainer(steps_per_epoch=100, glove_path='/Users/cschaefe/datasets/glove_welt_dedup.txt')
#    trainer.train(summarizer, train_data, val_data=val_data)


    logging.basicConfig(level=logging.INFO)
    data_list = [('You are the stars, earth and sky for me!', 'I love you.'),
            ('You are great, but I have other plans.', 'I like you.')]

    class data_iter:

        def __iter__(self):
            return (d for d in data_list)

    data = data_iter()
    summarizer = SummarizerAttention(lstm_size=16, embedding_size=10)
    trainer = Trainer(batch_size=2, steps_per_epoch=100)
    trainer.train(summarizer, data, num_epochs=2)
    pred = summarizer.predict('You are the stars, earth and sky for me!')
    print(pred)

