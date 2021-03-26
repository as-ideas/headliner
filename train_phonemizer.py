import json
import re
import tensorflow as tf
from random import Random

import tqdm

from headliner.preprocessing import Preprocessor

from build.lib.headliner.evaluation.scorer import Scorer
from evaluation import calculate_wer
from headliner.callbacks import EvaluationCallback
from thinc.neural.optimizers import Adam

from headliner.model.transformer_summarizer import TransformerSummarizer
from headliner.trainer import Trainer
import pickle


phonemes_set = set(' abdefhijklmnoprstuvwxyzæçøŋœɐɑɔəɛɡɪʁʃʊʏʒʔˈˌː̥̩̯̃̍͡')


if __name__ == '__main__':

    with open('/Users/cschaefe/datasets/nlp/heavily_cleaned_phoneme_dataset_DE.pkl', 'rb') as f:
        df = pickle.load(f)
    tuples = df[['title', 'pronunciation']]
    tuples = [tuple(x) for x in tuples.to_numpy()]
    data_set = {w for w, _ in tuples}
    train_data = []
    max_len = 50

    all_phons = set()
    for word, phon in tqdm.tqdm(tuples, total=len(tuples)):
        all_phons.update(set(phon))
        all_phons_list = sorted(list(all_phons))
        with open('/tmp/all_phons.txt', 'w+', encoding='utf-8') as f:
            f.write(''.join(all_phons_list))
        if 0 < len(phon) < max_len and ' ' not in word and 0 < len(word) < max_len:
            word_ = ' '.join(word)
            phon = ' '.join(phon)
            print(f'{word} {phon}')
            train_data.append((word_, phon))
            if word.lower() not in data_set:
                word_ = ' '.join(word.lower())
                train_data.append((word_, phon))
            if word.title() not in data_set:
                word_ = ' '.join(word.title())
                train_data.append((word_, phon))

    max_len = max([len(p) for _, p in train_data])
    train_data.sort()
    random = Random(42)
    random.shuffle(train_data)
    val_data, train_data = train_data[:1000], train_data[1000:]
    train_data_concat = []
    for (w1, p1), (w2, p2) in zip(train_data[:-1], train_data[1:]):
        train_data_concat.append((w1, p1))
        train_data_concat.append((w1 + ' ' + w2, p1 + ' ' + p2))
    for word, phon in train_data_concat:
        print(f'{word} --- {phon}')

    print(f'train: {len(train_data_concat)}, val: {len(val_data)}, max pred len: {max_len}')
    summarizer = TransformerSummarizer(num_heads=4,
                                       feed_forward_dim=1024,
                                       num_layers=4,
                                       embedding_size=512,
                                       dropout_rate=0.)
    summarizer.optimizer = tf.keras.optimizers.Adam(1e-4)

    class PERScorer(Scorer):

        def __call__(self, prediction):
            pred = prediction['predicted_text'].split()[:-1]
            gold = prediction['preprocessed_text'][1].split()[1:-1]
            wer = float(calculate_wer(gold, pred))
            return wer

    trainer = Trainer(batch_size=32,
                      steps_per_epoch=10000,
                      max_vocab_size_encoder=1000,
                      max_vocab_size_decoder=1000,
                      use_bucketing=True,
                      preprocessor=Preprocessor(start_token='<start>', end_token='<end>',
                                                lower_case=False, hash_numbers=False,
                                                filter_pattern=None),
                      tensorboard_dir='output/tensorboard_cased_stress',
                      model_save_path='output/summarizer_cased_stress',)

    trainer.train(summarizer,
                  train_data_concat,
                  val_data=val_data,
                  num_epochs=300)