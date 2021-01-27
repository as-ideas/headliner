import json
import re
import tensorflow as tf
from functools import _lru_cache_wrapper
from random import Random

from thinc.neural.optimizers import Adam

from headliner.model.transformer_summarizer import TransformerSummarizer
from headliner.trainer import Trainer
import pickle

# Phonemes
_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
_suprasegmentals = 'ː'
_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'

_other_symbols = 'ʍwɥʜʢʡɕʑɺɧ'
_diacrilics = 'ɚ˞ɫ'

phonemes = set(
   _vowels + _non_pulmonic_consonants + _suprasegmentals
   + _pulmonic_consonants + _other_symbols + _diacrilics)

phonemes_set = set(phonemes)

if __name__ == '__main__':

    with open('/Users/cschaefe/datasets/nlp/heavily_cleaned_phoneme_dataset_DE.pkl', 'rb') as f:
        df = pickle.load(f)
    tuples = df[['title', 'pronunciation']]
    tuples = [tuple(x) for x in tuples.to_numpy()]
    train_data = []
    max_len = 50
    for word, phon in tuples:
        if 0 < len(phon) < max_len and ' ' not in word and 0 < len(word) < max_len:
            word = ' '.join(word)
            phon = ' '.join(p for p in phon if p in phonemes)
            train_data.append((word, phon))



    max_len = max([len(p) for _, p in train_data])
    train_data.sort()
    random = Random(42)
    random.shuffle(train_data)
    train_data_concat = []
    for (w1, p1), (w2, p2) in zip(train_data[:-1], train_data[1:]):
        train_data_concat.append((w1, p1))
        train_data_concat.append((w1 + ' ' + w2, p1 + ' ' + p2))

    for word, phon in train_data_concat:
        print(f'{word} --- {phon}')

    val_data, train_data = train_data_concat[:1000], train_data_concat[1000:]
    print(f'train: {len(train_data)}, val: {len(val_data)}, max pred len: {max_len}')
    summarizer = TransformerSummarizer(num_heads=4,
                                       feed_forward_dim=1024,
                                       num_layers=4,
                                       embedding_size=512,
                                       dropout_rate=0.,
                                       max_prediction_len=max_len*2)
    summarizer.optimizer = tf.keras.optimizers.Adam(1e-4)
    trainer = Trainer(batch_size=32,
                      steps_per_epoch=500,
                      max_vocab_size_encoder=1000,
                      max_vocab_size_decoder=1000,
                      use_bucketing=True,
                      tensorboard_dir='output/tensorboard_large',
                      model_save_path='output/summarizer_large')

    trainer.train(summarizer, train_data, val_data=val_data, num_epochs=300)