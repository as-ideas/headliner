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

    with open('/Users/cschaefe/datasets/nlp/phon_dict_all.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    train_data = []
    max_len = 50

    for data in data_dict:
        word = data['title']
        phon = data['pronunciation']
        if 0 < len(phon) < max_len and ' ' not in word and 0 < len(word) < max_len:
            word = re.sub('[^a-zA-Zäöüß ]+', ' ', word)
            word = ' '.join(word)
            phon = ' '.join(p for p in phon if p in phonemes)
            print(f'{word} --- {phon}')
            train_data.append((word, phon))

    max_len = max([len(p) for _, p in train_data])
    train_data.sort()
    random = Random(42)
    random.shuffle(train_data)
    val_data, train_data = train_data[:1000], train_data[1000:]
    print(f'train: {len(train_data)}, val: {len(val_data)}, max pred len: {max_len}')
    summarizer = TransformerSummarizer(num_heads=4,
                                       feed_forward_dim=1024,
                                       num_layers=4,
                                       embedding_size=512,
                                       dropout_rate=0.,
                                       max_prediction_len=max_len+2)
    summarizer.optimizer = tf.keras.optimizers.Adam(1e-4)
    trainer = Trainer(batch_size=32,
                      steps_per_epoch=500,
                      max_vocab_size_encoder=1000,
                      max_vocab_size_decoder=1000,
                      use_bucketing=True,
                      tensorboard_dir='output/tensorboard_large',
                      model_save_path='output/summarizer_large')

    trainer.train(summarizer, train_data, val_data=val_data, num_epochs=300)