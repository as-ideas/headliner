import re
import tqdm
from headliner.model.transformer_summarizer import TransformerSummarizer


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


PUNCTUATION = '.,:?!;()-'

import pickle


def phonemize(text, summarizer, phon_dict):
    text = text.replace('-', ' ')
    text = text.replace('—', ' ')
    text = text.replace('–', ' ')
    words = re.split(' ', text)
    phons = []
    for word in words:
        end = ''
        start = ''
        if len(word) > 0 and word[-1] in PUNCTUATION:
            end = word[-1]
            word = word[:-1]
        if len(word) > 0 and word[0] in PUNCTUATION:
            start = word[0]
            word = word[1:]
        word = re.sub('[^a-zA-Zäöüß ]+', '', word)
        if len(word) > 0:
            if word in phon_dict:
                phon = phon_dict[word]
            elif word.lower() in phon_dict:
                phon = phon_dict[word.lower()]
            elif word.title() in phon_dict:
                phon = phon_dict[word.title()]
            else:
                word = ' '.join(word)
                phon = summarizer.predict(word).replace('<end>', '')
                phon = phon.replace(' ', '')
                w = word.replace(' ', '')
                print(f'{w} | {phon}')
        else:
            phon = ''
        phon = start + phon + end
        phons.append(phon)
    return ' '.join(phons)



if __name__ == '__main__':

    with open('/Users/cschaefe/datasets/nlp/heavily_cleaned_phoneme_dataset_DE.pkl', 'rb') as f:
        df = pickle.load(f)
    tuples = df[['title', 'pronunciation']]
    tuples = [tuple(x) for x in tuples.to_numpy()]
    phon_dict = {}
    for w, phon in tuples:
        p = ''.join(p for p in phon if p in phonemes)
        phon_dict[w] = p

    summarizer = TransformerSummarizer.load('output/summarizer_test')
    summarizer.max_prediction_len = 50

    metafile = '/Users/cschaefe/datasets/metadata_clean.csv'
    output = []
    with open(metafile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            line_split = line.split('|')
            id, text = line_split[0], line_split[1]
            phons = phonemize(text, summarizer, phon_dict)
            output.append(f'{id}|{phons}\n')
            if len(output) % 10 == 0:
                with open('/Users/cschaefe/datasets/metadata_phonemized.csv', 'w+', encoding='utf-8') as g:
                    g.write(''.join(output))

