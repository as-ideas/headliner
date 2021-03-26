import re
import tqdm
from headliner.model.transformer_summarizer import TransformerSummarizer
import string

# Phonemes
_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
#_suprasegmentals = 'ː'
_suprasegmentals = 'ˈˌːˑ'
_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'

_other_symbols = 'ʍwɥʜʢʡɕʑɺɧ'
_diacrilics = 'ɚ˞ɫ'

phonemes = set(
   _vowels + _non_pulmonic_consonants + _suprasegmentals
   + _pulmonic_consonants + _other_symbols + _diacrilics)

phonemes_set = set(phonemes)


PUNCTUATION = '.,:?!;()-——–'
SYMBOLS = set(string.ascii_letters + PUNCTUATION + 'äöüÄÖÜß ')
CHARS = set(string.ascii_letters + 'äöüÄÖÜß')

import pickle


import unicodedata as ud

def rmdiacritics(char):
    if char in 'äöüÄÖÜß':
        return char
    '''
    Return the base character of char, by "removing" any
    diacritics like accents or curls and strokes and the like.
    '''
    desc = ud.name(char)
    cutoff = desc.find(' WITH ')
    if cutoff != -1:
        desc = desc[:cutoff]
        try:
            char = ud.lookup(desc)
        except KeyError:
            pass  # removing "WITH ..." produced an invalid name
    return char


def phonemize(text, summarizer, phon_dict):
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = ''.join([rmdiacritics(c) for c in text])
    text_clean = ''.join([c for c in text if c in SYMBOLS])#re.sub(u'[^a-zA-ZäöüßÄÖÜ.,:?!;()—–\- ]+', ' ', text)
    words = re.split(' ', text_clean)
    phons = []
    for word in words:
        word_split = word.split('-')
        if len(word_split) > 1:
            split_phons = [phonemize(w, summarizer, phon_dict) for w in word_split]
            phons.append('-'.join(split_phons))
            #print(f'{word_split} - {split_phons}')
        else:
            end = ''
            start = ''
            if len(word) > 0 and word[-1] in PUNCTUATION:
                end = word[-1]
                word = word[:-1]
            if len(word) > 0 and word[0] in PUNCTUATION:
                start = word[0]
                word = word[1:]
            word = ''.join([c for c in word if c in CHARS])  # re.sub(u'[^a-zA-ZäöüßÄÖÜ.,:?!;()—–\- ]+', ' ', text)
            if len(word) > 0:
                if word in phon_dict:
                    phon = phon_dict[word]
                elif word.lower() in phon_dict:
                    phon = phon_dict[word.lower()]
                elif word.title() in phon_dict:
                    phon = phon_dict[word.title()]
                else:
                    is_upper = word.isupper()
                    word = ' '.join(word)
                    phon = summarizer.predict(word).replace('<end>', '')
                    phon = phon.replace(' ', '')
                    w = word.replace(' ', '')
                    if is_upper:
                        print(f'{w} | {phon}')
            else:
                phon = ''
            phon = start + phon + end
            phons.append(phon)
    phon_line = ' '.join(phons)
    phon_line = phon_line.strip()
    return phon_line



if __name__ == '__main__':

    with open('/Users/cschaefe/datasets/nlp/heavily_cleaned_phoneme_dataset_DE.pkl', 'rb') as f:
        df = pickle.load(f)
    tuples = df[['title', 'pronunciation']]
    tuples = [tuple(x) for x in tuples.to_numpy()]
    phon_dict = {}
    for w, phon in tuples:
        p = ''.join(p for p in phon if p in phonemes)
        phon_dict[w] = p
    phon_dict['Die'] = 'diː'
    summarizer = TransformerSummarizer.load('output/summarizer_cased')
    summarizer.max_prediction_len = 50

    res = phonemize('Audioanschlüssen: (CDU) - Älter Er sei leicht konisch, was eigentlich zur Präzision Präzision der Waffe beitrage, so der Hersteller, für den der Vorgang unverständlich ist.', summarizer, phon_dict)
    metafile = '/Users/cschaefe/datasets/metadata_clean.csv'
    output = []
    with open(metafile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            line_split = line.split('|')
            id, text = line_split[0], line_split[1]
            text = text.strip()
            phons = phonemize(text, summarizer, phon_dict)
            output.append(f'{id}|{phons}\n')
            if len(output) % 10 == 0:
                with open('/Users/cschaefe/datasets/metadata_phonemized_cased_fixed.csv', 'w+', encoding='utf-8') as g:
                    g.write(''.join(output))

