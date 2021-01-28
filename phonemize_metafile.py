import re
import tqdm
from headliner.model.transformer_summarizer import TransformerSummarizer

PUNCTUATION = '.,:?!;()-'


def phonemize(text, summarizer):
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    words = text.split()
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
            word = ' '.join(word)
            phon = summarizer.predict(word).replace('<end>', '')
            phon = phon.replace(' ', '')
        else:
            phon = ''
        phon = start + phon + end
        phons.append(phon)
    return ' '.join(phons)



if __name__ == '__main__':
    summarizer = TransformerSummarizer.load('output/summarizer_large')
    metafile = '/Users/cschaefe/datasets/metadata_clean.csv'
    output = []
    with open(metafile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            line_split = line.split('|')
            id, text = line_split[0], line_split[1]
            phons = phonemize(text, summarizer)
            print(f'{id} | {text} | {phons}')
            output.append(f'{id}|{phons}\n')
            if len(output) % 10 == 0:
                with open('/Users/cschaefe/datasets/metadata_phonemized.csv', 'w+', encoding='utf-8') as g:
                    g.write(''.join(output))

