from headliner.model.transformer_summarizer import TransformerSummarizer

if __name__ == '__main__':
    summarizer = TransformerSummarizer.load('output/summarizer_test')
    word = 'CDU'

    word = ' '.join(word)
    pred = summarizer.predict(word)
    pred = pred.replace(' ', '')
    word = word.replace(' ', '')
    print(f'{word} | {pred}')