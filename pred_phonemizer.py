from headliner.model.transformer_summarizer import TransformerSummarizer

if __name__ == '__main__':
    summarizer = TransformerSummarizer.load('output/summarizer')

    word = 'Knochenmarksspendenverein'

    word = ' '.join(word)
    pred = summarizer.predict(word)

    print(f'{pred}')