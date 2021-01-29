from headliner.model.transformer_summarizer import TransformerSummarizer

if __name__ == '__main__':
    summarizer = TransformerSummarizer.load('output/summarizer_test')
    summarizer.max_prediction_len = 50



    word = 'Jahrelang'
    word = ' '.join(word)
    pred = summarizer.predict(word)
    pred = pred.replace(' ', '').replace('<end>', '')
    word = word.replace(' ', '')
    print(f'{word} | {pred}')