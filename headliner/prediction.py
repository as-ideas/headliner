import numpy as np

from headliner.model.summarizer_attention import SummarizerAttention

if __name__ == '__main__':
    path_to_model = '//Users/cschaefe/saved_models/mod4'
    summarizer = SummarizerAttention.load(path_to_model)
    summarizer.vectorizer.max_output_len = 20
    summarizer.max_prediction_len=20

    while True:
        text = input('\nEnter text: ')
        prediction_vecs = summarizer.predict_vectors(text, '')
        alignment = prediction_vecs['alignment']
        tokens_input = prediction_vecs['preprocessed_text'][0].split()
        print('\n')
        print(prediction_vecs['predicted_text'])
        print('\n')
        for t in range(len(alignment)):
            al = alignment[t]
            top_att = np.argsort(al)[::-1]
            top_ind = top_att.tolist()
            top_tokens = [tokens_input[i] for i in top_ind][:5]
            print('{} {}'.format(prediction_vecs['predicted_text'].split()[t], top_tokens))
        sum_alignment = sum(alignment)
        top_att = np.argsort(sum_alignment)[::-1]
        top_ind = top_att.tolist()
        top_tokens = [tokens_input[i] for i in top_ind][:10]
        print()
        print(top_tokens)
        print()
