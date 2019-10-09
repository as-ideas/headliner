from headliner.model.summarizer_transformer import SummarizerTransformer

if __name__ == '__main__':
    path_to_model = '/tmp/summarizer_20191008_165615'
    summarizer = SummarizerTransformer.load(path_to_model)

    while True:
        text = input('\nEnter text: ')
        prediction_vecs = summarizer.predict_vectors(text, '')
        tokens_input = prediction_vecs['preprocessed_text'][0].split()
        print('\n')
        print(prediction_vecs['predicted_text'])
        print('\n')
        """
        alignment = prediction_vecs['alignment']
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
        """