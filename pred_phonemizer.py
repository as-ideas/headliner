import re
import numpy as np
import math
import tensorflow as tf
from headliner.model.transformer_summarizer import TransformerSummarizer

if __name__ == '__main__':
    summarizer = TransformerSummarizer.load('output/summarizer_cased')
    summarizer.max_prediction_len = 50
    word = 'Studentengruppe'
    word = ' '.join(word)
    pred_vecs = summarizer.predict_vectors(word, '')
    pred = pred_vecs['predicted_text']
    pred = pred.replace(' ', '')
    word = word.replace(' ', '')
    print(f'{word} | {pred}')

    logits = pred_vecs['logits']
    logits = [tf.convert_to_tensor(t) for t in logits]
    logits = [tf.nn.log_softmax(t).numpy() for t in logits]
    max_index = np.argmax(logits, axis=1)
    log_prob = sum([logits[i][ind] for i, ind in enumerate(max_index)])
    print(f'log prob {log_prob}, prob {math.exp(log_prob)}')