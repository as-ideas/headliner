# BERT Neural Machine Translation Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/as-ideas/headliner/blob/master/notebooks/BERT_Translation_Example.ipynb)

### Upgrade grpcio which is needed by tensorboard 2.0.2
```bash
!pip install --upgrade grpcio
```

### Install TensorFlow and also our package via PyPI
```bash
pip install tensorflow-gpu==2.0.0
pip install headliner
```

### Download the German-English sentence pairs
```bash
wget http://www.manythings.org/anki/deu-eng.zip
unzip deu-eng.zip
```

### Create the dataset but only take a subset for faster training
```python
import io

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[w for w in l.split('\t')[:2]]  for l in lines[:num_examples]]
    return zip(*word_pairs)

eng, ger, meta = create_dataset('deu.txt', 200000)
data = list(zip(eng, ger))
```

### Split the dataset into train and test
```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=100)
```

### Define custom preprocessing
```python
from headliner.preprocessing.bert_preprocessor import BertPreprocessor
from spacy.lang.en import English

preprocessor = BertPreprocessor(nlp=English())
train_prep = [preprocessor(t) for t in train]
train_prep[:5]
```

### Create custom tokenizers for input and target
```python
from tensorflow_datasets.core.features.text import SubwordTextEncoder
from transformers import BertTokenizer
from headliner.preprocessing.bert_vectorizer import BertVectorizer

targets_prep = [t[1] for t in train_prep]
tokenizer_input = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_target = SubwordTextEncoder.build_from_corpus(
    targets_prep, target_vocab_size=2**13, 
    reserved_tokens=[preprocessor.start_token, preprocessor.end_token])

vectorizer = BertVectorizer(tokenizer_input, tokenizer_target)
'vocab size input {}, target {}'.format(
    vectorizer.encoding_dim, vectorizer.decoding_dim)
```

### Start tensorboard
```
%load_ext tensorboard
%tensorboard --logdir /tmp/bert_tensorboard
```

### Define the model and train it
```python
# Define the model and train it
# You need to be quite patient, since the model has a lot of params
import tensorflow as tf
from headliner.model.bert_summarizer import BertSummarizer
from headliner.trainer import Trainer

summarizer = BertSummarizer(num_heads=8,
                            feed_forward_dim=1024,
                            num_layers_encoder=0,
                            num_layers_decoder=4,
                            bert_embedding_encoder='bert-base-uncased',
                            embedding_encoder_trainable=False,
                            embedding_size_encoder=768,
                            embedding_size_decoder=768,
                            dropout_rate=0,
                            max_prediction_len=50)
# Adjust learning rates of encoder and decoder optimizer schedules
# You may want to try different learning rates and observe the loss
summarizer.optimizer_decoder = BertSummarizer.new_optimizer_decoder(
    learning_rate_start=1e-2
)
summarizer.optimizer_encoder = BertSummarizer.new_optimizer_encoder(
    learning_rate_start=5e-4
)
summarizer.init_model(preprocessor, vectorizer)
trainer = Trainer(steps_per_epoch=5000,
                  batch_size=16,
                  model_save_path='/tmp/bert_summarizer',
                  tensorboard_dir='/tmp/bert_tensorboard',
                  steps_to_log=10)
trainer.train(summarizer, train, num_epochs=200, val_data=test)
```

### Load best model and do some prediction
```python
best_summarizer = BertSummarize.load('/tmp/bert_summarizer')
best_summarizer.predict('Do you like robots?')
```

### Plot attention alignment for a prediction
```python
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_attention_weights(summarizer, pred_vectors, layer_name):
    fig = plt.figure(figsize=(16, 8))
    input_text, _ = pred_vectors['preprocessed_text']
    input_sequence = summarizer.vectorizer.encode_input(input_text)
    pred_sequence = pred_vectors['predicted_sequence']
    attention = tf.squeeze(pred_vectors['attention_weights'][layer_name])
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(1, 2, head + 1)
        ax.matshow(attention[head][:-1, :], cmap='viridis')
        fontdict = {'fontsize': 10}
        ax.set_xticks(range(len(input_sequence)))
        ax.set_yticks(range(len(pred_sequence)))
        ax.set_ylim(len(pred_sequence) - 1.5, -0.5)
        ax.set_xticklabels(
            [summarizer.vectorizer.decode_input([i]) for i in input_sequence],
            fontdict=fontdict,
            rotation=90)
        ax.set_yticklabels([summarizer.vectorizer.decode_output([i]) 
                            for i in pred_sequence], fontdict=fontdict)
        ax.set_xlabel('Head {}'.format(head + 1))
    plt.tight_layout()
    plt.show()

pred_vectors = best_summarizer.predict_vectors(
    'Tom ran out of the burning house.', '')
plot_attention_weights(best_summarizer, pred_vectors, 'decoder_layer4_block2')
```

