# Neural Machine Translation Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/as-ideas/headliner/blob/master/notebooks/Advanced_Neural_Machine_Translation_Example.ipynb)

### Install TensorFlow and also our package via PyPI
```bash
pip install tensorflow-gpu==2.0.0
pip install headliner
```

### Download the German-English sentence pairs
```bash
wget http://www.manythings.org/anki/deu-eng.zip
unzip deu-eng.zip
head deu.txt
```

### Create the dataset but only take a subset for faster training
```
import io

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[w for w in l.split('\t')]  for l in lines[:num_examples]]
    return zip(*word_pairs)

eng, ger = create_dataset('deu.txt', 30000)
data = list(zip(ger, eng))
data[:10]
```

### Split the dataset into train and test
```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.1)
```

### Define custom preprocessing
```
from headliner.preprocessing import Preprocessor
preprocessor = Preprocessor(lower_case=True)
train_prep = [preprocessor(t) for t in train]
train_prep[:5]
```

### Fit custom tokenizers for input and target
```
from tensorflow_datasets.core.features.text import SubwordTextEncoder
from headliner.preprocessing import Vectorizer
inputs_prep = [t[0] for t in train_prep]
targets_prep = [t[1] for t in train_prep]
tokenizer_input = SubwordTextEncoder.build_from_corpus(
    inputs_prep, target_vocab_size=2**13)
tokenizer_target = SubwordTextEncoder.build_from_corpus(
    targets_prep, target_vocab_size=2**13, 
    reserved_tokens=[preprocessor.start_token, preprocessor.end_token])

vectorizer = Vectorizer(tokenizer_input, tokenizer_target)
'vocab size input {}, target {}'.format(
    vectorizer.encoding_dim, vectorizer.decoding_dim)
```

### Start tensorboard
```
%load_ext tensorboard
%tensorboard --logdir /tmp/summarizer_tensorboard
```

### Define the model and train it
```
from headliner.model.summarizer_transformer import SummarizerTransformer
from headliner.trainer import Trainer
summarizer = SummarizerTransformer(num_heads=1,
                                   feed_forward_dim=1024,
                                   num_layers=1,
                                   embedding_size=64,
                                   dropout_rate=0.1,
                                   max_prediction_len=50)
summarizer.init_model(preprocessor, vectorizer)
trainer = Trainer(steps_per_epoch=250,
                  batch_size=64,
                  model_save_path='/tmp/summarizer_transformer',
                  tensorboard_dir='/tmp/summarizer_tensorboard',
                  steps_to_log=50)
trainer.train(summarizer, train, num_epochs=10, val_data=test)
```

### Do some prediction
```
summarizer.predict('Wie geht es dir?')
```

### Continue training to improve the model and check the BLEU score
```
from headliner.evaluation import BleuScorer
bleu_scorer = BleuScorer(tokens_to_ignore=[preprocessor.start_token, 
                                           preprocessor.end_token])
trainer.train(best_summarizer, 
              train, 
              num_epochs=30, 
              val_data=test, 
              scorers={'bleu': bleu_scorer}
```
