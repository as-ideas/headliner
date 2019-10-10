# Headliner

[![Build Status](https://travis-ci.org/as-ideas/headliner.svg?branch=master)](https://travis-ci.org/as-ideas/headliner)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://as-ideas.github.io/headliner/)
[![codecov](https://codecov.io/gh/as-ideas/headliner/branch/master/graph/badge.svg)](https://codecov.io/gh/as-ideas/headliner)
[![PyPI Version](https://img.shields.io/pypi/v/headliner)](https://pypi.org/project/headliner/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/as-ideas/headliner/blob/master/LICENSE)

The goal of this project is to generate headlines from news articles.

In particular, we use sequence-to-sequence (seq2seq) under the hood, 
an encoder-decoder framework. We provide a very simple interface to train 
and deploy seq2seq models. Although this library was created internally to 
generate headlines, you can also use it for other tasks like machine translations,
text summarization and many more.

![Seq2seq architecture](figures/seq2seq.jpg)

We built this library with the following goals in mind:

* Easy-to-use for training and deployment
* Uses TensorFlow 2.0
* Modular classes: text preprocessing, modeling, evaluation
* Extensible for different encoder-decoder models
* Works on large text data

Read the documentation at: [https://as-ideas.github.io/headliner/](https://as-ideas.github.io/headliner/)

Headliner is compatible with Python 3.6 and is distributed under the MIT license.

## ⚙️ Installation
> ⚠️ Before installing Headliner, you need to install TensorFlow as we use this as our deep learning framework. For more 
> details on how to install it, have a look at the [TensorFlow installation instructions](https://www.tensorflow.org/install/).

Then you can install Headliner itself. There are two ways to install Headliner:

* Install Headliner from PyPI (recommended):
```
pip install headliner
```

* Install Headliner from the GitHub source:
```
git clone https://github.com/as-ideas/headliner.git
cd headliner
python setup.py install
```

## 📖 Usage 

### Training

```
from headliner.trainer import Trainer
from headliner.model.summarizer_transformer import SummarizerTransformer

data = [('You are the stars, earth and sky for me!', 'I love you.'),
        ('You are great, but I have other plans.', 'I like you.')]

summarizer = SummarizerTransformer(embedding_size=64, max_prediction_len=20)
trainer = Trainer(batch_size=2, steps_per_epoch=100)
trainer.train(summarizer, data, num_epochs=2)
summarizer.save('/tmp/summarizer')
```

### Prediction

```
from headliner.model.summarizer_transformer import SummarizerTransformer

summarizer = SummarizerTransformer.load('/tmp/summarizer')
summarizer.predict('You are the stars, earth and sky for me!')
```

### Advanced training

Training using a validation split and model checkpointing:

```
from headliner.model.summarizer_attention import SummarizerTransformer
from headliner.trainer import Trainer

train_data = [('You are the stars, earth and sky for me!', 'I love you.'),
              ('You are great, but I have other plans.', 'I like you.')]*1000
val_data = [('You are great, but I have other plans.', 'I like you.')] * 8

summarizer = SummarizerTransformer(num_heads=1,
                                   feed_forward_dim=512,
                                   num_layers=1,
                                   embedding_size=64,
                                   max_prediction_len=50)
trainer = Trainer(batch_size=8,
                  steps_per_epoch=50,
                  max_vocab_size=10000,
                  tensorboard_dir='/tmp/tensorboard',
                  model_save_path='/tmp/summarizer')

trainer.train(summarizer, train_data, val_data=val_data, num_epochs=3)
```

### Advanced prediction
Prediction information such as attention weights and logits can be accessed via predict_vectors returning a dictionary:
```
from headliner.model.summarizer_attention import SummarizerTransformer

summarizer = SummarizerTransformer.load('/tmp/summarizer')
summarizer.predict_vectors('You are the stars, earth and sky for me!')
```

### Resume training

A previously trained summarizer can be loaded and then retrained. In this case the data preprocessing and vectorization is loaded from the model.
```
train_data = [('Some new training data.', 'New data.')]*1000

summarizer_loaded = SummarizerTransformer.load('/tmp/summarizer')
trainer = Trainer(batch_size=2)
trainer.train(summarizer, train_data)
summarizer_loaded.save('/tmp/summarizer_retrained')
```

### Custom preprocessing

A model can be initialized with custom string cleanup and tokenization:

```
from headliner.preprocessing import Preprocessor

train_data = [('Some inputs.', 'Some outputs.')]*1000

preprocessor = Preprocessor(filter_pattern='', 
                            lower_case=True, 
                            hash_numbers=False)
train_prep = [custom_preprocessor(t) for t in train_data]
inputs_prep = [t[0] for t in train_prep]
targets_prep = [t[1] for t in train_prep]

# Build tf subword tokenizers. Other custom tokenizers can be implemented 
# by subclassing headliner.preprocessing.Tokenizer

tokenizer_input = SubwordTextEncoder.build_from_corpus(
    inputs_prep, target_vocab_size=2**13)
tokenizer_target = SubwordTextEncoder.build_from_corpus(
    targets_prep, target_vocab_size=2**13)

vectorizer = Vectorizer(tokenizer_input, tokenizer_target)
summarizer = SummarizerTransformer(embedding_size=64, max_prediction_len=50)
summarizer.init_model(preprocessor, vectorizer)

trainer = Trainer(batch_size=2)
trainer.train(summarizer, train_data, num_epochs=3)
```


### Training on large datasets

Large datasets can be handled by using an iterator:
```

def read_data_iteratively():
    return (('Some a b inputs.', 'Some a b c d e outputs.') for _ in range(1000))

class DataIterator:
    def __iter__(self):
        return read_data_iteratively()

data_iter = DataIterator()

summarizer = SummarizerTransformer(embedding_size=10, max_prediction_len=20)
trainer = Trainer(batch_size=16, steps_per_epoch=1000)
trainer.train(summarizer, data_iter, num_epochs=3)
```

## 🤝 Contribute
We welcome all kinds of contributions.
See the [Contribution](CONTRIBUTING.md) guide for more details.

## 📝 Cite this work
Please cite Headliner in your publications if this is useful for your research. Here is an example BibTeX entry:
```
@misc{axelspringerai2019headliners,
  title={Headliner},
  author={Christian Schäfer & Dat Tran},
  year={2019},
  howpublished={\url{https://github.com/as-ideas/headliner}},
}
```

## 🏗 Maintainers
* Christian Schäfer, github: [cschaefer26](https://github.com/cschaefer26)
* Dat Tran, github: [datitran](https://github.com/datitran)

## © Copyright

See [LICENSE](LICENSE) for details.