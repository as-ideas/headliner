# Neural Machine Translation Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/as-ideas/headliner/blob/master/notebooks/Neural_Machine_Translation_Example.ipynb)

### Install the package via PyPI
```bash
pip install headliner
```

### Download the German-English sentence pairs
```bash
wget http://www.manythings.org/anki/deu-eng.zip
unzip deu-eng.zip
head deu.txt
```

### Create the dataset
```python
import io

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[w for w in l.split('\t')]  for l in lines[:num_examples]]
    return zip(*word_pairs)

eng, ger = create_dataset('deu.txt', 500)
data = list(zip(ger, eng))
```

### Split the dataset into train and test
````python
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.1)
````

### Define the model and train it
````python
from headliner.trainer import Trainer
from headliner.model.summarizer_attention import SummarizerAttention

summarizer = SummarizerAttention(lstm_size=64, embedding_size=24)
trainer = Trainer(batch_size=32, steps_per_epoch=100, steps_to_log=20, model_save_path='/tmp/summarizer')
trainer.train(summarizer, train, num_epochs=10, val_data=test)
````

### Do some prediction
```python
summarizer.predict('Hi.')
```