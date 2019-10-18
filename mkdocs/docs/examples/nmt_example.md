# Neural Machine Translation Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/as-ideas/headliner/blob/master/notebooks/Neural_Machine_Translation_Example.ipynb)

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

eng, ger = create_dataset('deu.txt', 30000)
data = list(zip(eng, ger))
```

### Split the dataset into train and test
```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=100)
```

### Define the model and train it
```python
from headliner.trainer import Trainer
from headliner.model.summarizer_attention import SummarizerAttention

summarizer = SummarizerAttention(lstm_size=1024, embedding_size=256)
trainer = Trainer(batch_size=64, 
                  steps_per_epoch=100, 
                  steps_to_log=20, 
                  max_output_len=10, 
                  model_save_path='/tmp/summarizer')
trainer.train(summarizer, train, num_epochs=10, val_data=test)
```

### Do some prediction
```python
summarizer.predict('How are you?')
```