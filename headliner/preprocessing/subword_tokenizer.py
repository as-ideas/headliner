from abc import abstractmethod
from typing import List
from headliner.preprocessing.tokenizer import Tokenizer
from keras_preprocessing.text import Tokenizer as KTokenizer


class KerasTokenizer(Tokenizer):

    def __init__(self, keras_tokenizer: KTokenizer):
        self._keras_tokenizer = keras_tokenizer

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        return self._keras_tokenizer.texts_to_sequences([text])[0]

    @abstractmethod
    def decode(self, sequence: List[int]) -> str:
        return self._keras_tokenizer.sequences_to_texts([sequence])[0]

    @abstractmethod
    def vocab_size(self) -> int:
        return len(self._keras_tokenizer.word_index)

