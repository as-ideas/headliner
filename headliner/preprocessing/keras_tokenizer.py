from typing import List, Iterable, Dict

from keras_preprocessing.text import Tokenizer as KTokenizer

from headliner.preprocessing.tokenizer import Tokenizer


class KerasTokenizer(Tokenizer):

    def __init__(self, **kwargs):
        self._keras_tokenizer = KTokenizer(**kwargs)

    def encode(self, text: str) -> List[int]:
        return self._keras_tokenizer.texts_to_sequences([text])[0]

    def decode(self, sequence: List[int]) -> str:
        return self._keras_tokenizer.sequences_to_texts([sequence])[0]

    @property
    def vocab_size(self) -> int:
        return len(self._keras_tokenizer.word_index)

    def fit(self, texts: Iterable[str]):
        self._keras_tokenizer.fit_on_texts(texts)

    @property
    def token_index(self) -> Dict[str, int]:
        return self._keras_tokenizer.word_index
