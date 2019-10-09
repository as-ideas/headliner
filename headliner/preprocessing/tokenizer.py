import abc
from abc import abstractmethod
from typing import List, Iterable


class Tokenizer(abc.ABC):

    @abstractmethod
    def encode(self, string: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, sequence: List[int]) -> str:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def fit(self, texts: Iterable[str]):
        pass