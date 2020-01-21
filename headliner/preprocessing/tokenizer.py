import abc
from abc import abstractmethod
from typing import List


class Tokenizer(abc.ABC):
    """
    Encodes text to sequences and decodes sequences to text.
    """

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Encodes a given string into a sequence of indices.

        Args:
            text: Text to encode.

        Returns: Encoded sequence.
        """
        pass

    @abstractmethod
    def decode(self, sequence: List[int]) -> str:
        """
        Decodees a given sequence into a text.

        Args:
            sequence: Sequence to decode.

        Returns: Decoded text.

        """
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        Size of token vocab.
        """
        pass
