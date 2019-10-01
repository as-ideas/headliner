from typing import Tuple, List

from keras_preprocessing.text import Tokenizer


class Vectorizer:
    """
    Transforms tuples of text into tuples of vector sequences.
    """

    def __init__(self,
                 tokenizer_encoder: Tokenizer,
                 tokenizer_decoder: Tokenizer) -> None:
        self.encoding_dim = len(tokenizer_encoder.index_word) + 1
        self.decoding_dim = len(tokenizer_decoder.index_word) + 1
        self._tokenizer_encoder = tokenizer_encoder
        self._tokenizer_decoder = tokenizer_decoder

    def __call__(self, data: Tuple[str, str]) -> Tuple[List[int], List[int]]:
        """
        Encodes preprocessed strings into sequences of one-hot indices
        """
        text_encoder, text_decoder = data[0], data[1]
        vec_encoder = self._tokenizer_encoder.texts_to_sequences([text_encoder])[0]
        vec_decoder = self._tokenizer_decoder.texts_to_sequences([text_decoder])[0]
        return vec_encoder, vec_decoder

    def decode_input(self, sequence: List[int]) -> str:
        return self._tokenizer_encoder.sequences_to_texts([sequence])[0]

    def decode_output(self, sequence: List[int]) -> str:
        return self._tokenizer_decoder.sequences_to_texts([sequence])[0]
