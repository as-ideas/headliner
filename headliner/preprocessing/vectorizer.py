from typing import Tuple, List

from keras_preprocessing.text import Tokenizer


class Vectorizer:
    """
    Transforms tuples of text into tuples of vector sequences.
    """

    def __init__(self,
                 tokenizer_encoder: Tokenizer,
                 tokenizer_decoder: Tokenizer,
                 max_output_len) -> None:
        self.encoding_dim = len(tokenizer_encoder.index_word) + 1
        self.decoding_dim = len(tokenizer_decoder.index_word) + 1
        self.max_output_len = max_output_len
        self._tokenizer_encoder = tokenizer_encoder
        self._tokenizer_decoder = tokenizer_decoder

    def __call__(self, data: Tuple[str, str]) -> Tuple[List[int], List[int]]:
        """
        Encodes preprocessed strings into sequences of one-hot indices.
        """
        text_encoder, text_decoder = data[0], data[1]
        vec_encoder = self._tokenizer_encoder.texts_to_sequences([text_encoder])[0]
        vec_decoder = self._tokenizer_decoder.texts_to_sequences([text_decoder])[0]
        if len(vec_decoder) > self.max_output_len:
            vec_decoder = vec_decoder[:self.max_output_len-1] + vec_decoder[-1]
        else:
            vec_decoder = vec_decoder + [0] * (self.max_output_len - len(vec_decoder))

        return vec_encoder, vec_decoder

    def encode_input(self, text: str) -> List[int]:
        return self._tokenizer_encoder.texts_to_sequences([text])[0]

    def encode_output(self, text: str) -> List[int]:
        return self._tokenizer_decoder.texts_to_sequences([text])[0]

    def decode_input(self, sequence: List[int]) -> str:
        return self._tokenizer_encoder.sequences_to_texts([sequence])[0]

    def decode_output(self, sequence: List[int]) -> str:
        return self._tokenizer_decoder.sequences_to_texts([sequence])[0]
