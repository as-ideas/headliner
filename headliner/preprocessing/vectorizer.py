from typing import Tuple, List

from headliner.preprocessing.tokenizer import Tokenizer


class Vectorizer:
    """
    Transforms tuples of text into tuples of vector sequences.
    """

    def __init__(self,
                 tokenizer_encoder: Tokenizer,
                 tokenizer_decoder: Tokenizer,
                 max_output_len=None) -> None:
        """
        Initializes the vectorizer.

        Args:
            tokenizer_encoder: Tokenizer that encodes the input text.
            tokenizer_decoder: Tokenizer that encodes the target text.
            max_output_len (optional): Maximum length of target sequence,
                longer sequences will be truncated.
        """
        self.encoding_dim = tokenizer_encoder.vocab_size + 1
        self.decoding_dim = tokenizer_decoder.vocab_size + 1
        self.max_output_len = max_output_len
        self._tokenizer_encoder = tokenizer_encoder
        self._tokenizer_decoder = tokenizer_decoder

    def __call__(self, data: Tuple[str, str]) -> Tuple[List[int], List[int]]:
        """
        Encodes preprocessed strings into sequences of one-hot indices.
        """
        text_encoder, text_decoder = data[0], data[1]
        vec_encoder = self._tokenizer_encoder.encode(text_encoder)
        vec_decoder = self._tokenizer_decoder.encode(text_decoder)
        if self.max_output_len is not None:
            if len(vec_decoder) > self.max_output_len:
                vec_decoder = vec_decoder[:self.max_output_len-1] + [vec_decoder[-1]]
            else:
                vec_decoder = vec_decoder + [0] * (self.max_output_len - len(vec_decoder))

        return vec_encoder, vec_decoder

    def encode_input(self, text: str) -> List[int]:
        return self._tokenizer_encoder.encode(text)

    def encode_output(self, text: str) -> List[int]:
        return self._tokenizer_decoder.encode(text)

    def decode_input(self, sequence: List[int]) -> str:
        return self._tokenizer_encoder.decode(sequence)

    def decode_output(self, sequence: List[int]) -> str:
        return self._tokenizer_decoder.decode(sequence)
