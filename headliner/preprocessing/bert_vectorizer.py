from typing import Tuple, List

from transformers import BertTokenizer

from headliner.preprocessing.tokenizer import Tokenizer


class BertVectorizer:
    """
    Transforms tuples of text into tuples of vector sequences.
    """

    def __init__(self,
                 tokenizer_encoder: BertTokenizer,
                 tokenizer_decoder: Tokenizer,
                 max_input_len=512,
                 max_output_len=None) -> None:
        """
        Initializes the vectorizer.

        Args:
            tokenizer_encoder: Tokenizer that encodes the input text.
            tokenizer_decoder: Tokenizer that encodes the target text.
            max_input_len (optional): Maximum length of input sequence,
                longer sequences will be truncated.
            max_output_len (optional): Maximum length of target sequence,
                longer sequences will be truncated and shorter sequences
                will be padded to max len.
        """
        self.encoding_dim = tokenizer_encoder.vocab_size + 1
        self.decoding_dim = tokenizer_decoder.vocab_size + 1
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self._tokenizer_encoder = tokenizer_encoder
        self._tokenizer_decoder = tokenizer_decoder

    def __call__(self, data: Tuple[str, str]) -> Tuple[List[int], List[int], List[int]]:
        """
        Encodes preprocessed strings into sequences of one-hot indices.
        """
        text_encoder, text_decoder = data[0], data[1]
        sentences = text_encoder.split('[SEP]')[:-1]
        vec_encoder = []
        sentence_ids = []
        for i, sent in enumerate(sentences):
            sent = sent + ' [SEP]'
            vec = self._tokenizer_encoder.encode(sent, add_special_tokens=False)
            if len(vec_encoder) + len(vec) < self.max_input_len:
                vec_encoder.extend(vec)
                ids = [i % 2] * len(vec)
                sentence_ids.extend(ids)

        vec_decoder = self._tokenizer_decoder.encode(text_decoder)
        if self.max_output_len is not None:
            if len(vec_decoder) > self.max_output_len:
                vec_decoder = vec_decoder[:self.max_output_len - 1] + [vec_decoder[-1]]
            else:
                vec_decoder = vec_decoder + [0] * (self.max_output_len - len(vec_decoder))

        return vec_encoder, sentence_ids, vec_decoder

    def encode_input(self, text: str) -> List[int]:
        return self._tokenizer_encoder.encode(text, add_special_tokens=False)

    def encode_output(self, text: str) -> List[int]:
        return self._tokenizer_decoder.encode(text)

    def decode_input(self, sequence: List[int]) -> str:
        return self._tokenizer_encoder.decode(sequence)

    def decode_output(self, sequence: List[int]) -> str:
        return self._tokenizer_decoder.decode(sequence)
