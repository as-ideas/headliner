from typing import Tuple, List

from spacy.lang.de import German

from headliner.preprocessing.tokenizer import Tokenizer
import spacy
import re

class Vectorizer:
    """
    Transforms tuples of text into tuples of vector sequences.
    """

    def __init__(self,
                 tokenizer_encoder: Tokenizer,
                 tokenizer_decoder: Tokenizer,
                 max_input_len=None,
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
        self.nlp = German()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        self.count = 0

    def __call__(self, data: Tuple[str, str]) -> Tuple[List[int], List[int], List[int]]:
        """
        Encodes preprocessed strings into sequences of one-hot indices.
        """
        text_encoder, text_decoder = data[0], data[1]
        doc = self.nlp(text_encoder)
        sentences = [sent.string.strip() for sent in doc.sents]
        vec_encoder = []
        sentence_ids = []
        self.count += 1
        for i, sent in enumerate(sentences):
            sent = '[CLS] ' + sent + ' [SEP]'
            vec = self._tokenizer_encoder.encode(sent)
            if len(vec_encoder) + len(vec) < self.max_input_len:
                vec_encoder.extend(vec)
                ids = [i % 2] * len(vec)
                sentence_ids.extend(ids)

        #print(vec_encoder)
        vec_decoder = self._tokenizer_decoder.encode(text_decoder)
        if self.max_input_len is not None:
            if len(vec_encoder) > self.max_input_len:
                vec_encoder = vec_encoder[:self.max_input_len-1] + [vec_encoder[-1]]
            if len(sentence_ids) > self.max_input_len:
                sentence_ids = sentence_ids[:self.max_input_len-1] + [sentence_ids[-1]]
        if self.max_output_len is not None:
            if len(vec_decoder) > self.max_output_len:
                vec_decoder = vec_decoder[:self.max_output_len-1] + [vec_decoder[-1]]
            else:
                vec_decoder = vec_decoder + [0] * (self.max_output_len - len(vec_decoder))

        return vec_encoder, sentence_ids, vec_decoder

    def encode_input(self, text: str) -> List[int]:
        return self._tokenizer_encoder.encode(text)

    def encode_output(self, text: str) -> List[int]:
        return self._tokenizer_decoder.encode(text)

    def decode_input(self, sequence: List[int]) -> str:
        return self._tokenizer_encoder.decode(sequence)

    def decode_output(self, sequence: List[int]) -> str:
        return self._tokenizer_decoder.decode(sequence)
