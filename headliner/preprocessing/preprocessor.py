import re
from typing import Tuple


class Preprocessor:

    def __init__(self,
                 start_token='<start>',
                 end_token='<end>',
                 punctuation_pattern='([!.?,])',
                 filter_pattern='(["#$%&()*+/:;<=>@[\\]^_`{|}~\t\n])',
                 lower_case=True,
                 hash_numbers=True):
        """
        Initializes the preprocessor.

        Args:
            start_token: Unique start token to be inserted at the beginning of the target text.
            end_token: Unique end token to be attached at the end of a target text.
            punctuation_pattern: Regex pattern for punktuation that is splitted from the tokens.
            filter_pattern: Regex pattern for characters to be removed from the text.
            lower_case: Whether to perform lower casing.
            hash_numbers: Whether to replace numbers by a #.
        """
        self.start_token = start_token
        self.end_token = end_token
        self.punctuation_pattern = punctuation_pattern
        self.filter_pattern = filter_pattern
        self.lower_case = lower_case
        self.hash_numbers = hash_numbers

    def __call__(self, data: Tuple[str, str]) -> Tuple[str, str]:
        """ Performs regex logic for string cleansing and attaches start and end tokens to target. """
        text_encoder, text_decoder = self.normalize_string(data[0]), self.normalize_string(data[1])
        text_decoder = self.start_token + ' ' + text_decoder + ' ' + self.end_token
        return text_encoder, text_decoder

    def normalize_string(self, s: str) -> str:
        if self.lower_case:
            s = s.lower()
        s = re.sub(self.filter_pattern, '', s)
        if self.hash_numbers:
            s = re.sub(r'\d+', '#', s)
        s = re.sub(self.punctuation_pattern, r' \1', s)
        s = re.sub(r'\s+', r' ', s)
        return s
