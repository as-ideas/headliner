from typing import Tuple

from spacy.pipeline.pipes import Language


class BertPreprocessor:

    def __init__(self,
                 nlp: Language):
        """
        Initializes the preprocessor.

        Args:
            nlp: Spacy natural language processing pipeline.
        """
        self.nlp = nlp
        pipe = self.nlp.create_pipe('sentencizer')
        self.nlp.add_pipe(pipe)
        self.start_token = '[CLS]'
        self.end_token = '[SEP]'

    def __call__(self, data: Tuple[str, str]) -> Tuple[str, str]:
        """ Splits input text into sentences and adds start and end token to each sentence. """
        text_encoder, text_decoder = data[0], data[1]
        doc = self.nlp(text_encoder)
        sentences = [self._process_sentence(s) for s in doc.sents]
        text_encoder = ' '.join(sentences)
        text_decoder = self.start_token + ' ' + text_decoder + ' ' + self.end_token
        return text_encoder, text_decoder

    def _process_sentence(self, sentence):
        return self.start_token + ' ' + sentence.string.strip() + ' ' + self.end_token
