import unittest

from headliner.evaluation.bleu_scorer import BleuScorer


class TestBleuScorer(unittest.TestCase):

    def test_score(self):
        bleu_scorer = BleuScorer(tokens_to_ignore={'<end>', '<unk>'})

        text_preprocessed = ('', 'this is a test')
        pred = 'this is a test'
        score = bleu_scorer({'preprocessed_text': text_preprocessed, 'predicted_text': pred})
        self.assertAlmostEqual(1, score, 5)

        text_preprocessed = ('', 'this is a test')
        pred = 'this is a test <end>'
        score = bleu_scorer({'preprocessed_text': text_preprocessed, 'predicted_text': pred})
        self.assertAlmostEqual(1, score, 5)

        text_preprocessed = ('', 'it is a guide to action which ensures that the military '
                                 'always obeys the commands of the party')
        pred = 'it is a guide to action that ensures that the military will ' \
               'forever <unk> <unk> <unk> heed Party commands <end>'
        score = bleu_scorer({'preprocessed_text': text_preprocessed, 'predicted_text': pred})
        self.assertAlmostEqual(0.4138, score, 3)
