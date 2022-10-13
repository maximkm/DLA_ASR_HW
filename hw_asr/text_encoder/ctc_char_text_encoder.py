from tokenizers import Tokenizer
from tokenizers.models import BPE
from typing import List, NamedTuple
from collections import defaultdict
import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class BeamHypothesis(NamedTuple):
    text: str
    last_ind: int


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        last_ind = 0
        res = ''
        for ind in inds:
            if ind != last_ind and ind != 0:
                res += self.ind2char[ind]
            last_ind = ind
        return res

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        cur_hypos = {BeamHypothesis('', 0): 1.}
        for i in range(probs_length):
            # extend and merge
            next_hypos = defaultdict(float)
            for hyp, prob in cur_hypos.items():
                last_char = self.ind2char[hyp.last_ind] if hyp.last_ind != 0 else ''
                for j in range(voc_size):
                    if j == hyp.last_ind:
                        next_hypos[hyp] += prob * probs[i][j]
                    else:
                        next_hypos[BeamHypothesis(hyp.text + last_char, j)] += prob * probs[i][j]

            # cut beams
            cur_hypos = dict(sorted(next_hypos.items(), key=lambda x: x[1], reverse=True)[:beam_size])

        hypos = [Hypothesis(hyp.text, prob) for hyp, prob in cur_hypos.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)


class BPETextEncoder(CTCCharTextEncoder):
    def __init__(self, alphabet: List[str] = None, file_tokenizer=''):
        super().__init__(alphabet)
        tokenizer = Tokenizer(BPE())
        self.tokenizer = tokenizer.from_file(file_tokenizer)
        self.tokenizer.add_tokens([' '])
        self.ind2char = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor(self.tokenizer.encode(text).ids).unsqueeze(0)
        except KeyError as e:
            raise Exception(f"Can't encode text '{text}'.")

    def get_vocab(self):
        return list(map(lambda x: x[0] if x[0] != self.EMPTY_TOK else "",
                        sorted(self.tokenizer.get_vocab().items(), key=lambda x: x[1])))

    @classmethod
    def from_file(cls, file):
        raise NotImplementedError()
