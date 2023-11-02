from typing import List, Union

import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat
from gflownet.utils.scrabble.utils import read_alphabet, read_vocabulary


class ScrabbleScorer(Proxy):
    def __init__(self, **kwargs):
        self.alphabet_dict = read_alphabet()
        self.vocabulary = read_vocabulary()
        super().__init__(**kwargs)

    def setup(self, env=None):
        if env:
            # Make a copy before modifying because the dictionary is global
            self.alphabet_dict = self.alphabet_dict.copy()
            self.pad_token = env.pad_token
            self.alphabet_dict[self.pad_token] = 0

    def __call__(
        self, states: Union[List[str], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch"]:
        if not isinstance(states, list):
            raise NotImplementedError(
                "The Scrabble proxy currently only supports input states as a list of "
                "strings containing a token each"
            )
        scores = []
        for sample in states:
            if self._unpad_and_string(sample) not in self.vocabulary:
                scores.append(0.0)
            else:
                scores.append(-1.0 * self._sum_scores(sample))
        return tfloat(scores, device=self.device, float_type=self.float)

    def _sum_scores(self, sample: list) -> int:
        return sum(map(lambda x: self.alphabet_dict[x], sample))

    def _unpad_and_string(self, sample: list) -> str:
        return "".join(sample[: sample.index(self.pad_token)]).lower()
