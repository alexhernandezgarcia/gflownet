from typing import List, Union

import torch
from torchtyping import TensorType
from tqdm import tqdm

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat, tint, tlong
from gflownet.utils.scrabble.utils import read_alphabet, read_vocabulary


class ScrabbleScorer(Proxy):
    """
    Oracle to compute the Scrabble scores from words, that is the sum of the score of
    each letter in a sequence of letters (word).
    """

    def __init__(self, vocabulary_check: bool = False, **kwargs):
        self.vocabulary_check = vocabulary_check
        self.alphabet_dict = read_alphabet()
        self.vocabulary_orig = read_vocabulary()
        super().__init__(**kwargs)

    def setup(self, env=None):
        # Add pad_token to alphabet dict
        if env and not hasattr(self, "pad_token"):
            # Make a copy before modifying because the dictionary is global
            self.alphabet_dict = self.alphabet_dict.copy()
            self.pad_token = env.pad_token
            self.alphabet_dict[self.pad_token] = 0
        # Build scores tensor
        if env and not hasattr(self, "scores"):
            scores = [
                self.alphabet_dict[env.idx2token[idx]]
                for idx in range(len(env.idx2token))
            ]
            self.scores = tlong(scores, device=self.device)
        # Build index-based version of the vocabulary as a tensor
        self.vocabulary = torch.zeros(
            (len(self.vocabulary_orig), env.max_length),
            dtype=torch.int16,
            device=self.device,
        )
        for idx, word in enumerate(self.vocabulary_orig):
            word = "".join([letter + " " for letter in word.upper()])[:-1]
            self.vocabulary[idx] = tint(
                env.readable2state(word), device=self.device, int_type=torch.int16
            )

    def __call__(
        self, states: Union[List[str], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch"]:
        """
        Computes and returns the Scrabble score of sequence in a batch.

        In principle and in general, the input states is a tensor, where each state
        (row) is represented by the index of each token.

        However, for debugging purposes, this proxy also works if the input states is a
        list of:
            - Strings
            - List of string tokens
        See: tests/gflownet/proxy/test_scrabble_proxy.py

        Args
        ----
        states : tensor or list
            If a tensor: A batch of states, where each row is a state and each state
            represents a sequence by the indices of the token, including the padding.
            If a list: A batch of state, where each entry is either a string containing
            the word or a list of letters.

        Returns
        -------
        A vector with the score of each sequence in the batch.
        """
        if torch.is_tensor(states):
            output = torch.zeros(states.shape[0], device=self.device, dtype=self.float)
            if self.vocabulary_check:
                is_in_vocabulary = self._is_in_vocabulary(states)
            else:
                is_in_vocabulary = torch.ones_like(output, dtype=torch.bool)
            output[is_in_vocabulary] = tfloat(
                self.scores[states[is_in_vocabulary]].sum(dim=1),
                float_type=self.float,
                device=self.device,
            )
            return output
        elif isinstance(states, list):
            scores = []
            for sample in states:
                if (
                    self.vocabulary_check
                    and self._unpad_and_string(sample) not in self.vocabulary_orig
                ):
                    scores.append(0.0)
                else:
                    scores.append(self._sum_scores(sample))
            return tfloat(scores, device=self.device, float_type=self.float)
        else:
            raise NotImplementedError(
                "The Scrabble proxy currently only supports input states as a tensor "
                "of indices or as list of strings containing a token each"
            )

    def _sum_scores(self, sample: list) -> int:
        return sum(map(lambda x: self.alphabet_dict[x], sample))

    def _is_in_vocabulary(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch"]:
        """
        Returns the indices of the states that match any of the words in the
        vocabulary.

        See: https://stackoverflow.com/a/77419829/6194082
        """
        return (self.vocabulary == states.unsqueeze(1)).all(-1).any(-1)

    def _unpad_and_string(self, sample: list) -> str:
        if self.pad_token in sample:
            sample = sample[: sample.index(self.pad_token)]
        return "".join(sample).lower()
