"""
parent class to represent sequence-like environments, such as AMP and DNA. Sequences
are constructed by adding tokens from a dictionary. An alternative to this kind of
sequence environment (not-implemented as of July 2023) would be a "mutation-based"
modification of the sequences, or a combination of mutations and additions.
"""
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import set_device


class Sequence(GFlowNetEnv):
    """
    Parent class of sequence environments, in its simplest form: sequences are
    constructed starting from an empty sequence and adding one token at a time.

    By default, for illustration purposes, this parent class is functional and
    represents binary sequences of 0s and 1s, that can be padded with the special token
    [PAD] and are terminated by the special token [EOS].

    Attributes
    ----------
    tokens : iterable
        An iterable containing the vocabulary of tokens that make the sequences.

    max_length : int
        Maximum length of the sequences.

    eos_token : int, str
       EOS token. Default: -1.

    pad_token : int, str
       PAD token. Default: -2.
    """

    def __init__(
        self,
        tokens: Iterable = [0, 1],
        max_length: int = 10,
        eos_token: Union[int, str] = -1,
        pad_token: Union[int, str] = -2,
        **kwargs,
    ):
        assert max_length > 0
        self.device = set_device(kwargs["device"])
        # Main attributes
        self.tokens = set(tokens)
        self.max_length = max_length
        self.eos_idx = -1
        self.pad_idx = -2
        # Dictionaries
        self.idx2token = {idx: token for idx, token in enumerate(self.tokens)}
        self.idx2token[self.eos_idx] = eos_token
        self.idx2token[self.pad_idx] = pad_token
        self.token2idx = {token: idx for idx, token in self.idx2token.items()}
        # Source state: vector of length max_length filled with pad token
        self.source = torch.full(
            (self.max_length,), self.pad_idx, dtype=torch.long, device=self.device
        )
        # End-of-sequence action
        self.eos = (self.eos_idx,)
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        Constructs list with all possible actions, including eos.

        An action is represented by a single-element tuple indicating the index of the
        token to be added to the current sequence (state).

        The action space of this parent class is:
            action_space: [(0,), (1,), (-1,)]
        """
        return [(self.token2idx[token],) for token in self.tokens] + [(self.eos_idx,)]

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [True for _ in range(self.action_space_dim)]
        # If sequence is not at maximum length, all actions are valid
        if state[-1] == self.pad_idx:
            return [False for _ in range(self.action_space_dim)]
        # Otherwise, only EOS is valid
        mask = [True for _ in range(self.action_space_dim)]
        mask[self.action_space.index(self.eos)] = False
        return mask
