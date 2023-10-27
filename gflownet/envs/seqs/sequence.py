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
from gflownet.utils.common import copy, set_device, tlong


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
        self.token2idx = {token: idx for idx, token in self.idx2token.items()}
        self.idx2token[self.eos_idx] = eos_token
        self.idx2token[self.pad_idx] = pad_token
        # Source state: vector of length max_length filled with pad token
        self.source = tlong(
            torch.full((self.max_length,), self.pad_idx), device=self.device
        )
        # End-of-sequence action
        self.eos = (self.eos_idx,)
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self) -> List[Tuple]:
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
        state: Optional[TensorType["max_length"]] = None,
        done: Optional[bool] = None,
    ) -> List[bool]:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.

        Args
        ----
        state : tensor
            Input state. If None, self.state is used.

        done : bool
            Whether the trajectory is done. If None, self.done is used.

        Returns
        -------
        A list of boolean values.
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

    def get_parents(
        self,
        state: Optional[TensorType["max_length"]] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        The GFlowNet graph is a tree and there is only one parent per state.

        Args
        ----
        state : tensor
            Input state. If None, self.state is used.

        done : bool
            Whether the trajectory is done. If None, self.done is used.

        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format. This environment has a single parent per
            state.

        actions : list
            List of actions that lead to state for each parent in parents. This
            environment has a single parent per state.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [state], [self.eos]
        if self.equal(state, self.source):
            return [], []
        pos_last_token = self._get_seq_length(state) - 1
        parent = copy(state)
        parent[pos_last_token] = self.pad_idx
        p_action = (state[pos_last_token],)
        return [parent], [p_action]

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> [TensorType["max_length"], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple int values indicating the
            dimensions to increment by 1.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        # Generic pre-step checks
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, False
        valid = True
        self.n_actions += 1
        # If action is EOS, set done to True and return state as is
        if action == self.eos:
            self.done = True
            return self.state, action, valid
        # Update state
        self.state[self._get_seq_length()] = action[0]
        return self.state, action, valid

    def _get_seq_length(self, state: TensorType["max_length"] = None):
        """
        Returns the effective length of a state, that is ignoring the padding.

        Args
        ----
        state : tensor
            The input sequence. If None, self.state is used.

        Returns
        -------
        Single element int tensor.
        """
        state = self._get_state(state)
        if state[-1] == self.pad_idx:
            return torch.where(state == self.pad_idx)[0][0]
        else:
            return len(state)
