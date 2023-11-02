"""
parent class to represent sequence-like environments, such as AMP and DNA. Sequences
are constructed by adding tokens from a dictionary. An alternative to this kind of
sequence environment (not-implemented as of July 2023) would be a "mutation-based"
modification of the sequences, or a combination of mutations and additions.
"""
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
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
       PAD token. Default: -1.
    """

    def __init__(
        self,
        tokens: Iterable = [0, 1],
        max_length: int = 5,
        pad_token: Union[int, float, str] = -1,
        **kwargs,
    ):
        assert max_length > 0
        assert len(set(tokens)) == len(tokens)
        # Make sure that padding token is not one of the regular tokens
        if pad_token in tokens:
            raise ValueError(
                f"The padding token ({pad_token}) cannot be one of the regular tokens."
            )
        # Make sure that all tokens are the same type
        if (
            len(set([type(token) for token in set(tokens).union(set([pad_token]))]))
            != 1
        ):
            raise ValueError(
                f"All tokens must be the same type, but more than one type was found."
            )
        # Set device because it is needed in the init
        self.device = set_device(kwargs["device"])
        # Main attributes
        self.tokens = tuple(tokens)
        self.pad_token = pad_token
        self.n_tokens = len(self.tokens)
        self.max_length = max_length
        self.eos_idx = -1
        self.pad_idx = 0
        self.dtype = type(pad_token)
        # Dictionaries
        self.idx2token = {idx + 1: token for idx, token in enumerate(self.tokens)}
        self.idx2token[self.pad_idx] = pad_token
        self.token2idx = {token: idx for idx, token in self.idx2token.items()}
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

    def states2proxy(
        self,
        states: Union[
            List[TensorType["max_length"]], TensorType["batch", "max_length"]
        ],
    ) -> List[List]:
        """
        Prepares a batch of states in "environment format" for a proxy: states
        are represented by the tokens instead of the indices, with padding up to the
        max_length.

        Important: by default, the output of states2proxy() is a list of lists, instead
        of a tensor as in most environments. This is to allow for string tokens.

        Example, with max_length = 5:
          - Sequence (tokens): 0100
          - state: [1, 2, 1, 1, 0]
          - proxy format: [0, 1, 0, 0, -1]

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A list containing all the states in the batch, represented themselves as lists.
        """
        states = tlong(states, device=self.device).tolist()
        states_proxy = []
        for state in states:
            states_proxy.append([self.idx2token[idx] for idx in state])
        return states_proxy

    def states2policy(
        self,
        states: Union[
            List[TensorType["max_length"]], TensorType["batch", "max_length"]
        ],
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: states
        are one-hot encoded.

        Example, with max_length = 5:
          - Sequence (tokens): 0100
          - state: [1, 2, 1, 1, 0]
          - policy format: [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
                           |   0   |    1   |    0   |    0   |   PAD  |

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tlong(states, device=self.device)
        return (
            F.one_hot(states, self.n_tokens + 1)
            .reshape(states.shape[0], -1)
            .to(self.float)
        )

    def state2readable(self, state: TensorType["max_length"] = None) -> str:
        """
        Converts a state into a human-readable string.

        The output string contains the token corresponding to each index in the state,
        separated by spaces.

        Args
        ----
        states : tensor
            A state in environment format. If None, self.state is used.

        Returns
        -------
        A string of space-separated tokens.
        """
        state = self._get_state(state)
        state = self._unpad(state.tolist())
        return "".join([str(self.idx2token[idx]) + " " for idx in state])[:-1]

    def readable2state(self, readable: str) -> TensorType["max_length"]:
        """
        Converts a state in readable format into the "environment format" (tensor)

        Args
        ----
        readable : str
            A state in readable format - space-separated tokens.

        Returns
        -------
        A tensor containing the indices of the tokens.
        """
        if readable == "":
            return self.source
        return tlong(
            self._pad(
                [self.token2idx[self.dtype(token)] for token in readable.split(" ")]
            ),
            device=self.device,
        )

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[TensorType["batch", "max_length"]]:
        """
        Constructs a batch of n states uniformly sampled in the sample space of the
        environment.

        Args
        ----
        n_states : int
            The number of states to sample.

        seed : int
            Random seed.
        """
        n_tokens = len(self.tokens)
        n_per_length = tlong(
            [n_tokens**length for length in range(1, self.max_length + 1)],
            device=self.device,
        )
        lengths = Categorical(logits=n_per_length.repeat(n_states, 1)).sample() + 1
        samples = torch.randint(
            low=1, high=n_tokens + 1, size=(n_states, self.max_length)
        )
        for idx, length in enumerate(lengths):
            samples[idx, length:] = 0
        return samples

    def _pad(self, seq_list: list):
        """
        Pads a sequence represented as a list of indices.

        Args
        ----
        seq_list : list
            The input sequence. A list containing a list of indices.

        Returns
        -------
        The input list padded by the end with self.pad_idx.
        """
        return seq_list + [self.pad_idx] * (self.max_length - len(seq_list))

    def _unpad(self, seq_list: list):
        """
        Removes the padding from the end off a sequence represented as a list of
        indices.

        Args
        ----
        seq_list : list
            The input sequence. A list containing a list of indices, including possibly
            padding indices.

        Returns
        -------
        The input list padded by the end with self.pad_idx.
        """
        if self.pad_idx not in seq_list:
            return seq_list
        return seq_list[: seq_list.index(self.pad_idx)]

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
