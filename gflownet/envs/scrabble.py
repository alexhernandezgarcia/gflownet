"""
Scrabble environment: starting from an emtpy sequence, letters are added one by one up
to a maximum length.
"""

from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tlong

LETTERS = tuple(
    [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]
)


class Scrabble(GFlowNetEnv):
    """
    Scrabble environment: sequences are constructed starting from an empty sequence and
    adding one letter at a time.

    States are represented by a list of indices corresponding to each letter, starting
    from 1, and are padded with index 0.

    Actions are represented by a single-element tuple with the index of the letter to
    be added. The EOS action is by (-1, ).

    Attributes
    ----------
    letters : tuple
        An tuple containing the letters to form words. By default, LETTERS is used.

    max_length : int
        Maximum length of the sequences. Default is 7, like in the standard game.

    pad_token : str
       PAD token. Default: "0".
    """

    def __init__(
        self,
        letters: Iterable = None,
        max_length: int = 7,
        pad_token: str = "0",
        **kwargs,
    ):
        # Main attributes
        if letters is None:
            self.letters = LETTERS
        else:
            self.letters = letters
        self.pad_token = pad_token
        self.n_letters = len(self.letters)
        self.max_length = max_length
        self.eos_idx = -1
        self.pad_idx = 0
        # Dictionaries
        self.idx2token = {idx + 1: token for idx, token in enumerate(self.letters)}
        self.idx2token[self.pad_idx] = pad_token
        self.token2idx = {token: idx for idx, token in self.idx2token.items()}
        # Source state: list of length max_length filled with pad token
        self.source = [self.pad_idx] * self.max_length
        # End-of-sequence action
        self.eos = (self.eos_idx,)
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs list with all possible actions, including eos.

        An action is represented by a single-element tuple indicating the index of the
        letter to be added to the current sequence (state).

        The action space of this parent class is:
            action_space: [(0,), (1,), (-1,)]
        """
        return [(self.token2idx[token],) for token in self.letters] + [(self.eos_idx,)]

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List[int]] = None,
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
        state: Optional[List[int]] = None,
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
        pos_last_letter = self._get_seq_length(state) - 1
        parent = copy(state)
        parent[pos_last_letter] = self.pad_idx
        p_action = (state[pos_last_letter],)
        return [parent], [p_action]

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> [List[int], Tuple[int], bool]:
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
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: the batch is
        simply converted into a tensor of indices.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        A list containing all the states in the batch, represented themselves as lists.
        """
        return tlong(states, device=self.device)

    def states2policy(
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: states
        are one-hot encoded.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tlong(states, device=self.device)
        return (
            F.one_hot(states, self.n_letters + 1)
            .reshape(states.shape[0], -1)
            .to(self.float)
        )

    def state2readable(self, state: List[int] = None) -> str:
        """
        Converts a state into a human-readable string.

        The output string contains the letter corresponding to each index in the state,
        separated by spaces.

        Args
        ----
        states : tensor
            A state in environment format. If None, self.state is used.

        Returns
        -------
        A string of space-separated letters.
        """
        state = self._get_state(state)
        state = self._unpad(state)
        return "".join([str(self.idx2token[idx]) + " " for idx in state])[:-1]

    def readable2state(self, readable: str) -> List[int]:
        """
        Converts a state in readable format into the "environment format" (tensor)

        Args
        ----
        readable : str
            A state in readable format - space-separated letters.

        Returns
        -------
        A tensor containing the indices of the letters.
        """
        if readable == "":
            return self.source
        return self._pad([self.token2idx[token] for token in readable.split(" ")])

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[List[int]]:
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
        n_letters = len(self.letters)
        n_per_length = tlong(
            [n_letters**length for length in range(1, self.max_length + 1)],
            device=self.device,
        )
        lengths = Categorical(logits=n_per_length.repeat(n_states, 1)).sample() + 1
        samples = torch.randint(
            low=1, high=n_letters + 1, size=(n_states, self.max_length)
        )
        for idx, length in enumerate(lengths):
            samples[idx, length:] = 0
        return samples.tolist()

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

    def _get_seq_length(self, state: List[int] = None):
        """
        Returns the effective length of a state, that is ignoring the padding.

        Args
        ----
        state : list
            The input sequence. If None, self.state is used.

        Returns
        -------
        Length of the sequence, without counting the padding.
        """
        state = self._get_state(state)
        if state[-1] == self.pad_idx:
            return state.index(self.pad_idx)
        else:
            return len(state)
