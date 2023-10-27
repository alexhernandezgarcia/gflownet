"""
Parent class to represent sequence-like environments, such as AMP and DNA. Sequences
are constructed by adding tokens from a dictionary. An alternative to this kind of
sequence environment (not-implemented as of July 2023) would be a "mutation-based"
modification of the sequences, or a combination of mutations and additions.
"""
import itertools
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from polyleven import levenshtein
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv


class Sequence(GFlowNetEnv):
    """
    Parent of sequence environments. By default, for illustration purposes, this parent
    class is functional and represents binary sequences of 0s and 1s, that can be
    padded with the special token [PAD] and are terminated by the special token [EOS].

    Attributes
    ----------
    tokens : iterable
        An iterable containing the vocabulary of tokens that make the sequences.

    max_length : int
        Maximum length of the sequences.

    min_length : int
        Minimum length of the sequences

    max_word_length : int
        Maximum number of tokens allowed per action.

    min_word_length : int
        Minimum number of tokens allowed per action.

    eos_token : int, str
       EOS token. Default: -1.

    pad_token : int, str
       PAD token. Default: -2.
    """

    def __init__(
        self,
        tokens: Iterable = [0, 1],
        max_length: int = 10,
        min_length: int = 1,
        max_word_length: int = 1,
        min_word_length: int = 1,
        eos_token: Union[int, str] = -1,
        pad_token: Union[int, str] = -2,
        **kwargs,
    ):
        assert min_length > 0
        assert max_length > 0
        assert max_length >= min_length
        assert min_word_length > 0
        assert max_word_length > 0
        assert max_word_length >= min_word_length
        # Main attributes
        self.tokens = set(tokens)
        self.min_length = min_length
        self.max_length = max_length
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        self.eos_idx = -1
        self.pad_idx = -2
        # Dictionaries
        self.idx2token = {idx: token for idx, token in enumerate(self.tokens)}
        self.idx2token[self.eos_idx] = eos_token
        self.idx2token[self.pad_idx] = pad_token
        self.token2idx = {token: idx for idx, token in self.idx2token.items()}
        # Source state: vector of length max_length filled with pad token
        self.source = torch.full(
            self.max_length, self.pad_idx, dtype=torch.long, device=self.device
        )
        # End-of-sequence action
        self.eos = (self.eos_idx,) + (self.pad_idx,) * (self.max_word_length - 1)
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        Constructs list with all possible actions, including eos.

        An action is represented by a vector of length max_word_length where each
        element indicates the idex of the token to add to the sequence. Actions with a
        number of tokens smaller than max_word_length are padded with pad_idx.

        Examples:
            If min_word_length = 1 and max_word_length = 1:
                actions: [(0,), (1,), (-1,)]
            If min_word_length = 2 and max_word_length = 2:
                actions: [(0, 0,), (0, 1), (1, 0), (1, 1), (-1, -2)]
            If min_word_length = 1 and max_word_length = 2:
                actions: [(0, -2), (1, -2), (0, 0,), (0, 1), (1, 0), (1, 1), (-1, -2)]
        """
        valid_wordlens = np.arange(self.min_word_length, self.max_word_length + 1)
        alphabet = [a for a in range(self.n_alphabet)]
        actions = []
        for r in valid_wordlens:
            actions_r = [el for el in itertools.product(alphabet, repeat=r)]
            actions += actions_r
        # Add "eos" action
        # eos != n_alphabet in the init because it would break if max_word_length >1
        actions = actions + [(len(actions),)]
        self.eos = len(actions) - 1
        return actions

    def copy(self):
        return self.__class__(**self.__dict__)
        # return deepcopy(self)

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        Returns a vector of length the action space (where action space includes eos): True if action is invalid
        given the current state, False otherwise.
        """
        if state is None:
            state = self.state.clone().detach()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(len(self.action_space))]
        mask = [False for _ in range(len(self.action_space))]
        seq_length = (
            torch.where(state == self.padding_idx)[0][0]
            if state[-1] == self.padding_idx
            else len(state)
        )
        if seq_length < self.min_length:
            mask[self.eos] = True
        for idx, a in enumerate(self.action_space[:-1]):
            if seq_length + len(list(a)) > self.max_length:
                mask[idx] = True
        return mask

    def true_density(self, max_states=1e6):
        """
        Computes the reward density (reward / sum(rewards)) of the whole space, if the
        dimensionality is smaller than specified in the arguments.

        Returns
        -------
        Tuple:
          - normalized reward for each state
          - states
          - (un-normalized) reward)
        """
        if self._true_density is not None:
            return self._true_density
        if self.n_alphabet**self.max_length > max_states:
            return (None, None, None)
        state_all = np.int32(
            list(itertools.product(*[list(range(self.n_alphabet))] * self.max_length))
        )
        traj_rewards, state_end = zip(
            *[
                (self.proxy(state), state)
                for state in state_all
                if len(self.get_parents(state, False)[0]) > 0 or sum(state) == 0
            ]
        )
        traj_rewards = np.array(traj_rewards)
        self._true_density = (
            traj_rewards / traj_rewards.sum(),
            list(map(tuple, state_end)),
            traj_rewards,
        )
        return self._true_density

    # def state2oracle(self, state: List = None):
    #     return "".join(self.state2readable(state))

    def get_max_traj_length(
        self,
    ):
        return self.max_length / self.min_word_length + 1

    def statebatch2oracle(self, states: List[TensorType["max_length"]]) -> List[str]:
        state_oracle = []
        for state in states:
            if state[-1] == self.padding_idx:
                state = state[: torch.where(state == self.padding_idx)[0][0]]
            if self.tokenizer is not None and state[0] == self.tokenizer.bos_idx:
                state = state[1:-1]
            state_numpy = state.detach().cpu().numpy()
            state_oracle.append(self.state2oracle(state_numpy))
        return state_oracle

    def statetorch2oracle(
        self, states: TensorType["batch_dim", "max_length"]
    ) -> List[str]:
        return self.statebatch2oracle(states)

    # TODO: Deprecate as never used.
    def state2policy(self, state=None):
        """
        Transforms the sequence (state) given as argument (or self.state if None) into a
        one-hot encoding. The output is a list of length nalphabet * max_length,
        where each n-th successive block of nalphabet elements is a one-hot encoding of
        the letter in the n-th position.

        Example:
          - Sequence: AATGC
          - state: [0, 1, 3, 2]
                    A, T, G, C
          - state2obs(state): [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                              |     A    |      T    |      G    |      C    |

        If max_length > len(state), the last (max_length - len(state)) blocks are all
        0s.
        """
        if state is None:
            state = self.state.clone().detach()
        state = (
            state[: torch.where(state == self.padding_idx)[0][0]]
            if state[-1] == self.padding_idx
            else state
        )
        state_policy = torch.zeros(1, self.max_length, self.n_alphabet)
        if len(state) == 0:
            return state_policy.reshape(1, -1)
        state_onehot = F.one_hot(state, num_classes=self.n_alphabet + 1)[:, :, 1:].to(
            self.float
        )
        state_policy[:, : state_onehot.shape[1], :] = state_onehot
        return state_policy.reshape(state.shape[0], -1)

    def statebatch2policy(
        self, states: List[TensorType["1", "max_length"]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Transforms a batch of states into the policy model format. The output is a numpy
        array of shape [n_states, n_alphabet * max_seq_len].

        See state2policy()
        """
        state_tensor = torch.vstack(states)
        state_policy = self.statetorch2policy(state_tensor)
        return state_policy

    def statetorch2policy(
        self, states: TensorType["batch", "max_length"]
    ) -> TensorType["batch", "policy_input_dim"]:
        if states.dtype != torch.long:
            states = states.to(torch.long)
        state_onehot = (
            F.one_hot(states, self.n_alphabet + 2)[:, :, :-2]
            .to(self.float)
            .to(self.device)
        )
        state_padding_mask = (states != self.padding_idx).to(self.float).to(self.device)
        state_onehot_pad = state_onehot * state_padding_mask.unsqueeze(-1)
        # Assertion works as long as [PAD] is last key in lookup table.
        assert torch.eq(state_onehot_pad, state_onehot).all()
        state_policy = torch.zeros(
            states.shape[0],
            self.max_length,
            self.n_alphabet,
            device=self.device,
            dtype=self.float,
        )
        state_policy[:, : state_onehot.shape[1], :] = state_onehot
        return state_policy.reshape(states.shape[0], -1)

    def policytorch2state(self, state_policy: List) -> List:
        """
        Transforms the one-hot encoding version of a sequence (state) given as argument
        into a a sequence of letter indices.

        Example:
          - Sequence: AATGC
          - state_policy: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                 |     A    |      A    |      T    |      G    |      C    |
          - policy2state(state_policy): [0, 0, 1, 3, 2]
                    A, A, T, G, C
        """
        mat_state_policy = torch.reshape(
            state_policy, (self.max_length, self.n_alphabet)
        )
        state = torch.where(mat_state_policy)[1].tolist()
        return state

    # TODO: Deprecate as never used.
    def policy2state(self, state_policy: List) -> List:
        """
        Transforms the one-hot encoding version of a sequence (state) given as argument
        into a a sequence of letter indices.

        Example:
          - Sequence: AATGC
          - state_policy: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                 |     A    |      A    |      T    |      G    |      C    |
          - policy2state(state_policy): [0, 0, 1, 3, 2]
                    A, A, T, G, C
        """
        mat_state_policy = np.reshape(state_policy, (self.max_length, self.n_alphabet))
        state = np.where(mat_state_policy)[1].tolist()
        return state

    def state2oracle(self, state: List = None):
        return "".join(self.state2readable(state))

    def statebatch2oracle(self, states: List[TensorType["max_length"]]) -> List[str]:
        state_oracle = []
        for state in states:
            if state[-1] == self.padding_idx:
                state = state[: torch.where(state == self.padding_idx)[0][0]]
            if self.tokenizer is not None and state[0] == self.tokenizer.bos_idx:
                state = state[1:-1]
            state_numpy = state.detach().cpu().numpy()
            state_oracle.append(self.state2oracle(state_numpy))
        return state_oracle

    def statetorch2oracle(
        self, states: TensorType["batch_dim", "max_length"]
    ) -> List[str]:
        return self.statebatch2oracle(states)

    def state2readable(self, state: List) -> str:
        """
        Transforms a sequence given as a list of indices into a sequence of letters
        according to an alphabet.
        Used only in Buffer
        """
        if isinstance(state, torch.Tensor) == False:
            state = torch.tensor(state).long()
        if state[-1] == self.padding_idx:
            state = state[: torch.where(state == self.padding_idx)[0][0]]
        state = state.tolist()
        return "".join([self.inverse_lookup[el] for el in state])

    def statetorch2readable(self, state: TensorType["1", "max_length"]) -> str:
        if state[-1] == self.padding_idx:
            state = state[: torch.where(state == self.padding_idx)[0][0]]
        # TODO: neater way without having lookup as input arg
        if (
            self.lookup is not None
            and "[CLS]" in self.lookup.keys()
            and state[0] == self.lookup["[CLS]"]
        ):
            state = state[1:-1]
        state = state.tolist()
        readable = [self.inverse_lookup[el] for el in state]
        return "".join(readable)

    def readable2state(self, readable) -> TensorType["batch_dim", "max_length"]:
        """
        Transforms a list or string of letters into a list of indices according to an alphabet.
        """
        if isinstance(readable, str):
            encoded_readable = [self.lookup[el] for el in readable]
            state = torch.ones(self.max_length, dtype=torch.int64) * self.padding_idx
            state[: len(encoded_readable)] = torch.tensor(encoded_readable)
        else:
            encoded_readable = [[self.lookup[el] for el in seq] for seq in readable]
            state = (
                torch.ones((len(readable), self.max_length), dtype=torch.int64)
                * self.padding_idx
            )
            for i, seq in enumerate(encoded_readable):
                state[i, : len(seq)] = torch.tensor(seq)
        return state

    def get_parents(self, state=None, done=None, action=None):
        """
        Determines all parents and actions that lead to sequence state

        Args
        ----
        state : list
            Representation of a sequence (state), as a list of length max_length
            where each element is the index of a letter in the alphabet, from 0 to
            (nalphabet - 1).

        action : int
            Last action performed, only to determine if it was eos.

        Returns
        -------
        parents : list
            List of parents as state2obs(state)

        actions : list
            List of actions that lead to state for each parent in parents
        """
        # TODO: Adapt to tuple form actions
        if state is None:
            state = self.state.clone().detach()
        if done is None:
            done = self.done
        if done:
            return [state], [(self.eos,)]
        elif torch.eq(state, self.source).all():
            return [], []
        else:
            parents = []
            actions = []
            if state[-1] == self.padding_idx:
                state_last_element = int(torch.where(state == self.padding_idx)[0][0])
            else:
                state_last_element = len(state)
            max_parent_action_length = self.max_word_length + 1 - self.min_word_length
            for parent_action_length in range(1, max_parent_action_length + 1):
                parent_action = tuple(
                    state[
                        state_last_element - parent_action_length : state_last_element
                    ].numpy()
                )
                if parent_action in self.action_space:
                    parent = state.clone().detach()
                    parent[
                        state_last_element - parent_action_length : state_last_element
                    ] = self.padding_idx
                    parents.append(parent)
                    actions.append(parent_action)
        return parents, actions

    def step(self, action: Tuple[int]) -> Tuple[List[int], Tuple[int, int], bool]:
        """
        Executes step given an action index

        If action_idx is smaller than eos (no stop), add action to next
        position.

        Args
        ----
        action_idx : int
            Index of action in the action space. a == eos indicates "stop action"

        Returns
        -------
        self.state : list
            The sequence after executing the action

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        assert action in self.action_space
        # If only possible action is eos, then force eos
        if self.state[-1] != self.padding_idx:
            self.done = True
            self.n_actions += 1
            return self.state, (self.eos,), True
        # If action is not eos, then perform action
        state_last_element = int(torch.where(self.state == self.padding_idx)[0][0])
        if action[0] != self.eos:
            state_next = self.state.clone().detach()
            if state_last_element + len(action) > self.max_length:
                valid = False
            else:
                state_next[
                    state_last_element : state_last_element + len(action)
                ] = torch.LongTensor(action)
                self.state = state_next
                valid = True
                self.n_actions += 1
            return self.state, action, valid
        else:
            if state_last_element < self.min_length:
                valid = False
            else:
                self.done = True
                valid = True
                self.n_actions += 1
            return self.state, (self.eos,), valid
