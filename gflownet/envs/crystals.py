"""
Classes to represent a hyper-grid environments
"""
import itertools
from typing import List

import numpy as np

from gflownet.envs.base import GFlowNetEnv


class Crystal(GFlowNetEnv):
    """
    Crystal environment for ionic conductivity

    Attributes
    ----------
    ndim : int
        Dimensionality of the grid

    length : int
        Size of the grid (cells per dimension)

    cell_min : float
        Lower bound of the cells range

    cell_max : float
        Upper bound of the cells range
    """

    def __init__(
        self,
        max_diff_elem=4,
        min_diff_elem=2,
        periodic_table=84,
        oxidation_states=None,
        alphabet={},
        min_atoms=2,
        max_atoms=20,
        min_atom_i=1,
        max_atom_i=10,
        env_id=None,
        reward_beta=1,
        reward_norm=1.0,
        reward_norm_std_mult=0,
        reward_func="power",
        denorm_proxy=False,
        energies_stats=None,
        proxy=None,
        oracle=None,
        **kwargs,
    ):
        super(Crystal, self).__init__(
            env_id,
            reward_beta,
            reward_norm,
            reward_norm_std_mult,
            reward_func,
            energies_stats,
            denorm_proxy,
            proxy,
            oracle,
            **kwargs,
        )
        # self.state = []
        self.state = [0 for _ in range(periodic_table)]
        self.max_diff_elem = max_diff_elem
        self.min_diff_elem = min_diff_elem
        self.periodic_table = periodic_table
        self.alphabet = alphabet
        self.oxidation_states = oxidation_states
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.min_atom_i = min_atom_i
        self.max_atom_i = max_atom_i
        self.obs_dim = self.periodic_table
        self.action_space = self.get_actions_space()
        self.eos = len(self.action_space)

    def get_actions_space(self):
        """
        Constructs list with all possible actions
        """
        assert self.max_diff_elem > self.min_diff_elem
        assert self.max_atom_i > self.min_atom_i
        valid_word_len = np.arange(self.min_atom_i, self.max_atom_i + 1)
        valid_elem_types = np.arange(self.min_diff_elem, self.max_diff_elem + 1)
        # alphabet = [combo
        #            for i in valid_elem_types
        #            for combo in itertools.combinations(self.periodic_table, i)
        #        ]
        # alphabet = np.arange(self.periodic_table)
        # actions = []
        actions = [[(elem, r) for r in valid_word_len] for elem in alphabet]
        # for r in valid_word_len:
        #     # for combo in alphabet:
        #     actions_r = [el for el in itertools.product(alphabet, repeat=r)]
        #     actions += actions_r
        return actions

    def get_max_traj_len(self):
        return self.max_atoms / self.min_atom_i

    def get_mask_invalid_actions(self, state=None, done=None):
        """
        Returns a vector of length the action space + 1: True if forward action is
        invalid given the current state, False otherwise.
        """
        state_elem = [e for i, e in enumerate(state) if i > 0]
        state_atoms = sum(state)
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(len(self.action_space) + 1)]
        mask = [False for _ in range(len(self.action_space) + 1)]
        if state_atoms < self.min_atoms:
            mask[self.eos] = True
        for idx, a in enumerate(self.action_space):
            if state_atoms + action[1] > self.max_atoms:
                mask[idx] = True
            else:
                new_elem = action[0] in state_elem
                if new_elem and len(state_elem) >= self.max_diff_elems:
                    mask[idx] = True
        return mask

    def true_density(self):
        # Return pre-computed true density if already stored
        if self._true_density is not None:
            return self._true_density
        # Calculate true density
        all_states = np.int32(
            list(itertools.product(*[list(range(self.length))] * self.n_dim))
        )
        state_mask = np.array(
            [len(self.get_parents(s, False)[0]) > 0 or sum(s) == 0 for s in all_states]
        )
        all_oracle = self.state2oracle(all_states)
        rewards = self.oracle(all_oracle)[state_mask]
        self._true_density = (
            rewards / rewards.sum(),
            rewards,
            list(map(tuple, all_states[state_mask])),
        )
        return self._true_density

    def state2oracle(self, state_list):
        """
        Prepares a list of states in "GFlowNet format" for the oracles: a list of length
        n_dim with values in the range [cell_min, cell_max] for each state.

        Args
        ----
        state_list : list of lists
            List of states.
        """
        return [
            (
                self.state2obs(state).reshape((self.n_dim, self.length))
                * self.cells[None, :]
            ).sum(axis=1)
            for state in state_list
        ]

    def state2obs(self, state=None):
        """
        Transforms the state given as argument (or self.state if None) into a
        one-hot encoding. The output is a list of len length * n_dim,
        where each n-th successive block of length elements is a one-hot encoding of
        the position in the n-th dimension.

        Example:
          - State, state: [0, 3, 1] (n_dim = 3)
          - state2obs(state): [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0] (length = 4)
                              |     0    |      3    |      1    |
        """
        if state is None:
            state = self.state.copy()

        # z = np.zeros(self.obs_dim, dtype=np.float32)

        if len(state) > 0:
            if hasattr(
                state[0], "device"
            ):  # if it has a device at all, it will be cuda (CPU numpy array has no dev
                state = [subseq.cpu().detach().numpy() for subseq in state]

            z = np.bincount(state)
        return z

    def obs2state(self, obs: List) -> List:
        """
        Transforms the one-hot encoding version of a sequence (state) given as argument
        into a a sequence of letter indices.
        Example:
          - Sequence: AATGC
          - obs: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                 |     A    |      A    |      T    |      G    |      C    |
          - state: [0, 0, 1, 3, 2]
                    A, A, T, G, C
        """
        # obs_mat = np.reshape(obs, (self.max_atoms, self.periodic_table))
        # state = np.where(obs_mat)[1].tolist()
        state = []

        for e, i in enumerate(obs):
            state += [e for _ in range(i)]
        return state

    def state2readable(self, state):
        """
        Transforms a sequence given as a list of indices into a sequence of letters
        according to an alphabet.
        """
        values, counts = np.unique(state, return_counts=True)
        values = [self.alphabet[v] for v in values]
        formula = {v: c for v, c in zip(values, counts)}
        # return ''.join(formula)
        return formula

    def readable2state(self, letters):
        """
        Transforms a sequence given as a list of indices into a sequence of letters
        according to an alphabet.
        """
        alphabet = {v: k for k, v in self.alphabet.items()}
        return [alphabet[el] for el in letters]

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = []
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

    def get_parents(self, state=None, done=None):
        """
        Determines all parents and actions that lead to sequence state
        Args
        ----
        state : list
            Representation of a sequence (state), as a list of length max_seq_length
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
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [self.state2obs(state)], [self.eos]
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                is_parent = state[-len(a) :] == list(a)
                if not isinstance(is_parent, bool):
                    is_parent = all(is_parent)
                if is_parent:
                    parents.append(self.state2obs(state[: -len(a)]))
                    actions.append(idx)
        return parents, actions

    def get_parents_debug(self, state=None, done=None):
        """
        Like get_parents(), but returns state format
        """
        obs, actions = self.get_parents(state, done)
        parents = [self.obs2state(el) for el in obs]
        return parents, actions

    def step(self, action_idx):
        """
        Executes step given an action index
        If action_idx is smaller than eos (no stop), add action to next
        position.
        See: step_daug()
        See: step_chain()
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
        # If only possible action is eos, then force eos
        if len(self.state) == self.max_atoms:
            self.done = True
            self.n_actions += 1
            return self.state, [self.eos], True
        # If action is not eos, then perform action
        if action_idx != self.eos:
            atomic_number, num = self.action_space[action_idx]
            state_next = self.state[:]
            state_next = state_next[atmoic_number] + [num]
            if len(state_next) > self.max_atoms:
                valid = False
            else:
                self.state = state_next
                valid = True
                self.n_actions += 1
            return self.state, action_idx, valid
        # If action is eos, then perform eos
        else:
            if len(self.state) < self.min_atoms:
                valid = False
            else:
                self.done = True
                valid = True
                self.n_actions += 1
            return self.state, self.eos, valid
