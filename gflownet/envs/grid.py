"""
Classes to represent a hyper-grid environments
"""
import itertools
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv


class Grid(GFlowNetEnv):
    """
    Hyper-grid environment: A grid with n_dim dimensions and length cells per
    dimensions.

    The state space is the entire grid and each state is represented by the vector of
    coordinates of each dimensions. For example, in 3D, the origin will be at [0, 0, 0]
    and after incrementing dimension 0 by 2, dimension 1 by 3 and dimension 3 by 1, the
    state would be [2, 3, 1].

    The action space is the increment to be applied to each dimension. For instance,
    (0, 0, 1) will increment dimension 2 by 1 and the action that goes from [1, 1, 1]
    to [2, 3, 1] is (1, 2, 0).

    Attributes
    ----------
    n_dim : int
        Dimensionality of the grid

    length : int
        Size of the grid (cells per dimension)

    max_increment : int
        Maximum increment of each dimension by the actions.

    max_dim_per_action : int
        Maximum number of dimensions to increment per action. If -1, then
        max_dim_per_action is set to n_dim.

    cell_min : float
        Lower bound of the cells range

    cell_max : float
        Upper bound of the cells range
    """

    def __init__(
        self,
        n_dim: int = 2,
        length: int = 3,
        max_increment: int = 1,
        max_dim_per_action: int = 1,
        cell_min: float = -1,
        cell_max: float = 1,
        **kwargs,
    ):
        assert n_dim > 0
        assert length > 1
        assert max_increment > 0
        assert max_dim_per_action == -1 or max_dim_per_action > 0
        self.n_dim = n_dim
        self.length = length
        self.max_increment = max_increment
        if max_dim_per_action == -1:
            max_dim_per_action = self.n_dim
        self.max_dim_per_action = max_dim_per_action
        self.cells = np.linspace(cell_min, cell_max, length)
        # Source state: position 0 at all dimensions
        self.source = [0 for _ in range(self.n_dim)]
        # End-of-sequence action
        self.eos = tuple([0 for _ in range(self.n_dim)])
        # Base class init
        super().__init__(**kwargs)
        # Proxy format
        # TODO: assess if really needed
        if self.proxy_state_format == "ohe":
            self.statebatch2proxy = self.statebatch2policy
        elif self.proxy_state_format == "oracle":
            self.statebatch2proxy = self.statebatch2oracle
            self.statetorch2proxy = self.statetorch2oracle

    def get_action_space(self):
        """
        Constructs list with all possible actions, including eos. An action is
        represented by a vector of length n_dim where each index d indicates the
        increment to apply to dimension d of the hyper-grid.
        """
        increments = [el for el in range(self.max_increment + 1)]
        actions = []
        for action in itertools.product(increments, repeat=self.n_dim):
            if (
                sum(action) != 0
                and len([el for el in action if el > 0]) <= self.max_dim_per_action
            ):
                actions.append(tuple(action))
        actions.append(self.eos)
        return actions

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
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(self.policy_output_dim)]
        mask = [False for _ in range(self.policy_output_dim)]
        for idx, action in enumerate(self.action_space[:-1]):
            child = state.copy()
            for dim, incr in enumerate(action):
                child[dim] += incr
            if any(el >= self.length for el in child):
                mask[idx] = True
        return mask

    def state2oracle(self, state: List = None) -> List:
        """
        Prepares a state in "GFlowNet format" for the oracles: a list of length
        n_dim with values in the range [cell_min, cell_max] for each state.

        See: state2policy()

        Args
        ----
        state : list
            State
        """
        if state is None:
            state = self.state.copy()
        return (
            (
                np.array(self.state2policy(state)).reshape((self.n_dim, self.length))
                * self.cells[None, :]
            )
            .sum(axis=1)
            .tolist()
        )

    def statebatch2oracle(
        self, states: List[List]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in "GFlowNet format" for the oracles: each state is
        a vector of length n_dim with values in the range [cell_min, cell_max].

        See: statetorch2oracle()

        Args
        ----
        state : list
            State
        """
        return self.statetorch2oracle(
            torch.tensor(states, device=self.device, dtype=self.float)
        )

    def statetorch2oracle(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in "GFlowNet format" for the oracles: each state is
        a vector of length n_dim with values in the range [cell_min, cell_max].

        See: statetorch2policy()
        """
        return (
            self.statetorch2policy(states).reshape(
                (len(states), self.n_dim, self.length)
            )
            * torch.tensor(self.cells[None, :]).to(states)
        ).sum(axis=2)

    def state2policy(self, state: List = None) -> List:
        """
        Transforms the state given as argument (or self.state if None) into a
        one-hot encoding. The output is a list of len length * n_dim,
        where each n-th successive block of length elements is a one-hot encoding of
        the position in the n-th dimension.

        Example:
          - State, state: [0, 3, 1] (n_dim = 3)
          - state2policy(state): [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0] (length = 4)
                              |     0    |      3    |      1    |
        """
        if state is None:
            state = self.state.copy()
        state_policy = np.zeros(self.length * self.n_dim, dtype=np.float32)
        state_policy[(np.arange(len(state)) * self.length + state)] = 1
        return state_policy.tolist()

    def statebatch2policy(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Transforms a batch of states into a one-hot encoding. The output is a numpy
        array of shape [n_states, length * n_dim].

        See state2policy().
        """
        cols = np.array(states) + np.arange(self.n_dim) * self.length
        rows = np.repeat(np.arange(len(states)), self.n_dim)
        state_policy = np.zeros(
            (len(states), self.length * self.n_dim), dtype=np.float32
        )
        state_policy[rows, cols.flatten()] = 1.0
        return state_policy

    def statetorch2policy(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "policy_output_dim"]:
        """
        Transforms a batch of states into a one-hot encoding. The output is a numpy
        array of shape [n_states, length * n_dim].

        See state2policy().
        """
        device = states.device
        cols = (states + torch.arange(self.n_dim).to(device) * self.length).to(int)
        rows = torch.repeat_interleave(
            torch.arange(states.shape[0]).to(device), self.n_dim
        )
        state_policy = torch.zeros(
            (states.shape[0], self.length * self.n_dim), dtype=states.dtype
        ).to(device)
        state_policy[rows, cols.flatten()] = 1.0
        return state_policy

    def policy2state(self, state_policy: List) -> List:
        """
        Transforms the one-hot encoding version of a state given as argument
        into a state (list of the position at each dimension).

        Example:
          - state_policy: [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0] (length = 4, n_dim = 3)
                          |     0    |      3    |      1    |
          - policy2state(state_policy): [0, 3, 1]
        """
        return np.where(np.reshape(state_policy, (self.n_dim, self.length)))[1].tolist()

    def readable2state(self, readable, alphabet={}):
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        return [int(el) for el in readable.strip("[]").split(" ")]

    def state2readable(self, state, alphabet={}):
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state.
        """
        return str(state).replace("(", "[").replace(")", "]").replace(",", "")

    def get_parents(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state, as a list of length length where each element is
            the position at each dimension.

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]
        else:
            parents = []
            actions = []
            for idx, action in enumerate(self.action_space[:-1]):
                parent = state.copy()
                for dim, incr in enumerate(action):
                    if parent[dim] - incr >= 0:
                        parent[dim] -= incr
                    else:
                        break
                else:
                    parents.append(parent)
                    actions.append(action)
        return parents, actions

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
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
        # If only possible action is eos, then force eos
        # All dimensions are at the maximum length
        if all([s == self.length - 1 for s in self.state]):
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If action is not eos, then perform action
        elif action != self.eos:
            state_next = self.state.copy()
            for dim, incr in enumerate(action):
                state_next[dim] += incr
            if any([s >= self.length for s in state_next]):
                valid = False
            else:
                self.state = state_next
                valid = True
                self.n_actions += 1
            return self.state, action, valid
        # If action is eos, then perform eos
        else:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True

    def get_max_traj_length(self):
        return self.n_dim * self.length

    def get_all_terminating_states(self) -> List[List]:
        all_x = np.int32(
            list(itertools.product(*[list(range(self.length))] * self.n_dim))
        )
        return all_x.tolist()

    def get_uniform_terminating_states(self, n_states: int, seed: int) -> List[List]:
        rng = np.random.default_rng(seed)
        states = rng.integers(low=0, high=self.length, size=(n_states, self.n_dim))
        return states.tolist()
