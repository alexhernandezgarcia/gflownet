"""
Classes to represent hyper-torus environments
"""

import itertools
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import tfloat, tlong


class Torus(GFlowNetEnv):
    """
    Hyper-torus environment in which the action space consists of:
        - Increasing the angle index of dimension d
        - Decreasing the angle index of dimension d
        - Keeping all dimensions as are
    and the trajectory is of fixed length length_traj.

    The states space is the concatenation of the angle index at each dimension and the
    number of actions.

    Attributes
    ----------
    ndim : int
        Dimensionality of the torus

    n_angles : int
        Number of angles into which each dimension is divided

    length_traj : int
       Fixed length of the trajectory.
    """

    def __init__(
        self,
        n_dim: int = 2,
        n_angles: int = 3,
        length_traj: int = 1,
        max_increment: int = 1,
        max_dim_per_action: int = 1,
        **kwargs,
    ):
        assert n_dim > 0
        assert n_angles > 1
        assert length_traj > 0
        assert max_increment > 0
        assert max_dim_per_action == -1 or max_dim_per_action > 0
        self.n_dim = n_dim
        self.n_angles = n_angles
        self.length_traj = length_traj
        self.max_increment = max_increment
        if max_dim_per_action == -1:
            max_dim_per_action = self.n_dim
        self.max_dim_per_action = max_dim_per_action
        # Source state: position 0 at all dimensions and number of actions 0
        self.source_angles = [0 for _ in range(self.n_dim)]
        self.source = self.source_angles + [0]
        # End-of-sequence action: (self.max_incremement + 1) in all dimensions
        self.eos = tuple([self.max_increment + 1 for _ in range(self.n_dim)])
        # Angle increments in radians
        self.angle_rad = 2 * np.pi / self.n_angles
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        Constructs list with all possible actions, including eos. An action is
        represented by a vector of length n_dim where each index d indicates the
        increment/decrement to apply to dimension d of the hyper-torus. A negative
        value indicates a decrement. The action "keep" (no increment/decrement of any
        dimensions) is valid and is indicated by all zeros.
        """
        increments = [el for el in range(-self.max_increment, self.max_increment + 1)]
        actions = []
        for action in itertools.product(increments, repeat=self.n_dim):
            if len([el for el in action if el != 0]) <= self.max_dim_per_action:
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
        All actions except EOS are valid if the maximum number of actions has not been
        reached, and vice versa.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(self.action_space_dim)]
        if state[-1] >= self.length_traj:
            mask = [True for _ in range(self.action_space_dim)]
            mask[-1] = False
        else:
            mask = [False for _ in range(self.action_space_dim)]
            mask[-1] = True
        return mask

    def states2proxy(
        self, states: Union[List[List], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in "environment format" for the proxy: each state is
        a vector of length n_dim where each value is an angle in radians. The n_actions
        item is removed.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        return (
            tfloat(states, device=self.device, float_type=self.float)[:, :-1]
            * self.angle_rad
        )

    # TODO: circular encoding as in htorus
    def states2policy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: the
        policy format is a one-hot encoding of the states.

        Each row is a vector of length n_angles * n_dim + 1, where each n-th successive
        block of length elements is a one-hot encoding of the position in the n-th
        dimension.

        Example, n_dim = 2, n_angles = 4:
          - state: [1, 3, 4]
                          | a  | n | (a = angles, n = n_actions)
          - policy format: [0, 1, 0, 0, 0, 0, 0, 1, 4]
                           |     1    |     3     | 4 |
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
        cols = states[:, :-1] + torch.arange(self.n_dim).to(self.device) * self.n_angles
        rows = torch.repeat_interleave(
            torch.arange(states.shape[0]).to(self.device), self.n_dim
        )
        states_policy = torch.zeros(
            (states.shape[0], self.n_angles * self.n_dim + 1)
        ).to(states)
        states_policy[rows, cols.flatten()] = 1.0
        states_policy[:, -1] = states[:, -1]
        return states_policy.to(self.float)

    def state2readable(self, state: Optional[List] = None) -> str:
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state.
        """
        state = self._get_state(state)
        angles = (
            str(state[: self.n_dim])
            .replace("(", "[")
            .replace(")", "]")
            .replace(",", "")
        )
        n_actions = str(state[-1])
        return angles + " | " + n_actions

    def readable2state(self, readable: str) -> List:
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        pair = readable.split(" | ")
        angles = [int(el) for el in pair[0].strip("[]").split(" ")]
        n_actions = [int(pair[1])]
        return angles + n_actions

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
            Representation of a state, as a list of length n_angles where each element
            is the position at each dimension.

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
        # If source state
        elif state[-1] == 0:
            return [], []
        else:
            parents = []
            actions = []
            for idx, action in enumerate(self.action_space[:-1]):
                state_p = state.copy()
                angles_p = state_p[: self.n_dim]
                n_actions_p = state_p[-1]
                # Get parent
                n_actions_p -= 1
                for dim, incr in enumerate(action):
                    angles_p[dim] -= incr
                    # If negative angle index, restart from the back
                    if angles_p[dim] < 0:
                        angles_p[dim] = self.n_angles + angles_p[dim]
                    # If angle index larger than n_angles, restart from 0
                    if angles_p[dim] >= self.n_angles:
                        angles_p[dim] = angles_p[dim] - self.n_angles
                if self._get_min_actions_to_source(angles_p) < state[-1]:
                    state_p = angles_p + [n_actions_p]
                    parents.append(state_p)
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
            Action to be executed. See: get_action_space()

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
        # If the number of actions is equal to trajectory length
        if self.n_actions == self.length_traj:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # Perform non-EOS action
        else:
            angles_next = self.state.copy()[: self.n_dim]
            for dim, incr in enumerate(action):
                angles_next[dim] += incr
                # If negative angle index, restart from the back
                if angles_next[dim] < 0:
                    angles_next[dim] = self.n_angles + angles_next[dim]
                # If angle index larger than n_angles, restart from 0
                if angles_next[dim] >= self.n_angles:
                    angles_next[dim] = angles_next[dim] - self.n_angles
            self.n_actions += 1
            self.state = angles_next + [self.n_actions]
            valid = True
            return self.state, action, valid

    def get_all_terminating_states(self):
        all_x = itertools.product(*[list(range(self.n_angles))] * self.n_dim)
        all_x_valid = []
        for x in all_x:
            if self._get_min_actions_to_source(x) <= self.length_traj:
                all_x_valid.append(x)
        all_x = np.int32(all_x_valid)
        n_actions = self.length_traj * np.ones([all_x.shape[0], 1], dtype=np.int32)
        all_x = np.concatenate([all_x, n_actions], axis=1)
        return all_x.tolist()

    def fit_kde(x, kernel="exponential", bandwidth=0.1):
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(last_states.numpy())

    def _get_min_actions_to_source(self, angles):
        def _get_min_actions_dim(u, v):
            return np.min([np.abs(u - v), np.abs(u - (v - self.n_angles))])

        return np.sum(
            [_get_min_actions_dim(u, v) for u, v in zip(self.source_angles, angles)]
        )
