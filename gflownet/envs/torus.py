"""
Classes to represent hyper-torus environments
"""
from typing import List, Tuple
import itertools
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torchtyping import TensorType
from gflownet.envs.base import GFlowNetEnv


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
        n_dim=2,
        n_angles=3,
        length_traj=1,
        min_step_len=1,
        max_step_len=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_dim = n_dim
        self.eos = self.n_dim
        self.n_angles = n_angles
        self.length_traj = length_traj
        # Initialize angles and state attributes
        self.source_angles = [0.0 for _ in range(self.n_dim)]
        # States are the concatenation of the angle state and number of actions
        self.source = self.source_angles + [0]
        self.reset()
        self.source = self.angles.copy()
        self.min_step_len = min_step_len
        self.max_step_len = max_step_len
        self.action_space = self.get_actions_space()
        self.fixed_policy_output = self.get_fixed_policy_output()
        self.random_policy_output = self.get_fixed_policy_output()
        self.policy_output_dim = len(self.fixed_policy_output)
        self.policy_input_dim = len(self.state2policy())
        self.angle_rad = 2 * np.pi / self.n_angles
        # Oracle
        self.state2oracle = self.state2proxy
        self.statebatch2oracle = self.statebatch2proxy
        # Setup proxy
        self.proxy.set_n_dim(self.n_dim)

    def get_actions_space(self):
        """
        Constructs list with all possible actions. The actions are tuples with two
        values: (dimension, direction) where dimension indicates the index of the
        dimension on which the action is to be performed and direction indicates to
        increment or decrement with 1 or -1, respectively. The action "keep" is
        indicated by (-1, 0).
        """
        valid_steplens = np.arange(self.min_step_len, self.max_step_len + 1)
        dims = [a for a in range(self.n_dim)]
        directions = [1, -1]
        actions = []
        for r in valid_steplens:
            actions_r = [el for el in itertools.product(dims, directions, repeat=r)]
            actions += actions_r
        # Add "keep" action
        actions = actions + [(-1, 0)]
        # Add "eos" action
        actions = actions + [(self.eos, 0)]
        return actions

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        Returns a vector of length the action space + 1: True if action is invalid
        given the current state, False otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(len(self.action_space))]
        if state[-1] >= self.length_traj:
            mask = [True for _ in range(len(self.action_space))]
            mask[-1] = False
        else:
            mask = [False for _ in range(len(self.action_space))]
            mask[-1] = True
        return mask

    def true_density(self):
        # Return pre-computed true density if already stored
        if self._true_density is not None and self._log_z is not None:
            return self._true_density, self._log_z
        # Calculate true density
        x = self.get_all_terminating_states()
        rewards = self.reward_batch(x)
        self._z = rewards.sum()
        self._true_density = (
            rewards / self._z,
            rewards,
            list(map(tuple, x)),
        )
        import ipdb

        ipdb.set_trace()
        return self._true_density

    def fit_kde(x, kernel="exponential", bandwidth=0.1):
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(last_states.numpy())

    def statebatch2proxy(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Prepares a batch of states in "GFlowNet format" for the proxy: an array where
        each state is a row of length n_dim with an angle in radians. The n_actions
        item is removed.
        """
        return torch.tensor(states, device=self.device)[:, :-1] * self.angle_rad

    def statetorch2proxy(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for the proxy.
        """
        return states[:, :-1] * self.angle_rad

    # TODO: A circular encoding of the policy state would be better?
    def state2policy(self, state=None) -> List:
        """
        Transforms the angles part of the state given as argument (or self.state if
        None) into a one-hot encoding. The output is a list of len n_angles * n_dim +
        1, where each n-th successive block of length elements is a one-hot encoding of
        the position in the n-th dimension.

        Example, n_dim = 2, n_angles = 4:
          - State, state: [1, 3, 4]
                          | a  | n | (a = angles, n = n_actions)
          - state2policy(state): [0, 1, 0, 0, 0, 0, 0, 1, 4]
                                 |     1    |     3     | 4 |
        """
        if state is None:
            state = self.state.copy()
        # TODO: do we need float32?
        # TODO: do we need one-hot?
        state_policy = np.zeros(self.n_angles * self.n_dim + 1, dtype=np.float32)
        # Angles
        state_policy[: self.n_dim * self.n_angles][
            (np.arange(self.n_dim) * self.n_angles + state[: self.n_dim])
        ] = 1
        # Number of actions
        state_policy[-1] = state[-1]
        return state_policy

    def statebatch2policy(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Transforms a batch of states into the policy model format. The output is a numpy
        array of shape [n_states, n_angles * n_dim + 1].

        See state2policy().
        """
        states = np.array(states)
        cols = states[:, :-1] + np.arange(self.n_dim) * self.n_angles
        rows = np.repeat(np.arange(states.shape[0]), self.n_dim)
        state_policy = np.zeros(
            (len(states), self.n_angles * self.n_dim + 1), dtype=np.float32
        )
        state_policy[rows, cols.flatten()] = 1.0
        state_policy[:, -1] = states[:, -1]
        return state_policy

    def statetorch2policy(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "policy_output_dim"]:
        """
        Transforms a batch of torch states into the policy model format. The output is
        a tensor of shape [n_states, n_angles * n_dim + 1].

        See state2policy().
        """
        device = states.device
        cols = (
            states[:, :-1] + torch.arange(self.n_dim).to(device) * self.n_angles
        ).to(int)
        rows = torch.repeat_interleave(
            torch.arange(states.shape[0]).to(device), self.n_dim
        )
        state_policy = torch.zeros(
            (states.shape[0], self.n_angles * self.n_dim + 1)
        ).to(states)
        state_policy[rows, cols.flatten()] = 1.0
        state_policy[:, -1] = states[:, -1]
        return state_policy

    def policy2state(self, state_policy: List) -> List:
        """
        Transforms the one-hot encoding version of a state given as argument
        into a state (list of the position at each dimension).

        Example, n_dim = 2, n_angles = 4:
          - state_policy: [0, 1, 0, 0, 0, 0, 0, 1, 4]
                          |     0    |     3     | 4 |
          - policy2state(state_policy): [1, 3, 4]
                            | a  | n | (a = angles, n = n_actions)
        """
        mat_angles_policy = np.reshape(
            state_policy[: self.n_dim * self.n_angles], (self.n_dim, self.n_angles)
        )
        angles = np.where(mat_angles_policy)[1].tolist()
        return angles + [int(state_policy[-1])]

    def state2readable(self, state: List) -> str:
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state.
        """
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

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = self.source.copy()
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

    def get_parents(self, state=None, done=None, action=None):
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

        def _get_min_actions_to_source(source, ref):
            def _get_min_actions_dim(u, v):
                return np.min([np.abs(u - v), np.abs(u - (v - self.n_angles))])

            return np.sum([_get_min_actions_dim(u, v) for u, v in zip(source, ref)])

        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [(self.eos, 0)]
        # If source state
        elif state[-1] == 0:
            return [], []
        else:
            parents = []
            actions = []
            for idx, (a_dim, a_dir) in enumerate(self.action_space[:-1]):
                state_p = state.copy()
                angles_p = state_p[: self.n_dim]
                n_actions_p = state_p[-1]
                # Get parent
                n_actions_p -= 1
                if a_dim != -1:
                    angles_p[a_dim] -= a_dir
                    # If negative angle index, restart from the back
                    if angles_p[a_dim] < 0:
                        angles_p[a_dim] = self.n_angles + angles_p[a_dim]
                    # If angle index larger than n_angles, restart from 0
                    if angles_p[a_dim] >= self.n_angles:
                        angles_p[a_dim] = angles_p[a_dim] - self.n_angles
                if _get_min_actions_to_source(self.source_angles, angles_p) < state[-1]:
                    state_p = angles_p + [n_actions_p]
                    parents.append(state_p)
                    actions.append((a_dim, a_dir))
        return parents, actions

    def step(self, action: Tuple[int]) -> Tuple[List[int], Tuple[int, int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. See: get_actions_space()

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        assert action in self.action_space
        a_dim, a_dir = action
        if self.done:
            return self.state, action, False
        # If only possible action is eos, then force eos
        # If the number of actions is equal to trajectory length
        elif self.n_actions == self.length_traj:
            self.done = True
            self.n_actions += 1
            return self.state, (self.eos, 0), True
        # If action is not eos, then perform action
        elif a_dim != self.eos:
            angles_next = self.angles.copy()
            # If action is not "keep"
            if a_dim != -1:
                angles_next[a_dim] += a_dir
                # If negative angle index, restart from the back
                if angles_next[a_dim] < 0:
                    angles_next[a_dim] = self.n_angles + angles_next[a_dim]
                # If angle index larger than n_angles, restart from 0
                if angles_next[a_dim] >= self.n_angles:
                    angles_next[a_dim] = angles_next[a_dim] - self.n_angles
            self.angles = angles_next
            self.n_actions += 1
            self.state = self.angles + [self.n_actions]
            valid = True
            return self.state, action, valid
        # If action is eos, then it is invalid
        else:
            return self.state, (self.eos, 0), False

    def make_train_set(self, ntrain, oracle=None, seed=168, output_csv=None):
        """
        Constructs a randomly sampled train set.

        Args
        ----
        """
        rng = np.random.default_rng(seed)
        angles = rng.integers(low=0, high=self.n_angles, size=(ntrain,) + (self.n_dim,))
        n_actions = self.length_traj * np.ones([ntrain, 1], dtype=np.int32)
        samples = np.concatenate([angles, n_actions], axis=1)
        if oracle:
            energies = oracle(self.state2oracle(samples))
        else:
            energies = self.oracle(self.state2oracle(samples))
        df_train = pd.DataFrame({"samples": list(samples), "energies": energies})
        if output_csv:
            df_train.to_csv(output_csv)
        return df_train

    def make_test_set(self, config):
        """
        Constructs a test set.

        Args
        ----
        """
        if "all" in config and config.all:
            samples = self.get_all_terminating_states()
            energies = self.oracle(self.state2oracle(samples))
        df_test = pd.DataFrame(
            {"samples": [self.state2readable(s) for s in samples], "energies": energies}
        )
        return df_test

    def get_all_terminating_states(self):
        all_x = np.int32(
            list(itertools.product(*[list(range(self.n_angles))] * self.n_dim))
        )
        n_actions = self.length_traj * np.ones([all_x.shape[0], 1], dtype=np.int32)
        all_x = np.concatenate([all_x, n_actions], axis=1)
        return all_x.tolist()
