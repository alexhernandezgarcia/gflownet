"""
Classes to represent hyper-torus environments
"""
from typing import List
import itertools
import numpy as np
import pandas as pd
from src.gflownet.envs.base import GFlowNetEnv


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
        env_id=None,
        reward_beta=1,
        reward_norm=1.0,
        reward_norm_std_mult=0,
        reward_func="boltzmann",
        denorm_proxy=False,
        energies_stats=None,
        proxy=None,
        oracle=None,
        **kwargs,
    ):
        super(Torus, self).__init__(
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
        self.n_dim = n_dim
        self.n_angles = n_angles
        self.length_traj = length_traj
        # Initialize angles and state attributes
        self.reset()
        self.source = self.angles.copy()
        # TODO: A circular encoding of obs would be better?
        self.obs_dim = self.n_angles * self.n_dim + 1
        self.min_step_len = min_step_len
        self.max_step_len = max_step_len
        self.action_space = self.get_actions_space()
        self.eos = len(self.action_space)
        self.angle_rad = 2 * np.pi / self.n_angles

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
        actions = actions + [(-1, 0)]
        return actions

    def get_mask_invalid_actions(self, state=None, done=None):
        """
        Returns a vector of length the action space + 1: True if action is invalid
        given the current state, False otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(len(self.action_space) + 1)]
        if state[-1] >= self.length_traj:
            mask = [True for _ in range(len(self.action_space) + 1)]
            mask[-1] = False
        else:
            mask = [False for _ in range(len(self.action_space) + 1)]
            mask[-1] = True
        return mask

    def true_density(self):
        # Return pre-computed true density if already stored
        if self._true_density is not None:
            return self._true_density
        # Calculate true density
        all_x = self.get_all_terminating_states()
        all_oracle = self.state2oracle(all_x)
        rewards = self.oracle(all_oracle)
        self._true_density = (
            rewards / rewards.sum(),
            rewards,
            list(map(tuple, all_x)),
        )
        return self._true_density

    def state2proxy(self, state_list: List[List]) -> List[List]:
        """
        Prepares a list of states in "GFlowNet format" for the proxy: a list of length
        n_dim with an angle in radians. The n_actions item is removed.

        Args
        ----
        state_list : list of lists
            List of states.
        """
        # TODO: do we really need to convert back to list?
        # TODO: split angles and round?
        return (np.array(state_list)[:, :-1] * self.angle_rad).tolist()

    def state2oracle(self, state_list: List[List]) -> List[List]:
        """
        Prepares a list of states in "GFlowNet format" for the oracle

        Args
        ----
        state_list : list of lists
            List of states.
        """
        return self.state2proxy(state_list)

    def state2obs(self, state=None) -> List:
        """
        Transforms the angles part of the state given as argument (or self.state if
        None) into a one-hot encoding. The output is a list of len n_angles * n_dim +
        1, where each n-th successive block of length elements is a one-hot encoding of
        the position in the n-th dimension.

        Example, n_dim = 2, n_angles = 4:
          - State, state: [1, 3, 4]
                          | a  | n | (a = angles, n = n_actions)
          - state2obs(state): [0, 1, 0, 0, 0, 0, 0, 1, 4]
                              |     1    |     3     | 4 |
        """
        if state is None:
            state = self.state.copy()
        # TODO: do we need float32?
        # TODO: do we need one-hot?
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        # Angles
        obs[: self.n_dim * self.n_angles][
            (np.arange(self.n_dim) * self.n_angles + state[: self.n_dim])
        ] = 1
        # Number of actions
        obs[-1] = state[-1]
        return obs

    def obs2state(self, obs: List) -> List:
        """
        Transforms the one-hot encoding version of a state given as argument
        into a state (list of the position at each dimension).

        Example, n_dim = 2, n_angles = 4:
          - obs: [0, 1, 0, 0, 0, 0, 0, 1, 4]
                 |     0    |     3     | 4 |
          - obs2state(obs): [1, 3, 4]
                            | a  | n | (a = angles, n = n_actions)
        """
        obs_mat_angles = np.reshape(
            obs[: self.n_dim * self.n_angles], (self.n_dim, self.n_angles)
        )
        angles = np.where(obs_mat_angles)[1].tolist()
        return angles + [int(obs[-1])]

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
        # TODO: random start
        self.angles = [0 for _ in range(self.n_dim)]
        self.n_actions = 0
        # States are the concatenation of the angle state and number of actions
        self.state = self.angles + [self.n_actions]
        self.done = False
        self.id = env_id
        return self

    def get_parents(self, state=None, done=None):
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state, as a list of length n_angles where each element
            is the position at each dimension.

        action : int
            Last action performed

        Returns
        -------
        parents : list
            List of parents as state2obs(state)

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
            return [self.state2obs(state)], [self.eos]
        # If source state
        elif state[-1] == 0:
            return [], []
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                state_p = state.copy()
                angles_p = state_p[: self.n_dim]
                n_actions_p = state_p[-1]
                # Get parent
                n_actions_p -= 1
                if a[0] != -1:
                    angles_p[a[0]] -= a[1]
                    # If negative angle index, restart from the back
                    if angles_p[a[0]] < 0:
                        angles_p[a[0]] = self.n_angles + angles_p[a[0]]
                    # If angle index larger than n_angles, restart from 0
                    if angles_p[a[0]] >= self.n_angles:
                        angles_p[a[0]] = angles_p[a[0]] - self.n_angles
                if _get_min_actions_to_source(self.source, angles_p) < state[-1]:
                    state_p = angles_p + [n_actions_p]
                    parents.append(self.state2obs(state_p))
                    actions.append(idx)
        return parents, actions

    def step(self, action_idx):
        """
        Executes step given an action index.

        Args
        ----
        action_idx : int
            Index of action in the action space. a == eos indicates "stop action"

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action_idx : int
            Action index

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        if self.done:
            return self.state, action_idx, False
        # If only possible action is eos, then force eos
        # If the number of actions is equal to maximum trajectory length
        elif self.n_actions == self.length_traj:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If action is not eos, then perform action
        elif action_idx != self.eos:
            a_dim, a_dir = self.action_space[action_idx]
            angles_next = self.angles.copy()
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
            return self.state, action_idx, valid
        # If action is eos, then it is invalid
        else:
            return self.state, self.eos, False

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
        return all_x
