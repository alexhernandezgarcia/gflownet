"""
Classes to represent hyper-torus environments
"""
from typing import List
import itertools
import numpy as np
import pandas as pd
from gflownet.envs.base import GFlowNetEnv


class Torus(GFlowNetEnv):
    """
    Hyper-torus environment

    Attributes
    ----------
    ndim : int
        Dimensionality of the torus

    n_angles : int
        Number of angles into which each dimension is divided

    max_rounds : int
        If larger than one, the action space allows for reaching the initial angle and
        restart again up to max_rounds; and the state space contain the round number.
        If zero, only one round is allowed, without reaching the initial angle.
    """

    def __init__(
        self,
        n_dim=2,
        n_angles=3,
        max_rounds=1,
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
        self.max_rounds = max_rounds
        # TODO: do we need to one-hot encode the coordinates and rounds?
        self.angles = [0 for _ in range(self.n_dim)]
        self.rounds = [0 for _ in range(self.n_dim)]
        # States are the concatenation of the angle state and the round state
        self.state = self.angles + self.rounds
        # TODO: A circular encoding of obs would be better
        self.obs_dim = self.n_angles * self.n_dim * 2
        self.min_step_len = min_step_len
        self.max_step_len = max_step_len
        self.action_space = self.get_actions_space()
        self.eos = len(self.action_space)
        self.angle_rad = 2 * np.pi / self.n_angles

    def get_actions_space(self):
        """
        Constructs list with all possible actions
        """
        valid_steplens = np.arange(self.min_step_len, self.max_step_len + 1)
        dims = [a for a in range(self.n_dim)]
        actions = []
        for r in valid_steplens:
            actions_r = [el for el in itertools.product(dims, repeat=r)]
            actions += actions_r
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
            return [True for _ in range(len(self.action_space) + 1)]
        mask = [False for _ in range(len(self.action_space) + 1)]
        for idx, a in enumerate(self.action_space):
            for d in a:
                if (
                    state[d] + 1 >= self.n_angles
                    and state[self.n_dim + d] + 1 >= self.max_rounds
                ):
                    mask[idx] = True
                    break
        return mask

    def true_density(self):
        # Return pre-computed true density if already stored
        if self._true_density is not None:
            return self._true_density
        # Calculate true density
        all_angles = np.int32(
            list(itertools.product(*[list(range(self.n_angles))] * self.n_dim))
        )
        all_oracle = self.state2oracle(all_angles)
        rewards = self.oracle(all_oracle)
        self._true_density = (
            rewards / rewards.sum(),
            rewards,
            list(map(tuple, all_angles)),
        )
        return self._true_density

    def state2proxy(self, state_list):
        """
        Prepares a list of states in "GFlowNet format" for the proxy: a list of length
        n_dim with an angle in radians.

        Args
        ----
        state_list : list of lists
            List of states.
        """
        # TODO: do we really need to convert back to list?
        # TODO: split angles and round?
        return (np.array(state_list) * self.angle_rad).tolist()

    def state2oracle(self, state_list):
        """
        Prepares a list of states in "GFlowNet format" for the oracles: a list of length
        n_dim with values in the range [cell_min, cell_max] for each state.

        Args
        ----
        state_list : list of lists
            List of states.
        """
        return self.state2proxy(state_list)

    def state2policy(self, state=None):
        """
        Transforms the state given as argument (or self.state if None) into a
        one-hot encoding. The output is a list of len n_angles * n_dim,
        where each n-th successive block of length elements is a one-hot encoding of
        the position in the n-th dimension.

        Example, n_dim = 2, n_angles = 4:
          - State, state: [0, 3, 1, 0]
                          | a  |  r  | (a = angles, r = rounds)
          - state2policy(state): [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
                              |     0    |      3    |      1    |      0    |
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
        # Rounds
        obs[self.n_dim * self.n_angles :][
            (np.arange(self.n_dim) * self.n_angles + state[self.n_dim :])
        ] = 1
        return obs

    def obs2state(self, obs: List) -> List:
        """
        Transforms the one-hot encoding version of a state given as argument
        into a state (list of the position at each dimension).

        Example, n_dim = 2, n_angles = 4:
          - obs: [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
                 |     0    |      3    |      1    |      0    |
          - obs2state(obs): [0, 3, 1, 0]
                            | a  |  r  | (a = angles, r = rounds)
        """
        obs_mat_angles = np.reshape(
            obs[: self.n_dim * self.n_angles], (self.n_dim, self.n_angles)
        )
        obs_mat_rounds = np.reshape(
            obs[self.n_dim * self.n_angles :], (self.n_dim, self.n_angles)
        )
        angles = np.where(obs_mat_angles)[1]
        rounds = np.where(obs_mat_rounds)[1]
        # TODO: do we need to convert to list?
        return np.concatenate([angles, rounds]).tolist()

    def state2readable(self, state, alphabet={}):
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
        rounds = (
            str(state[self.n_dim :])
            .replace("(", "[")
            .replace(")", "]")
            .replace(",", "")
        )
        return angles + " | " + rounds

    def readable2state(self, readable, alphabet={}):
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        pair = readable.split(" | ")
        angles = [int(el) for el in pair[0].strip("[]").split(" ")]
        rounds = [int(el) for el in pair[1].strip("[]").split(" ")]
        return angles + rounds

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.angles = [0 for _ in range(self.n_dim)]
        self.rounds = [0 for _ in range(self.n_dim)]
        self.state = self.angles + self.rounds
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

    def get_parents(self, state=None, done=None):
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state, as a list of length n_angles where each element is
            the position at each dimension.

        action : int
            Last action performed

        Returns
        -------
        parents : list
            List of parents as state2policy(state)

        actions : list
            List of actions that lead to state for each parent in parents
        """
        # TODO
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [self.state2policy(state)], [self.eos]
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                state_aux = state.copy()
                angles_aux = state_aux[: self.n_dim]
                rounds_aux = state_aux[self.n_dim :]
                for a_sub in a:
                    if angles_aux[a_sub] == 0 and rounds_aux[a_sub] > 0:
                        angles_aux[a_sub] = self.n_angles - 1
                        rounds_aux[a_sub] -= 1
                    elif angles_aux[a_sub] > 0:
                        angles_aux[a_sub] -= 1
                    else:
                        break
                else:
                    state_aux = angles_aux + rounds_aux
                    parents.append(self.state2policy(state_aux))
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
        # If only possible action is eos, then force eos
        # All dimensions are at the maximum angle and maximum round
        if all([a == self.n_angles - 1 for a in self.angles]) and all(
            [r == self.max_rounds - 1 for r in self.rounds]
        ):
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If action is not eos, then perform action
        if action_idx != self.eos:
            action = self.action_space[action_idx]
            angles_next = self.angles.copy()
            rounds_next = self.rounds.copy()
            for a in action:
                angles_next[a] += 1
                # Increment round and reset angle if necessary
                if angles_next[a] == self.n_angles:
                    angles_next[a] = 0
                    rounds_next[a] += 1
            if any([r >= self.max_rounds for r in rounds_next]):
                valid = False
            else:
                self.angles = angles_next
                self.rounds = rounds_next
                self.state = self.angles + self.rounds
                valid = True
                self.n_actions += 1
            return self.state, action_idx, valid
        # If action is eos, then perform eos
        else:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True

    def make_train_set(self, ntrain, oracle=None, seed=168, output_csv=None):
        """
        Constructs a randomly sampled train set.

        Args
        ----
        """
        rng = np.random.default_rng(seed)
        angles = rng.integers(low=0, high=self.n_angles, size=(ntrain,) + (self.n_dim,))
        rounds = rng.integers(
            low=0, high=self.max_rounds, size=(ntrain,) + (self.n_dim,)
        )
        samples = np.concatenate([angles, rounds], axis=1)
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
            list(
                itertools.product(
                    *[list(range(self.n_angles))] * self.n_dim
                    + [list(range(self.max_rounds))] * self.n_dim
                )
            )
        )
        return all_x
