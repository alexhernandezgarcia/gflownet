"""
Classes to represent a hyper-grid environments
"""
import itertools
import numpy as np


class Grid:
    """
    Hyper-grid environment

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
        n_dim=2,
        length=4,
        min_step_len=1,
        max_step_len=1,
        cell_min=-1,
        cell_max=1,
        env_id=None,
        reward_beta=1,
        reward_norm=1.0,
        denorm_proxy=False,
        stats_scores=[-1.0, 0.0, 0.5, 1.0, -1.0],
        proxy=None,
        oracle_func="default",
        debug=False,
    ):
        self.n_dim = n_dim
        self.state = [0] * self.n_dim
        self.length = length
        self.obs_dim = self.length * self.n_dim
        self.min_step_len = min_step_len
        self.max_step_len = max_step_len
        self.cells = np.linspace(cell_min, cell_max, length)
        self.done = False
        self.id = env_id
        self.n_actions = 0
        self.stats_scores = stats_scores
        self.oracle = {
            "default": None,
            "cos_N": self.func_cos_N,
            "corners": self.func_corners,
            "corners_floor_A": self.func_corners_floor_A,
            "corners_floor_B": self.func_corners_floor_B,
        }[oracle_func]
        if proxy:
            self.proxy = proxy
        else:
            self.proxy = self.oracle
        self.reward = (
            lambda x: [0]
            if not self.done
            else self.proxy2reward(self.proxy(self.state2oracle(x)))
        )
        self._true_density = None
        self.debug = debug
        self.reward_beta = reward_beta
        self.min_reward = 1e-8
        self.reward_norm = reward_norm
        self.denorm_proxy = denorm_proxy
        self.action_space = self.get_actions_space(
            self.n_dim, np.arange(self.min_step_len, self.max_step_len + 1)
        )
        self.eos = len(self.action_space)
        # Aliases and compatibility
        self.seq = self.state
        self.seq2obs = self.state2obs
        self.obs2seq = self.obs2state

    def get_actions_space(self, n_dim, valid_steplens):
        """
        Constructs list with all possible actions
        """
        dims = [a for a in range(n_dim)]
        actions = []
        for r in valid_steplens:
            actions_r = [el for el in itertools.product(dims, repeat=r)]
            actions += actions_r
        return actions

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

    def reward_batch(self, state, done):
        state = [s for s, d in zip(state, done) if d]
        reward = np.zeros(len(done))
        reward[list(done)] = self.proxy2reward(self.proxy(self.state2oracle(state)))
        return reward

    def proxy2reward(self, proxy_vals):
        """
        Prepares the output of an oracle for GFlowNet.
        """
        if self.denorm_proxy:
            proxy_vals = proxy_vals * self.stats_scores[3] + self.stats_scores[2]
        return np.clip(
            (-1.0 * proxy_vals / self.reward_norm) ** self.reward_beta,
            self.min_reward,
            None,
        )

    def reward2proxy(self, reward):
        """
        Converts a "GFlowNet reward" into energy or values as returned by an oracle.
        """
        return -np.exp(
            (np.log(reward) + self.reward_beta * np.log(self.reward_norm))
            / self.reward_beta
        )

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
            state = self.state
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[(np.arange(len(state)) * self.length + state)] = 1
        return obs

    def obs2state(self, obs):
        """
        Transforms the one-hot encoding version of a state given as argument
        into a state (list of the position at each dimension).

        Example:
          - obs: [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0] (length = 4, n_dim = 3)
                 |     0    |      3    |      1    |
          - obs2state(obs): [0, 3, 1]
        """
        obs_mat = np.reshape(obs, (self.n_dim, self.length))
        state = np.where(obs_mat)[1]
        return state

    def seq2letters(self, state, alphabet={}):
        """
        Dummy function for compatibility reasons.
        """
        return str(state)

    def letters2seq(self, letters, alphabet={}):
        """
        Dummy function for compatibility reasons.
        """
        return letters

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = [0] * self.n_dim
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

    def parent_transitions(self, state, action):
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state, as a list of length length where each element is
            the position at each dimension.

        action : int
            Last action performed

        Returns
        -------
        parents : list
            List of parents as state2obs(state)

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if action == self.eos:
            return [self.state2obs(state)], [action]
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                state_aux = state.copy()
                for a_sub in a:
                    if state_aux[a_sub] > 0:
                        state_aux[a_sub] -= 1 
                    else:
                        break
                else:
                    parents.append(self.state2obs(state_aux))
                    actions.append(idx)
        return parents, actions

    def get_trajectories(self, traj_list, actions):
        """
        Determines all trajectories leading to each state in traj_list, recursively.

        Args
        ----
        traj_list : list
            List of trajectories (lists)

        actions : list
            List of actions within each trajectory

        Returns
        -------
        traj_list : list
            List of trajectories (lists)

        actions : list
            List of actions within each trajectory
        """
        current_traj = traj_list[-1].copy()
        current_traj_actions = actions[-1].copy()
        parents, parents_actions = self.parent_transitions(list(current_traj[-1]), -1)
        parents = [self.obs2state(el).tolist() for el in parents]
        if parents == []:
            return traj_list, actions
        for idx, (p, a) in enumerate(zip(parents, parents_actions)):
            if idx > 0:
                traj_list.append(current_traj)
                actions.append(current_traj_actions)
            traj_list[-1] += [p]
            actions[-1] += [a]
            traj_list, actions = self.get_trajectories(traj_list, actions)
        return traj_list, actions

    def step(self, action):
        """
        Executes step given an action.

        Args
        ----
        a : int (tensor)
            Index of action in the action space. a == eos indicates "stop action"

        Returns
        -------
        self.state : list
            The sequence after executing the action

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        # All dimensions are at the maximum length
        if all([s == self.length - 1 for s in self.state]):
            self.done = True
            self.n_actions += 1
            return self.state, True
        if action < self.eos:
            state_next = self.state.copy()
            if action.ndim == 0:
                action = [action]
            for a in action:
                state_next[a] += 1
            if any([s >= self.length for s in state_next]):
                valid = False
            else:
                self.state = state_next
                valid = True
                self.n_actions += 1
        else:
            self.done = True
            valid = True
            self.n_actions += 1

        return self.state, valid

    def no_eos_mask(self, state=None):
        """
        Returns True if no eos action is allowed given state
        """
        if state is None:
            state = self.state
        return False

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
        if self.nalphabet ** self.max_state_length > max_states:
            return (None, None, None)
        state_all = np.int32(
            list(
                itertools.product(
                    *[list(range(self.nalphabet))] * self.max_state_length
                )
            )
        )
        traj_rewards, state_end = zip(
            *[
                (self.proxy(state), state)
                for state in state_all
                if len(self.parent_transitions(state, 0)[0]) > 0 or sum(state) == 0
            ]
        )
        traj_rewards = np.array(traj_rewards)
        self._true_density = (
            traj_rewards / traj_rewards.sum(),
            list(map(tuple, state_end)),
            traj_rewards,
        )
        return self._true_density

    @staticmethod
    def func_corners(x_list):
        def _func_corners(x):
            ax = abs(x)
            return (
                (ax > 0.5).prod(-1) * 0.5
                + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2
                + 1e-1
            )

        return np.asarray([_func_corners(x) for x in x_list])

    @staticmethod
    def func_corners_floor_B(x_list):
        def _func_corners_floor_B(x_list):
            ax = abs(x)
            return (
                (ax > 0.5).prod(-1) * 0.5
                + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2
                + 1e-2
            )

        return np.asarray([_func_corners_floor_B(x) for x in x_list])

    @staticmethod
    def func_corners_floor_A(x_list):
        def _func_corners_floor_A(x_list):
            ax = abs(x)
            return (
                (ax > 0.5).prod(-1) * 0.5
                + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2
                + 1e-3
            )

        return np.asarray([_func_corners_floor_A(x) for x in x_list])

    @staticmethod
    def func_cos_N(x_list):
        def _func_cos_N(x_list):
            ax = abs(x)
            return ((np.cos(x * 50) + 1) * norm.pdf(x * 5)).prod(-1) + 0.01

        return np.asarray([_func_cos_N(x) for x in x_list])

    @staticmethod
    def make_train_set(*args):
        """
        Constructs a randomly sampled train set.

        Args
        ----
        """
        return None

    @staticmethod
    def make_approx_uniform_test_set(*args):
        """
        Constructs an approximately uniformly distributed (on the score) set, by
        selecting samples from a larger base set.

        Args
        ----
        """
        return None

    @staticmethod
    def np2df(*args):
        """
        Args
        ----
        """
        return None
