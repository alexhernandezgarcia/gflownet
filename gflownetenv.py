"""
Base class of GFlowNet environments
"""
import numpy as np
import pandas as pd
from pathlib import Path


class GFlowNetEnv:
    """
    Base class of GFlowNet environments
    """

    def __init__(
        self,
        env_id=None,
        reward_beta=1,
        reward_norm=1.0,
        reward_func="power",
        energies_stats=None,
        denorm_proxy=False,
        proxy=None,
        oracle_func=None,
        debug=False,
    ):
        self.state = []
        self.done = False
        self.n_actions = 0
        self.id = env_id
        self.min_reward = 1e-8
        self.reward_beta = reward_beta
        self.reward_norm = reward_norm
        self.reward_func = reward_func
        self.energies_stats = energies_stats
        self.denorm_proxy = denorm_proxy
        self.oracle = oracle_func
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
        self.action_space = []
        self.eos = len(self.action_space)
        # Assertions
        assert self.reward_norm > 0
        assert self.reward_beta > 0
        assert self.min_reward > 0

    def set_energies_stats(self, energies_stats):
        self.energies_stats = energies_stats

    def set_reward_norm(self, reward_norm):
        self.reward_norm = reward_norm

    def get_actions_space(self):
        """
        Constructs list with all possible actions (excluding end of sequence)
        """
        return []

    def get_max_path_len(
        self,
    ):
        return 1

    def state2oracle(self, state_list):
        """
        Prepares a list of states in "GFlowNet format" for the oracle

        Args
        ----
        state_list : list of lists
            List of states.
        """
        return state_list

    def reward_batch(self, states, done):
        """
        Computes the rewards of a batch of states, given a list of states and 'dones'
        """
        states = [s for s, d in zip(states, done) if d]
        reward = np.zeros(len(done))
        reward[list(done)] = self.proxy2reward(self.proxy(self.state2oracle(states)))
        return reward

    def proxy2reward(self, proxy_vals):
        """
        Prepares the output of an oracle for GFlowNet: the inputs proxy_vals is
        expected to be a negative value (energy), unless self.denorm_proxy is True. If
        the latter, the proxy values are first de-normalized according to the mean and
        standard deviation in self.energies_stats. The output of the function is a
        strictly positive reward - provided self.reward_norm and self.reward_beta are
        positive - and larger than self.min_reward.
        """
        if self.denorm_proxy:
            proxy_vals = proxy_vals * self.energies_stats[3] + self.energies_stats[2]
        if self.reward_func == "power":
            return np.clip(
                (-1.0 * proxy_vals / self.reward_norm) ** self.reward_beta,
                self.min_reward,
                None,
            )
        elif self.reward_func == "boltzmann":
            return np.clip(
                np.exp(-1.0 * self.reward_beta * proxy_vals),
                self.min_reward,
                None,
            )
        else:
            raise NotImplemented

    def reward2proxy(self, reward):
        """
        Converts a "GFlowNet reward" into a (negative) energy or values as returned by
        an oracle.
        """
        if self.reward_func == "power":
            return -np.exp(
                (np.log(reward) + self.reward_beta * np.log(self.reward_norm))
                / self.reward_beta
            )
        elif self.reward_func == "boltzmann":
            return -1.0 * np.log(reward) / self.reward_beta
        else:
            raise NotImplemented

    def state2obs(self, state=None):
        """
        Converts a state into a format suitable for a machine learning model, such as a
        one-hot encoding.
        """
        if state is None:
            state = self.state
        return state

    def obs2state(self, obs):
        """
        Converts the model (e.g. one-hot encoding) version of a state given as
        argument into a state.
        """
        return obs

    def state2readable(self, state=None):
        """
        Converts a state into human-readable representation.
        """
        if state is None:
            state = self.state
        return str(state)

    def readable2state(self, readable):
        """
        Converts a human-readable representation of a state into the standard format.
        """
        return readable

    def path2readable(self, path=None):
        """
        Converts a path into a human-readable string.
        """
        return str(path).replace("(", "[").replace(")", "]").replace(",", "")

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
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state

        action : int
            Last action performed

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
        return parents, actions

    def get_paths(self, path_list, actions):
        """
        Determines all paths leading to each state in path_list, recursively.

        Args
        ----
        path_list : list
            List of paths (lists)

        actions : list
            List of actions within each path

        Returns
        -------
        path_list : list
            List of paths (lists)

        actions : list
            List of actions within each path
        """
        current_path = path_list[-1].copy()
        current_path_actions = actions[-1].copy()
        parents, parents_actions = self.get_parents(list(current_path[-1]), False)
        parents = [self.obs2state(el).tolist() for el in parents]
        if parents == []:
            return path_list, actions
        for idx, (p, a) in enumerate(zip(parents, parents_actions)):
            if idx > 0:
                path_list.append(current_path)
                actions.append(current_path_actions)
            path_list[-1] += [p]
            actions[-1] += [a]
            path_list, actions = self.get_paths(path_list, actions)
        return path_list, actions

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
        if action < self.eos:
            self.done = False
            valid = True
        else:
            self.done = True
            valid = True
            self.n_actions += 1
        return self.state, action, valid

    def no_eos_mask(self, state=None):
        """
        Returns True if no eos action is allowed given state
        """
        if state is None:
            state = self.state
        return False

    def get_mask_invalid_actions(self, state=None):
        """
        Returns a vector of length the action space + 1: True if action is invalid
        given the current state, False otherwise.
        """
        if state is None:
            state = self.state
        mask = [False for _ in range(len(self.action_space) + 1)]
        return mask

    def set_state(self, state, done):
        """
        Sets the state and done of an environment.
        """
        self.state = state
        self.done = done
        return self

    def true_density(self):
        """
        Computes the reward density (reward / sum(rewards)) of the whole space

        Returns
        -------
        Tuple:
          - normalized reward for each state
          - un-normalized reward
          - states
        """
        return (None, None, None)

    def make_train_set(self, ntrain, oracle=None, seed=168, output_csv=None):
        """
        Constructs a randomly sampled train set.

        Args
        ----
        """
        return None

    def make_test_set(
        self,
        ntest,
        oracle=None,
        seed=167,
        output_csv=None,
    ):
        """
        Constructs a test set.

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


class Buffer:
    """
    Implements the functionality to manage various buffers of data: the records of
    training samples, the train and test data sets, a replay buffer for training, etc.
    """

    def __init__(self, env, replay_capacity=0, output_csv=None):
        self.env = env
        self.replay_capacity = replay_capacity
        self.action_space = self.env.get_actions_space()
        self.main = pd.DataFrame(columns=["state", "path", "reward", "energy", "iter"])
        self.replay = pd.DataFrame(
            np.empty((self.replay_capacity, 5), dtype=object),
            columns=["state", "path", "reward", "energy", "iter"],
        )
        self.replay.reward = pd.to_numeric(self.replay.reward)
        self.replay.energy = pd.to_numeric(self.replay.energy)
        self.replay.reward = [-1 for _ in range(self.replay_capacity)]
        self.train = None
        self.test = None

    def add(
        self,
        states,
        paths,
        rewards,
        energies,
        it,
        buffer="main",
        criterion="greater",
    ):
        if buffer == "main":
            self.main = self.main.append(
                pd.DataFrame(
                    {
                        "state": [self.env.state2readable(s) for s in states],
                        "path": [self.env.path2readable(p) for p in paths],
                        "reward": rewards,
                        "energy": energies,
                        "iter": it,
                    }
                )
            )
        elif buffer == "replay" and self.replay_capacity > 0:
            if criterion == "greater":
                self.replay = self._add_greater(states, paths, rewards, energies, it)

    def _add_greater(
        self,
        states,
        paths,
        rewards,
        energies,
        it,
    ):
        rewards_old = self.replay["reward"].values
        rewards_new = rewards.copy()
        while np.max(rewards_new) > np.min(rewards_old):
            idx_new_max = np.argmax(rewards_new)
            self.replay.iloc[self.replay.reward.argmin()] = {
                "state": self.env.state2readable(states[idx_new_max]),
                "path": self.env.path2readable(paths[idx_new_max]),
                "reward": rewards[idx_new_max],
                "energy": energies[idx_new_max],
                "iter": it,
            }
            rewards_new[idx_new_max] = -1
            rewards_old = self.replay["reward"].values
        return self.replay

    def make_train_test(
        self, data_path=None, train_path=None, test_path=None, oracle=None, *args
    ):
        """
        Initializes the train and test sets. Depending on the arguments, the sets can
        be formed in different ways:

        (1) data_path is not None and is ".npy": data_path is the path to a npy file
        containing data set of the (aptamers) active learning pipeline. An aptamers
        specific function creates the train/test split.
        (2) separate train and test file paths are provided
        (3) no file paths are provided and the train and test data are generated by
          environment-specific functions.
        """
        # (1) data_path is not None
        if data_path:
            data_path = Path(data_path)
            if data_path.suffix == ".npy":
                df_data = self.env.np2df(
                    data_path,
                    args[0].dataset.init_length,
                    args[0].al.queries_per_iter,
                    args[0].gflownet.test.pct_test,
                    args[0].seeds.dataset,
                )
                self.train = df_data.loc[df_data.train]
                self.test = df_data.loc[df_data.test]
        # Otherwise
        else:
            # Train set
            # (2) Separate train file path is provided
            if train_path:
                self.train = pd.read_csv(train_path, index_col=0)
            # (3) Make environment specific train set
            elif oracle is not None:
                self.train = self.env.make_train_set(
                    ntrain=args[0].gflownet.train.n,
                    oracle=oracle,
                    seed=args[0].gflownet.train.seed,
                    output_csv=args[0].gflownet.train.output,
                )
            # Test set
            # (2) Separate test file path is provided
            if test_path:
                self.test = pd.read_csv(test_path, index_col=0)
            # (3) Make environment specific test set
            else:
                self.test, _ = self.env.make_test_set(
                    path_base_dataset=args[0].gflownet.test.base,
                    ntest=args[0].gflownet.test.n,
                    min_length=args[0].gflownet.min_seq_length,
                    max_length=args[0].gflownet.max_seq_length,
                    seed=args[0].gflownet.test.seed,
                    output_csv=args[0].gflownet.test.output,
                )

    def sample(
        self,
    ):
        pass

    def __len__(self):
        return self.capacity

    @property
    def transitions(self):
        pass

    def save(
        self,
    ):
        pass

    @classmethod
    def load():
        pass

    @property
    def dummy(self):
        pass
