"""
Buffer class to handle train and test data sets, reply buffer, etc.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


class Buffer:
    """
    Implements the functionality to manage various buffers of data: the records of
    training samples, the train and test data sets, a replay buffer for training, etc.
    """

    def __init__(
        self,
        env,
        make_train_test=False,
        replay_capacity=0,
        output_csv=None,
        data_path=None,
        train=None,
        test=None,
        logger=None,
        **kwargs,
    ):
        self.logger = logger
        self.env = env
        self.replay_capacity = replay_capacity
        self.main = pd.DataFrame(columns=["state", "traj", "reward", "energy", "iter"])
        self.replay = pd.DataFrame(
            np.empty((self.replay_capacity, 5), dtype=object),
            columns=["state", "traj", "reward", "energy", "iter"],
        )
        self.replay.reward = pd.to_numeric(self.replay.reward)
        self.replay.energy = pd.to_numeric(self.replay.energy)
        self.replay.reward = [-1 for _ in range(self.replay_capacity)]
        # Define train and test data sets
        if train is not None and "type" in train:
            self.train_type = train.type
        else:
            self.train_type = None
        self.train, dict_tr, train_stats = self.make_data_set(train)
        if (
            self.train is not None
            and "output_csv" in train
            and train.output_csv is not None
        ):
            self.train.to_csv(train.output_csv)
        if (
            dict_tr is not None
            and "output_pkl" in train
            and train.output_pkl is not None
        ):
            with open(train.output_pkl, "wb") as f:
                pickle.dump(dict_tr, f)
                self.train_pkl = train.output_pkl
        else:
            print(
                """
            Important: offline trajectories will NOT be sampled. In order to sample
            offline trajectories, the train configuration of the buffer should be
            complete and feasible and an output pkl file should be defined in
            env.buffer.train.output_pkl.
            """
            )
            self.train_pkl = None
        if test is not None and "type" in test:
            self.test_type = test.type
        else:
            self.train_type = None
        self.test, dict_tt, test_stats = self.make_data_set(test)
        if (
            self.test is not None
            and "output_csv" in test
            and test.output_csv is not None
        ):
            self.test.to_csv(test.output_csv)
        if dict_tt is not None and "output_pkl" in test and test.output_pkl is not None:
            with open(test.output_pkl, "wb") as f:
                pickle.dump(dict_tt, f)
                self.test_pkl = test.output_pkl
        else:
            print(
                """
            Important: test metrics will NOT be computed. In order to compute
            test metrics the test configuration of the buffer should be complete and
            feasible and an output pkl file should be defined in
            env.buffer.test.output_pkl.
            """
            )
            self.test_pkl = None
        # Compute buffer statistics
        if self.train is not None:
            (self.mean_tr, self.std_tr, self.min_tr, self.max_tr, self.max_norm_tr,) = (
                train_stats[0],
                train_stats[1],
                train_stats[2],
                train_stats[3],
                train_stats[4],
            )
        if self.test is not None:
            self.mean_tt, self.std_tt, self.min_tt, self.max_tt, _ = (
                test_stats[0],
                test_stats[1],
                test_stats[2],
                test_stats[3],
                test_stats[4],
            )

    def add(
        self,
        states,
        trajs,
        rewards,
        energies,
        it,
        buffer="main",
        criterion="greater",
    ):
        if buffer == "main":
            self.main = pd.concat(
                [
                    self.main,
                    pd.DataFrame(
                        {
                            "state": [self.env.state2readable(s) for s in states],
                            "traj": [self.env.traj2readable(p) for p in trajs],
                            "reward": rewards,
                            "energy": energies,
                            "iter": it,
                        }
                    ),
                ],
                axis=0,
                join="outer",
            )
        elif buffer == "replay" and self.replay_capacity > 0:
            if criterion == "greater":
                self.replay = self._add_greater(states, trajs, rewards, energies, it)

    def _add_greater(
        self,
        states,
        trajs,
        rewards,
        energies,
        it,
    ):
        rewards_old = self.replay["reward"].values
        rewards_new = rewards.copy()
        while np.max(rewards_new) > np.min(rewards_old):
            idx_new_max = np.argmax(rewards_new)
            readable_state = self.env.state2readable(states[idx_new_max])
            if not self.replay["state"].isin([readable_state]).any():
                self.replay.iloc[self.replay.reward.argmin()] = {
                    "state": self.env.state2readable(states[idx_new_max]),
                    "traj": self.env.traj2readable(trajs[idx_new_max]),
                    "reward": rewards[idx_new_max],
                    "energy": energies[idx_new_max],
                    "iter": it,
                }
                rewards_old = self.replay["reward"].values
            rewards_new[idx_new_max] = -1
        return self.replay

    def make_data_set(self, config):
        """
        Constructs a data set as a DataFrame according to the configuration.
        """
        stats = None
        if config is None:
            return None, None, None
        elif "path" in config and config.path is not None:
            path = self.logger.logdir / Path("data") / config.path
            df = pd.read_csv(path, index_col=0)
            samples = [self.env.readable2state(s) for s in df["samples"].values]
            stats = self.compute_stats(df)
        elif "type" not in config:
            return None, None, None
        elif config.type == "all" and hasattr(self.env, "get_all_terminating_states"):
            samples = self.env.get_all_terminating_states()
        elif (
            config.type == "grid"
            and "n" in config
            and hasattr(self.env, "get_grid_terminating_states")
        ):
            samples = self.env.get_grid_terminating_states(config.n)
        elif (
            config.type == "uniform"
            and "n" in config
            and hasattr(self.env, "get_uniform_terminating_states")
        ):
            samples = self.env.get_uniform_terminating_states(config.n)
        else:
            return None, None, None
        energies = self.env.proxy(self.env.statebatch2proxy(samples)).tolist()
        df = pd.DataFrame(
            {
                "samples": [self.env.state2readable(s) for s in samples],
                "energies": energies,
            }
        )
        if stats is None:
            stats = self.compute_stats(df)
        return df, {"x": samples, "energy": energies}, stats

    def compute_stats(self, data):
        mean_data = data["energies"].mean()
        std_data = data["energies"].std()
        min_data = data["energies"].min()
        max_data = data["energies"].max()
        data_zscores = (data["energies"] - mean_data) / std_data
        max_norm_data = data_zscores.max()
        return mean_data, std_data, min_data, max_data, max_norm_data
