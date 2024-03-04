"""
Buffer class to handle train and test data sets, reply buffer, etc.
"""

import pickle
from typing import List

import numpy as np
import pandas as pd
import torch


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
        self.replay_states = {}
        self.replay_trajs = {}
        self.replay_rewards = {}
        self.replay_pkl = "replay.pkl"
        self.save_replay()
        # Define train and test data sets
        if train is not None and "type" in train:
            self.train_type = train.type
        else:
            self.train_type = None
        self.train, dict_tr = self.make_data_set(train)
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
        self.test, dict_tt = self.make_data_set(test)
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
            (
                self.mean_tr,
                self.std_tr,
                self.min_tr,
                self.max_tr,
                self.max_norm_tr,
            ) = self.compute_stats(self.train)
        if self.test is not None:
            self.mean_tt, self.std_tt, self.min_tt, self.max_tt, _ = self.compute_stats(
                self.test
            )

    def save_replay(self):
        with open(self.replay_pkl, "wb") as f:
            pickle.dump(
                {
                    "x": self.replay_states,
                    "trajs": self.replay_trajs,
                    "rewards": self.replay_rewards,
                },
                f,
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
        allow_duplicate_states=False,
    ):
        for idx, (state, traj, reward, energy) in enumerate(
            zip(states, trajs, rewards, energies)
        ):
            if not allow_duplicate_states:
                if isinstance(state, torch.Tensor):
                    is_duplicate = False
                    for replay_state in self.replay_states.values():
                        if torch.allclose(state, replay_state, equal_nan=True):
                            is_duplicate = True
                            break
                else:
                    is_duplicate = state in self.replay_states.values()
                if is_duplicate:
                    continue
            if (
                reward > self.replay.iloc[-1]["reward"]
                and traj not in self.replay_trajs.values()
            ):
                self.replay.iloc[-1] = {
                    "state": self.env.state2readable(state),
                    "traj": self.env.traj2readable(traj),
                    "reward": reward,
                    "energy": energy,
                    "iter": it,
                }
                self.replay_states[(idx, it)] = state
                self.replay_trajs[(idx, it)] = traj
                self.replay_rewards[(idx, it)] = reward
                self.replay.sort_values(by="reward", ascending=False, inplace=True)
        self.save_replay()
        return self.replay

    def make_data_set(self, config):
        """
        Constructs a data set as a DataFrame according to the configuration.
        """
        if config is None:
            return None, None
        print("\nConstructing data set ", end="")
        if "type" not in config:
            return None, None
        elif config.type == "pkl" and "path" in config:
            print(f"from pickled file: {config.path}\n")
            with open(config.path, "rb") as f:
                data_dict = pickle.load(f)
                samples = data_dict["x"]
                n_samples_orig = len(samples)
                print(f"The data set containts {n_samples_orig} samples", end="")
                samples = self.env.process_data_set(samples)
                n_samples_new = len(samples)
                if n_samples_new != n_samples_orig:
                    print(
                        f", but only {n_samples_new} are valid according to the "
                        "environment settings. Invalid samples have been discarded."
                    )
                print("Remember to write a function to normalise the data in code")
                print("Max number of elements in data set has to match config")
                print("Actually, write a function that contrasts the stats")
        elif config.type == "csv" and "path" in config:
            print(f"from CSV: {config.path}\n")
            df = pd.read_csv(config.path, index_col=0)
            samples = df.iloc[:, :-1].values
        elif config.type == "all" and hasattr(self.env, "get_all_terminating_states"):
            samples = self.env.get_all_terminating_states()
        elif (
            config.type == "grid"
            and "n" in config
            and hasattr(self.env, "get_grid_terminating_states")
        ):
            print(f"by sampling a grid of {config.n} points\n")
            samples = self.env.get_grid_terminating_states(config.n)
        elif (
            config.type == "uniform"
            and "n" in config
            and "seed" in config
            and hasattr(self.env, "get_uniform_terminating_states")
        ):
            print(f"by sampling {config.n} points uniformly\n")
            samples = self.env.get_uniform_terminating_states(config.n, config.seed)
        elif (
            config.type == "random"
            and "n" in config
            and hasattr(self.env, "get_random_terminating_states")
        ):
            print(f"by sampling {config.n} points randomly\n")
            samples = self.env.get_random_terminating_states(config.n)
        else:
            return None, None
        energies = self.env.proxy(self.env.states2proxy(samples)).tolist()
        df = pd.DataFrame(
            {
                "samples": [self.env.state2readable(s) for s in samples],
                "energies": energies,
            }
        )
        return df, {"x": samples, "energy": energies}

    @staticmethod
    def compute_stats(data):
        mean_data = data["energies"].mean()
        std_data = data["energies"].std()
        min_data = data["energies"].min()
        max_data = data["energies"].max()
        data_zscores = (data["energies"] - mean_data) / std_data
        max_norm_data = data_zscores.max()
        return mean_data, std_data, min_data, max_data, max_norm_data

    @staticmethod
    def select(
        data_dict: dict,
        n: int,
        mode: str = "permutation",
        rng: np.random.Generator = None,
    ) -> List:
        """
        Selects a subset of n data points from data_dict, according to the criterion
        indicated by mode.

        The data dict may be a training set or a replay buffer.

        The mode argument can be one of the following:
            - permutation: data points are sampled uniformly from the dictionary, using
              the random generator rng.
            - weighted: data points are sampled with probability proportional to their
              score.

        Args
        ----
        data_dict : dict
            A dictionary with samples (key "x") and scores (key "energy" or "rewards").

        n : int
            The number of samples to select from the dictionary.

        mode : str
            Sampling mode. Options: permutation, weighted.

        rng : np.random.Generator
            A numpy random number generator, used for the permutation mode. Ignored
            otherwise.

        Returns
        -------
        list
            A batch of n samples, selected from data_dict.
        """
        if n == 0:
            return []
        samples = data_dict["x"]
        # If the data_dict comes from the replay buffer, then samples is a dict and we
        # need to keep its values only
        if isinstance(samples, dict):
            samples = list(samples.values())
        if mode == "permutation":
            assert rng is not None
            samples = [samples[idx] for idx in rng.permutation(n)]
        elif mode == "weighted":
            if "rewards" in data_dict:
                score = "rewards"
            elif "energy" in data_dict:
                score = "energy"
            else:
                raise ValueError(f"Data set does not contain reward or energy key.")
            scores = data_dict[score]
            # If the data_dict comes from the replay buffer, then scores is a dict and we
            # need to keep its values only
            if isinstance(scores, dict):
                scores = np.fromiter(scores.values(), dtype=float)
            indices = np.random.choice(
                len(samples),
                size=n,
                replace=False,
                p=scores / scores.sum(),
            )
            samples = [samples[idx] for idx in indices]
        else:
            raise ValueError(f"Unrecognized sampling mode: {mode}.")
        return samples
