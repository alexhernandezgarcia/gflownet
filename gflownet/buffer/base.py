"""
Base Buffer class to handle train and test data sets, reply buffer, etc.
"""

import ast
import pickle
from pathlib import Path, PosixPath
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torchtyping import TensorType

from gflownet.utils.common import tfloat


class BaseBuffer:
    """
    Implements the functionality to manage various buffers of data: the records of
    training samples, the train and test data sets, a replay buffer for training, etc.

    Attributes
    ----------
    replay_updated : bool
        Whether the replay buffer was updated in the last iteration, that is whether at
        least one sample was added to the buffer.
    """

    def __init__(
        self,
        env,
        proxy,
        datadir: Union[str, PosixPath],
        replay_buffer: Union[str, PosixPath] = None,
        replay_capacity: int = 0,
        train: Dict = None,
        test: Dict = None,
        use_main_buffer=False,
        check_diversity: bool = False,
        **kwargs,
    ):
        """
        Initializes the Buffer.

        Parameters
        ----------
        datadir : str or PosixPath
            The directory where the data sets and buffers are stored. By default, it is
            ./data/ but it is first set by the logger and passed as an argument to the
            Buffer for consistency, especially to handle resumed runs.
        replay_buffer : str or PosixPath
            A path to a file containing a replay buffer. If provided, the initial
            replay buffer will be loaded from this file. This is useful for for
            resuming runs. By default it is None, which initializes an empty buffer and
            creates a new file.
        replay_capacity : int
            Size of the replay buffer. By default, it is zero, thus no replay buffer is
            used.
        train : dict
            A dictionary describing the training data. The dictionary can have the
            following keys:
                - type : str
                    Type of data. It can be one of the following:
                        - pkl: a pickled file. Requires path.
                        - csv: a CSV file. Requires path.
                        - all: all terminating states of the environment.
                        - grid: a grid of terminating states. Requires n.
                        - uniform: terminating states uniformly sampled. Requires n.
                        - random: terminating states sampled randomly from the intial
                          GFN policy. Requires n.
                - path : str
                    Path to a CSV of pickled file (for type={pkl, csv})
                - n : int
                    Number of samples (for type={grid, uniform, random})
                - seed : int
                    Seed for random sampling (for type={uniform, random})
        test : dict
            A dictionary describing the test data. The dictionary is akin the train
            dictionarity.
        use_main_buffer : bool
            If True, a main buffer is kept up to date, that is all training samples are
            added to a buffer. It is False by default because of the potentially large
            memory usage it can incur.
        check_diversity : bool
            If True, new samples are only added to the buffer if they are not close to
            any of the samples already present in the buffer. env.isclose() is used
            for the comparison. It is False by default because this comparison can
            easily take most of the running time with an uncertain impact on the
            performance. The implementation should be improved to make this functional.
        """
        self.datadir = datadir
        self.env = env
        self.proxy = proxy
        self.replay_capacity = replay_capacity
        self.train_config = self._process_data_config(train)
        self.test_config = self._process_data_config(test)
        self.use_main_buffer = use_main_buffer
        self.check_diversity = check_diversity
        if self.use_main_buffer:
            self.main = pd.DataFrame(
                columns=["samples", "trajectories", "rewards", "iter"]
            )
        else:
            self.main = None
        self.replay, self.replay_csv = self.init_replay(replay_buffer)
        self.replay_updated = False
        self.save_replay()

        # Setup proxy
        self.proxy.setup(env)

        # Define train data set
        self.train, dict_tr = self.make_data_set(self.train_config)
        if self.train is None:
            print(
                "\tImportant: offline trajectories will NOT be sampled. In order to "
                " sample offline trajectories, the train configuration of the buffer "
                " should be provided."
            )
        # Save train.csv and train.pkl
        # TODO: implement flag to prevent storing data set for large data sets. Store
        # path instead.
        if self.train is not None:
            self.train.to_csv(self.datadir / "train.csv")
        if dict_tr is not None:
            self.train_config.pkl = self.datadir / "train.pkl"
            with open(self.train_config.pkl, "wb") as f:
                pickle.dump(dict_tr, f)

        # Define test data set
        self.test, dict_tr = self.make_data_set(self.test_config)
        if self.test is None:
            print(
                "\tImportant: test metrics will NOT be computed. In order to compute "
                "test metrics, the test configuration of the buffer should be "
                "provided."
            )
        # Save test.csv and test.pkl
        # TODO: implement flag to prevent storing data set for large data sets. Store
        # path instead.
        if self.test is not None:
            self.test.to_csv(self.datadir / "test.csv")
        if dict_tr is not None:
            self.test_config.pkl = self.datadir / "test.pkl"
            with open(self.test_config.pkl, "wb") as f:
                pickle.dump(dict_tr, f)

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

    def init_replay(self, replay_buffer_path: Union[str, PosixPath] = None):
        """
        Initializes the replay buffer.

        If a path to an existing replay buffer file is provided, then the replay buffer
        is initialized from it. Otherwise, a new empty buffer is created.

        Parameters
        ----------
        replay_buffer : str or PosixPath
            A path to a file containing a replay buffer. If provided, the initial
            replay buffer will be loaded from this file. This is useful for for
            resuming runs. By default it is None, which initializes an empty buffer and
            creates a new file.

        Returns
        -------
        replay : pandas.DataFrame
            DataFrame with the initial replay buffer.
        replay_csv : PosixPath
            Path of the CSV that will store the replay buffer.
        """
        if replay_buffer_path is not None:
            replay_buffer_path = Path(replay_buffer_path)
        if replay_buffer_path and replay_buffer_path.exists():
            replay_csv = replay_buffer_path
            replay = self.load_replay_from_path(replay_csv)
        else:
            if replay_buffer_path:
                print(
                    f"Replay buffer file {replay_buffer_path} does not exist, "
                    "initializing empty buffer."
                )
            replay = pd.DataFrame(
                columns=[
                    "samples",
                    "trajectories",
                    "rewards",
                    "iter",
                    "samples_readable",
                    "trajectories_readable",
                ],
            )
            replay_csv = self.datadir / "replay.csv"

        return replay, replay_csv

    @property
    def replay_samples(self):
        return self.replay.samples.values

    @property
    def replay_trajectories(self):
        return self.replay.trajectories.values

    @property
    def replay_rewards(self):
        return self.replay.reward.values

    def save_replay(self):
        self.replay.to_csv(self.replay_csv)

    # TODO: there might be environment specific issues with loading csv correctly.
    # TODO: test with other envs
    # TODO: alternatively, the replay buffer could be stored as part of the checkpoint
    # and there is no need to save it every iteration
    def load_replay_from_path(self, path: PosixPath = None):
        """
        Loads a replay buffer stored as a CSV file.
        """
        if path is None:
            path = self.replay_csv

        # to load lists with inf values correctly
        converter = lambda x: ast.literal_eval(x.replace("inf", "2e308"))

        replay = pd.read_csv(
            path,
            index_col=0,
            converters={"samples": converter, "trajectories": converter},
        )
        replay.replace(float("nan"), None, inplace=True)
        return replay

    def add(
        self,
        samples,
        trajectories,
        rewards: Union[List, TensorType],
        it,
        buffer="main",
        criterion="greater",
    ):
        """
        Adds a batch of samples (with the trajectory actions and rewards) to the buffer.

        Parameters
        ----------
        samples : list
            A batch of terminating states.
        trajectories : list
            The list of trajectory actions of each terminating state.
        rewards : list or tensor
            The reward of each terminating state.
        it : int
            Iteration number.
        buffer : str
            Identifier of the buffer: main or replay
        criterion : str
            Identifier of the criterion. Currently, only greater is implemented.
        """
        if torch.is_tensor(rewards):
            rewards = rewards.tolist()

        if buffer == "main":
            self.main = pd.concat(
                [
                    self.main,
                    pd.DataFrame(
                        {
                            "samples": [self.env.state2readable(s) for s in samples],
                            "trajectories": [
                                self.env.traj2readable(t) for t in trajectories
                            ],
                            "rewards": rewards,
                            "iter": it,
                        }
                    ),
                ],
                axis=0,
                join="outer",
            )
        elif buffer == "replay":
            if self.replay_capacity > 0:
                self.replay_updated = False
                if criterion == "greater":
                    self.replay = self._add_greater(samples, trajectories, rewards, it)
                else:
                    raise ValueError(
                        f"Unknown criterion identifier. Received {buffer}, "
                        "expected greater"
                    )
        else:
            raise ValueError(
                f"Unknown buffer identifier. Received {buffer}, expected main or replay"
            )

    # TODO: update docstring
    # TODO: consider not saving the replay buffer every iteration. Instead it could be
    # save alongside the model checkpoints.
    def _add_greater(
        self,
        samples: List,
        trajectories: List,
        rewards: List,
        it: int,
    ):
        """
        Adds a batch of samples (with the trajectory actions and rewards) to the buffer
        if the state reward is larger than the minimum reward in the buffer and the
        trajectory is not yet in the buffer.

        Parameters
        ----------
        samples : list
            A batch of terminating states.
        trajectories : list
            The list of trajectory actions of each terminating state.
        rewards : list
            The reward of each terminating state.
        it : int
            Iteration number.

        Returns
        -------
        self.replay : The updated replay buffer
        """
        for sample, traj, reward in zip(samples, trajectories, rewards):
            self._add_greater_single_sample(sample, traj, reward, it)
        self.save_replay()
        return self.replay

    # TODO: there may be issues with certain state types
    # TODO: add parameter(s) to control isclose()
    def _add_greater_single_sample(self, sample, trajectory, reward, it):
        """
        Adds a single sample (with the trajectory actions and reward) to the buffer
        if the state reward is larger than the minimum reward in the buffer and the
        trajectory is not yet in the buffer.

        If the sample is similar to any sample already present in the buffer, then the
        sample will not be added.

        Parameters
        ----------
        samples : list, tensor, array, dict
            A terminating state.
        trajectory : list
            A list of trajectory actions of leading to the terminating state.
        reward : float
            The reward of the terminating state.
        it : int
            Iteration number.

        Returns
        -------
        self.replay : The updated replay buffer
        """
        # If the buffer is not full, sample can be added to the buffer
        if len(self.replay) < self.replay_capacity:
            can_add = True
            index_min = -1
        else:
            can_add = False

        # If the buffer is full (can_add is False), check if the reward is larger than
        # the minimum reward in the buffer. If so, set can_add to True, which may
        # result in dropping the sample with the minimum reward. Otherwise, return.
        if not can_add:
            index_min = self.replay.index[self.replay["rewards"].argmin()]
            if reward > self.replay["rewards"].loc[index_min]:
                can_add = True
            else:
                return

        # Return without adding if the sample is close to any sample already present in
        # the buffer
        # TODO: this could be optimized by comparing only with samples with similar
        # reward
        if self.check_diversity:
            for rsample in self.replay["samples"]:
                if self.env.isclose(sample, rsample):
                    return

        # If index_min is larger than zero, drop the sample with the minimum reward
        if index_min >= 0:
            self.replay.drop(self.replay.index[index_min], inplace=True)

        # Add the sample to the buffer
        self.replay_updated = True
        if torch.is_tensor(sample):
            sample = sample.tolist()
        if torch.is_tensor(trajectory):
            trajectory = trajectory.tolist()
        if torch.is_tensor(reward):
            reward = reward.item()
        self.replay = pd.concat(
            [
                self.replay,
                pd.DataFrame(
                    {
                        "samples": [sample],
                        "trajectories": [trajectory],
                        "rewards": [reward],
                        "iter": [it],
                        "samples_readable": [self.env.state2readable(sample)],
                        "trajectories_readable": [self.env.traj2readable(trajectory)],
                    }
                ),
            ],
            ignore_index=True,
        )

    def make_data_set(self, config):
        """
        Constructs a data set as a DataFrame according to the configuration.
        """

        if config is None:
            return None, None

        if "type" not in config:
            return None, None

        print("\nConstructing data set ", end="")

        if config.type == "pkl" and "path" in config:
            # TODO: clean up this mess, avoid recomputing scores from load from path
            print(f"from pickled file: {config.path}\n")
            with open(config.path, "rb") as f:
                data_dict = pickle.load(f)
                samples = data_dict["samples"]
                if hasattr(self.env, "process_data_set"):
                    n_samples_orig = len(samples)
                    print(f"The data set containts {n_samples_orig} samples", end="")
                    samples = self.env.process_data_set(samples)
                    n_samples_new = len(samples)
                    if n_samples_new != n_samples_orig:
                        print(
                            f", but only {n_samples_new} are valid according to the "
                            "environment settings. Invalid samples have been discarded."
                        )
        elif config.type == "csv" and "path" in config:
            print(f"from CSV: {config.path}\n")
            if "samples_column" in config:
                df = pd.read_csv(
                    config.path,
                    index_col=0,
                    converters={config.samples_column: ast.literal_eval},
                )
                samples = df[config.samples_column].values
            else:
                samples = pd.read_csv(config.path, index_col=0)
            if hasattr(self.env, "process_data_set"):
                n_samples_orig = len(samples)
                print(f"The data set containts {n_samples_orig} samples", end="")
                samples = self.env.process_data_set(samples)
                n_samples_new = len(samples)
                if n_samples_new != n_samples_orig:
                    print(
                        f", but only {n_samples_new} are valid according to the "
                        "environment settings. Invalid samples have been discarded."
                    )
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
            and hasattr(self.env, "get_uniform_terminating_states")
        ):
            print(f"by sampling {config.n} points uniformly\n")
            seed = config.seed if "seed" in config else None
            samples = self.env.get_uniform_terminating_states(config.n, seed)
        elif (
            config.type == "random"
            and "n" in config
            and hasattr(self.env, "get_random_terminating_states")
        ):
            print(f"by sampling {config.n} points randomly\n")
            samples = self.env.get_random_terminating_states(config.n)
        else:
            return None, None
        rewards, scores = self.proxy.rewards(
            self.env.states2proxy(samples), log=False, return_proxy=True
        )
        rewards = rewards.tolist()
        scores = scores.tolist()
        df = pd.DataFrame(
            {
                "samples_readable": [self.env.state2readable(s) for s in samples],
                "scores": scores,
                "samples": samples,
                "rewards": rewards,
            }
        )
        return df, {"samples": samples, "scores": scores, "rewards": rewards}

    @staticmethod
    def compute_stats(data):
        mean_data = data["scores"].mean()
        std_data = data["scores"].std()
        min_data = data["scores"].min()
        max_data = data["scores"].max()
        data_zscores = (data["scores"] - mean_data) / std_data
        max_norm_data = data_zscores.max()
        return mean_data, std_data, min_data, max_data, max_norm_data

    # TODO: update docstring
    @staticmethod
    def select(
        df: pd.DataFrame,
        n: int,
        mode: str = "permutation",
        rng: Optional[np.random.Generator] = None,
    ) -> pd.DataFrame:
        """
        Selects a subset of n data points from data_dict, according to the criterion
        indicated by mode.

        The data dict may be a training set or a replay buffer.

        The mode argument can be one of the following:
            - permutation: data points are sampled uniformly from the dictionary, without
              replacement, using the random generator rng.
            - uniform: data points are sampled uniformly from the dictionary, with
              replacement, using the random generator rng.
            - weighted: data points are sampled with probability proportional to their
              score.

        Args
        ----
        data_dict : dict
            A dictionary containing data for various data samples. The keys of the
            dictionary represent the sample attributes and the values are lists
            that contain the values of these attributes for all the samples.
            All the values in the data dictionary should have the same length.
            If mode == "weighted", the data dictionary must contain sample scores
            (key "scores" or "rewards").

        n : int
            The number of samples to select from the dictionary.

        mode : str
            Sampling mode. Options: permutation, weighted.

        rng : np.random.Generator
            A numpy random number generator, used for the permutation mode. Ignored
            otherwise.

        Returns
        -------
        filtered_data_dict
            A dict containing the data of n samples, selected from data_dict.
        """
        # If random number generator is None, start a new one
        if rng is None:
            rng = np.random.default_rng()

        # Identify the indices of the samples to select
        index = df.index.tolist()
        if mode in ["permutation", "uniform"]:
            assert rng is not None
            with_replacement = mode == "uniform" or n >= len(index)
            selected_index = rng.choice(
                index,
                size=n,
                replace=with_replacement,
            )
        elif mode == "weighted":
            # Determine which attribute to compute the sample probabilities from
            score = None
            for name in ["rewards", "reward", "score", "scores"]:
                if name in df:
                    score = name
                    break
            if score is None:
                raise ValueError(
                    f"Data set does not contain reward(s) or score(s) key. "
                    "Cannot sample in weighted mode."
                )
            scores = df[score].values

            # Turn scores into probabilities
            if np.any(scores < 0):
                scores = softmax(scores)
            else:
                scores = scores / np.sum(scores)

            # Obtain the indices of the selected samples
            selected_index = rng.choice(
                index,
                size=n,
                replace=True,
                p=scores,
            )
        else:
            raise ValueError(f"Unrecognized sampling mode: {mode}.")

        return df.loc[selected_index]

    @staticmethod
    def _process_data_config(config: dict = None):
        if config is None:
            return None
        if all([v is None for v in config.values()]):
            return None
        else:
            return config
