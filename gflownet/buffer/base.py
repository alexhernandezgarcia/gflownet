"""
Base Buffer class to handle train and test data sets, reply buffer, etc.
"""

import ast
import pickle
from pathlib import Path, PosixPath
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from gflownet.utils.common import tfloat

# tests for init
#  - check replay for inti: it is created, capasity is correct, dumped file is the same as self.replay?
# implement comditional add / select
# - change allclose in adding to replay buffer
# test it
# docstrings
# adjust gflownet.py to use the new buffer
#  - (especially new select method)
#  - replay_csv instead of replay_pkl


class BaseBuffer:
    """
    Implements the functionality to manage various buffers of data: the records of
    training samples, the train and test data sets, a replay buffer for training, etc.

    Parameters
    ----------
    datadir : str or PosixPath
        The directory where the data sets and buffers are stored. By default, it is
        ./data/ but it is first set by the logger and passed as an argument to the
        Buffer for consistency, especially to handle resumed runs.
    replay_buffer : str or PosixPath
        A path to a file containing a replay buffer. If provided, the initial replay
        buffer will be loaded from this file. This is useful for for resuming runs. By
        default it is None, which initializes an empty buffer and creates a new file.
    """

    def __init__(
        self,
        env,
        proxy,
        datadir: Union[str, PosixPath],
        replay_buffer: Union[str, PosixPath] = None,
        replay_capacity=0,
        train=None,
        test=None,
        conditions=None,
        use_main_buffer=False,
        **kwargs,
    ):
        """
        TODO: write proper docstring
        if env conditional replay_capacity means capacity per condition
        """
        self.datadir = datadir
        self.env = env
        self.proxy = proxy
        self.replay_capacity = replay_capacity
        self.train_config = train
        self.test_config = test
        self.conditions_config = conditions
        self.use_main_buffer = use_main_buffer
        if use_main_buffer:
            # TODO: adapt for conditional?
            self.main = pd.DataFrame(columns=["state", "traj", "reward", "iter"])

        self.init_replay(replay_buffer)

        # self.test_csv = None
        # self.test_pkl = None

        # Load conditions sets:
        # should be one set containing a labels whether this is train or test condition
        self.conditions = self.load_conditions(conditions)

        if env.conditional and (self.conditions is None):
            raise ValueError(
                "Condition datasets are required for conditional environments. "
                "Please provide conditions in the buffer configuration."
            )

        train_condition_ids = None
        test_condition_ids = None
        if env.conditional:
            self.env.setup_conditions(self.conditions)
            train_condition_ids = self.conditions.get_condition_id(label="train")
            # TODO: add option to use only test labels for test data set
            test_condition_ids = self.conditions.get_condition_id(label="all")

        self.proxy.setup(env)

        # Define train data set
        self.train, dict_tr = self.make_data_set(
            self.train_config,
            conditions=self.conditions,
            condition_ids=train_condition_ids,
        )
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
        self.test, dict_tr = self.make_data_set(
            test, conditions=self.conditions, condition_ids=test_condition_ids
        )
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

        Parameters
        ----------
        replay_buffer : str or PosixPath
            A path to a file containing a replay buffer. If provided, the initial
            replay buffer will be loaded from this file. This is useful for for
            resuming runs. By default it is None, which initializes an empty buffer and
            creates a new file.
        """
        if replay_buffer_path is not None:
            replay_buffer_path = Path(replay_buffer_path)
        if replay_buffer_path and replay_buffer_path.exists():
            self.replay_csv = replay_buffer_path
            self.replay = self.load_replay_from_path(self.replay_csv)
        else:
            if replay_buffer_path:
                print(f"Replay buffer file {replay_buffer_path} does not exist, initializing empty buffer.")
            self.replay = pd.DataFrame(
                columns=[
                    "state",
                    "traj",
                    "reward",
                    "condition_id",
                    "iter",
                    "state_readable",
                    "traj_readable",
                ],
            )
            self.replay_csv = self.datadir / "replay.csv"
            self.save_replay()

    @property
    def replay_states(self):
        return self.replay.state.values

    @property
    def replay_trajs(self):
        return self.replay.traj.values

    @property
    def replay_rewards(self):
        return self.replay.reward.values

    @property
    def replay_condition_ids(self):
        return self.replay.condition_id.values

    def save_replay(self):
        self.replay.to_csv(self.replay_csv)

    def load_replay_from_path(self, path=None):
        # TODO: there might be environment specific issues with loading csv correctly, test with other envs
        if path is None:
            path = self.replay_csv

        # to load lists with inf values correctly
        converter = lambda x: ast.literal_eval(x.replace("inf", "2e308"))

        replay = pd.read_csv(
            path, index_col=0, converters={"state": converter, "traj": converter}
        )
        replay.replace(float("nan"), None, inplace=True)
        return replay

    def add(
        self,
        states,
        trajs,
        rewards,
        it,
        condition_ids=None,
        buffer="main",
        criterion="greater",
    ):
        """
        Adds a batch of states (with the trajectory actions and rewards) to the buffer.

        Note that the rewards may be log-rewards.

        Parameters
        ----------
        states : list
            A batch of terminating states.
        trajs : list
            The list of trajectory actions of each terminating state.
        rewards : list
            The reward or log-reward of each terminating state.
        it : int
            Iteration number.
        buffer : str
            Identifier of the buffer: main or replay
        criterion : str
            Identifier of the criterion. Currently, only greater is implemented.
        """
        if self.env.conditional and condition_ids is None:
            raise ValueError(
                "Condition ids are required for conditional environments. "
                "Please provide condition_ids in the buffer.add call."
            )
        if condition_ids is None:
            condition_ids = [None] * len(states)
        # TODO: adapt main for conditional?
        if buffer == "main":
            self.main = pd.concat(
                [
                    self.main,
                    pd.DataFrame(
                        {
                            "state": [self.env.state2readable(s) for s in states],
                            "traj": [self.env.traj2readable(p) for p in trajs],
                            "reward": rewards,
                            "iter": it,
                        }
                    ),
                ],
                axis=0,
                join="outer",
            )
        elif buffer == "replay":
            if self.replay_capacity > 0:
                if criterion == "greater":
                    self.replay = self._add_greater(
                        states, trajs, rewards, condition_ids, it
                    )
                else:
                    raise ValueError(
                        f"Unknown criterion identifier. Received {buffer}, "
                        "expected greater"
                    )
        else:
            raise ValueError(
                f"Unknown buffer identifier. Received {buffer}, expected main or replay"
            )

    def _add_greater(self, states, trajs, rewards, condition_ids, it):
        """
        TODO: update docstring
        Adds a batch of states (with the trajectory actions and rewards) to the buffer
        if the state reward is larger than the minimum reward in the buffer and the
        trajectory is not yet in the buffer.

        Note that the rewards may be log-rewards. The reward is only used to check the
        inclusion criterion. Since the logarithm is a monotonic function, using the log
        or natural rewards is equivalent for this purpose.

        Parameters
        ----------
        states : list
            A batch of terminating states.
        trajs : list
            The list of trajectory actions of each terminating state.
        rewards : list
            The reward or log-reward of each terminating state.
        it : int
            Iteration number.
        """
        if torch.is_tensor(rewards):
            rewards = rewards.tolist()
        for state, traj, reward, condition_id in zip(
            states, trajs, rewards, condition_ids
        ):
            self._add_greater_single_state(state, traj, reward, it, condition_id)
        self.save_replay()
        return self.replay

    def _add_greater_single_state(self, state, traj, reward, it, condition_id=None):
        if condition_id is not None:
            # TODO: I (AHG) am adding .item() because a device error when resuming a
            # run, but this could potentially add time overhead and there might be a
            # better solution.
            relevant_replay = self.replay[
                self.replay["condition_id"] == condition_id.item()
            ]
        else:
            relevant_replay = self.replay

        for rstate in relevant_replay["state"]:
            # TODO: add this method to base env and htorus
            if hasattr(self.env, "states_are_close"):
                # TODO: add this method to base env and htorus
                # TODO (AHG): or just use isclose
                if self.env.states_are_close(state, rstate):
                    return
            else:
                if self.env.isclose(state, rstate):
                    return

        if len(relevant_replay) < self.replay_capacity:
            self._concat_item_to_replay(state, traj, reward, it, condition_id)
            return

        argmin = relevant_replay["reward"].argmin()
        index_argmin = relevant_replay.index[argmin]
        if reward > relevant_replay["reward"].loc[index_argmin]:
            self.replay.drop(self.replay.index[index_argmin], inplace=True)
            self._concat_item_to_replay(state, traj, reward, it, condition_id)

    def _concat_item_to_replay(self, state, traj, reward, it, condition_id=None):
        if torch.is_tensor(state):
            state = state.tolist()
        if torch.is_tensor(traj):
            traj = traj.tolist()
        if torch.is_tensor(reward):
            reward = reward.item()
        if torch.is_tensor(condition_id):
            condition_id = condition_id.item()
        self.replay = pd.concat(
            [
                self.replay,
                pd.DataFrame(
                    {
                        "state": [state],
                        "traj": [traj],
                        "reward": [reward],
                        "condition_id": [condition_id],
                        "iter": [it],
                        "state_readable": [self.env.state2readable(state)],
                        "traj_readable": [self.env.traj2readable(traj)],
                    }
                ),
            ],
            ignore_index=True,
        )

    def load_conditions(self, config):
        """
        Loads conditions data set from a file.
        """
        if config is None:
            return None
        if "path" not in config:
            return None
        path = Path(config.path)
        if not path.exists():
            print("\nThe conditions path does not exist: ", path)
            return None
        if hasattr(self.env, "load_conditions"):
            print("\nLoading conditions set from file: ", path)
            return self.env.load_conditions(path)

    def make_data_set_with_conditions(self, config, conditions, condition_ids=None):
        # downloading data set from file if path provided
        if "path" in config and config.type in ["pkl", "csv"]:
            if config.type == "pkl":
                print(f"from pickled file: {config.path}\n")
                with open(config.path, "rb") as f:
                    data_dict = pickle.load(f)
                    samples = data_dict["x"]
                    if "condition_id" not in data_dict:
                        raise ValueError(
                            "Condition_id is not provided in the data set."
                        )
                    condition_ids = data_dict["condition_id"]
            elif config.type == "csv":
                print(f"from CSV: {config.path}\n")
                df = pd.read_csv(
                    config.path,
                    index_col=0,
                    converters={config.samples_column: ast.literal_eval},
                )
                samples = df[config.samples_column].values
                if config.condition_ids_column not in df.columns:
                    raise ValueError("Condition_id is not provided in the data set.")
                condition_ids = df[config.condition_ids_column].values
            if hasattr(self.env, "process_data_set"):
                n_samples_orig = len(samples)
                print(f"The data set containts {n_samples_orig} samples", end="")
                samples, condition_ids = self.env.process_data_set(
                    samples, condition_ids, conditions
                )
                n_samples_new = len(samples)
                if n_samples_new != n_samples_orig:
                    print(
                        f", but only {n_samples_new} are valid according to the "
                        "environment settings. Invalid samples have been discarded."
                    )
        # various ways to sample dataset if path is not provided
        elif (
            config.type == "grid"
            and "n" in config
            and hasattr(self.env, "get_grid_terminating_states")
        ):
            print(f"by sampling a grid points for each condition\n")
            samples, condition_ids = self.env.get_grid_terminating_states(
                config.n, conditions, condition_ids=condition_ids
            )
        elif (
            config.type == "uniform"
            and "n" in config
            and hasattr(self.env, "get_uniform_terminating_states")
        ):
            print(f"by sampling points uniformly for each condition\n")
            seed = config.seed if "seed" in config else None
            samples, condition_ids = self.env.get_uniform_terminating_states(
                config.n,
                conditions,
                condition_ids=condition_ids,
                seed=seed,
            )
        else:
            print("Dataset type {config.type} is not supported for conditional case")
            return None, None
        # reward amd energy mb check the other file?
        states_proxy = self.env.states2proxy(
            samples, condition_ids=condition_ids, conditions=conditions
        )

        rewards, energies = self.proxy.rewards(
            states_proxy, log=False, return_proxy=True
        )
        rewards = rewards.tolist()
        energies = energies.tolist()

        df = pd.DataFrame(
            {
                "samples_readable": [self.env.state2readable(s) for s in samples],
                "energies": energies,
                "samples": samples,
                "rewards": rewards,
                "condition_id": condition_ids,
            }
        )
        return df, {
            "x": samples,
            "energy": energies,
            "reward": rewards,
            "condition_id": condition_ids,
        }

    def make_data_set(self, config, conditions=None, condition_ids=None):
        """
        Constructs a data set as a DataFrame according to the configuration.
        """

        if config is None:
            return None, None

        if "type" not in config:
            return None, None

        print("\nConstructing data set ", end="")
        if conditions is not None:
            return self.make_data_set_with_conditions(
                config, conditions, condition_ids=condition_ids
            )

        if config.type == "pkl" and "path" in config:
            # TODO: clean up this mess, avoid recompputing energy from load from path
            print(f"from pickled file: {config.path}\n")
            with open(config.path, "rb") as f:
                data_dict = pickle.load(f)
                samples = data_dict["x"]
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
            df = pd.read_csv(
                config.path,
                index_col=0,
                converters={config.samples_column: ast.literal_eval},
            )
            samples = df[config.samples_column].values
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
        rewards, energies = self.proxy.rewards(
            self.env.states2proxy(samples), log=False, return_proxy=True
        )
        rewards = rewards.tolist()
        energies = energies.tolist()
        df = pd.DataFrame(
            {
                "samples_readable": [self.env.state2readable(s) for s in samples],
                "energies": energies,
                "samples": samples,
                "rewards": rewards,
            }
        )
        return df, {"x": samples, "energy": energies, "reward": rewards}

    @staticmethod
    def compute_stats(data):
        # TODO: ideally, should be adapted for conditional case,
        # but it is never used outside buffer, ignoring for now
        mean_data = data["energies"].mean()
        std_data = data["energies"].std()
        min_data = data["energies"].min()
        max_data = data["energies"].max()
        data_zscores = (data["energies"] - mean_data) / std_data
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
            (key "energy" or "rewards").

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

        index = df.index.tolist()
        # Identify the indices of the samples to select
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
            for name in ["rewards", "reward", "energy", "energies"]:
                if name in df:
                    score = name
                    break
            if score is None:
                raise ValueError(
                    f"Data set does not contain reward(s) or energy(ies) key, cannot sample in weighted mode"
                )
            scores = df[score].values

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
