"""Logger for the data used by the Visualizations.

Use the provided logger to log n on-policy samples every m iterations.
The logger saves the data in a SQLite database to be queried by the dashboard.
Note that for running the dashboard you will also need a text-to-image function
and a state aggregation function to calculate the image representation of a state.
This is documented in dashboard.py.
Logging is already integrated in the main training loop.

Functions:
    - log() stores all given data in a df.
    - write_to_db() writes the logged data to the database after logging a batch is
        done.
    - compute_graph() computes the dag from the trajectories.
        By default this is done each time write_to_db is called.
        You can disable this behavior to save resources and call compute_graph() after
        training is finished if you do not need the data while training.
    create_and_append_testset() allows you to write batches of data to the testset in
        the expected format.

Scalability:
    The complete trajectories get saved, so the resulting rowcount is
    n_samples * (total iterations / log_every_m_iterations) * average trajectory length.
    Keeping this below 1e6 should work fine.
"""

import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from .graphdbs_from_db import create_graph_dbs


class VisLogger:
    """Logger class."""

    def __init__(
        self,
        path: str | None = None,
        s0_included: bool = True,
        fn_state_to_text: callable | None = None,
        fn_compute_features: callable | None = None,
        metrics: list[str] | None = None,
        features: list[str] | None = None,
    ):
        """Logger for the visualizations.

        Parameters
        ----------
        path: str
            path to a folder to save the data. If none creates one based on datetime
            in root.
        s0_included: bool
            if True, the trajectories are expected to have the (empty)
            start state included.
            The start states will then be removed before writing to the database.
            s0 will be specified as '#' in the visualizations.
        fn_state_to_text: callable
            Function to convert a batch of states to a list of readable strings to
            identify a single state.
            Neccessary, used to distinguish states.
            Consecutive states with the same text will be merged (logprobs will be
            added).
            s0 will be specified as '#' in the visualizations,
            make sure no state has the same identifier.
        fn_compute_features: callable
            Function to compute features from the states.
            Should take the states in the same format as given (torch Tensor, np array
            or list) and return a tuple consisting of:
                1. A np array of size (sum(trajectory_lengths), n_features) containing
                the features
                2. A list(bool) that tells for each state if the feature computation was
                 successful.
                Unsuccessful states are skipped in the trajectory visualizations
            These will be used for the downprojections.
            Additional features can also be logged with the features parameter
        metrics: list[str]
            Optional.
            Names of additional metrics for final objects. Might be different losses or
            rewards.
            If the reward or loss function consists of multiple parts you can specify
            all of them here (list of strings).
            They will need to be logged each iteration.
            Otherwise only the total reward and the loss will be logged.
        features: list[str]
            Optional.
            If you want to log features, specify them here (list of strings).
            They will need to be logged each iteration.
            The features will be used for the downprojections.
            If features can be calculated from the states you can additionally use the
            fn_compute_features parameter.
        """
        if path:
            self.path = path
        else:
            self.path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(self.path)

        self.db = f"{self.path}/data.db"
        conn = sqlite3.connect(self.db)
        conn.close()
        self.metrics = metrics
        self.features = features
        self.s0_included = s0_included
        self.fn_state_to_text = fn_state_to_text
        self.fn_compute_features = fn_compute_features
        self.cols = [
            "final_id",
            "step",
            "final_object",
            "text",
            "iteration",
            "total_reward",
            "loss",
            *(metrics if metrics is not None else []),
            "logprobs_forward",
            "logprobs_backward",
            "features_valid_provided",
            *(features if features is not None else []),
        ]
        self.df = pd.DataFrame(columns=self.cols)

        # tracks the logged data
        self.current = {
            "batch_idx": None,
            "states": None,
            "total_reward": None,
            "loss": None,
            "iteration": None,
            "logprobs_forward": None,
            "logprobs_backward": None,
        }
        self.current.update({name: None for name in metrics or []})
        self.current.update({name: None for name in features or []})
        if self.features:
            self.current.update({"features_valid_provided": None})

    def attach_fns(self, fn_state_to_text=None, fn_compute_features=None):
        """Attach the functions after initialization."""
        if fn_state_to_text is not None:
            self.fn_state_to_text = fn_state_to_text
        if fn_compute_features is not None:
            self.fn_compute_features = fn_compute_features

    def log(
        self,
        batch_idx: np.ndarray | torch.Tensor | None,
        states: np.ndarray | torch.Tensor | list | None,
        total_reward: np.ndarray | torch.Tensor | None,
        loss: np.ndarray | torch.Tensor | None,
        iteration: int | None,
        logprobs_forward: np.ndarray | torch.Tensor | None,
        logprobs_backward: np.ndarray | torch.Tensor | None,
        metrics: np.ndarray | torch.Tensor | list[list[float]] | None = None,
        features_valid_provided: np.ndarray | torch.Tensor | list[bool] = None,
        features: np.ndarray | torch.Tensor | list[list[float]] | None = None,
    ):
        """
        Log everything provided for the current iteration.

        Parameters
        ----------
        batch_idx: np.ndarray | torch.Tensor
            indices of final object of the batch.
            Each final object should have one index repeating for the length of its
            trajectory. Eg.:
            [0,0,0,0,1,1,2,...] -> Object 0, trajectory length 4, Object 1,
            trajectory length 2 ...
            Trajectory data is expected to be logged from first state to last.
            In the example above the final objects would be at index 3 for the first
            Trajectory and index 5 for the second.
            Expected size: (sum(trajectory_lengths), ).
        total_reward: np.ndarray | torch.Tensor
            array or tensor of total rewards.
            Expected size: (batchsize, ).
        loss: np.ndarray | torch.Tensor
            array or tensor of losses.
            Expected size: (batchsize, ).
        iteration: int
            current iteration
        states: np.ndarray | torch.Tensor
            array or tensor of states (full trajectories).
            Expected size: (sum(trajectory_lengths), ).
        logprobs_forward: np.ndarray | torch.Tensor
            array or tensor of forward logprobabilities.
            Expected size: (sum(trajectory_lengths), ).
            The logprob of a state s is expected to be the logprob of reaching s,
            see example in lobprobs_backward.
        logprobs_backward: np.ndarray | torch.Tensor
            array or tensor of backward logprobabilities.
            Expected size: (sum(trajectory_lengths), ).
            The lobprob of a state s is expected to be the logprob of reaching s, eg.:

            state   | logprob_forward       | logprobs_backward
            --------|-----------------------|------------------
            s0      | 0                     | logprob(s1->s0)=0
            s1      | logprob(s0->s1)       | logprob(s2->s1)
            s2      | logprob(s1->s2)       | logprob(s3->s2)
            st      | logprob(s(t-1)->st)   | logprob(s(t+1)->st) = 0, not applicable

        metrics: np.ndarray | torch.Tensor | list[list[float]]
            Optional.
            Additionally logged metrics of final objects based on the initialized
            metrics. Total reward and loss are logged seperately.
            A torch Tensor or np array of shape (len(metrics), batchsize,) or
            a nested list of the same shape.
        features_valid_provided: np.ndarray | torch.Tensor | list[bool]
            boolean array of shape(sum(trajectory_lengths), ).
            Flags if the provided features are valid. Otherwise they are ignored.
        features: np.ndarray | torch.Tensor | list[list[float]]
            Optional.
            Additionally logged features based on the initialized features.
            A torch Tensor or np array of shape (len(features), sum(trajectory_lengths))
            or a nested list of the same shape.
        """
        if batch_idx is not None:
            self.current["batch_idx"] = self.__to_np__(batch_idx, np.int16)
        if states is not None:
            if isinstance(states, list):
                self.current["states"] = states
            elif isinstance(states, torch.Tensor):
                self.current["states"] = states.detach().cpu()
            else:
                self.current["states"] = self.__to_np__(states, np.float16)
        if total_reward is not None:
            self.current["total_reward"] = self.__to_np__(total_reward, np.float16)
        if loss is not None:
            self.current["loss"] = self.__to_np__(loss, np.float16)
        if iteration is not None:
            self.current["iteration"] = int(iteration)
        if logprobs_forward is not None:
            self.current["logprobs_forward"] = self.__to_np__(
                logprobs_forward, np.float16
            )
        if logprobs_backward is not None:
            self.current["logprobs_backward"] = self.__to_np__(
                logprobs_backward, np.float16
            )
        if metrics is not None:
            for i, r in enumerate(self.metrics):
                self.current[r] = metrics[i]
        if features_valid_provided is not None:
            self.current["features_valid_provided"] = self.__to_np__(
                features_valid_provided, np.bool_
            )
        if features is not None:
            for i, r in enumerate(self.features):
                self.current[r] = features[:, i]

    def __check_data__(self):
        """Check the data for completeness before writing to db."""
        for k, v in self.current.items():
            assert v is not None, f"{k} has not been logged"

        datalength = len(self.current["batch_idx"])
        for i in [
            self.current["states"],
            self.current["logprobs_forward"],
            self.current["logprobs_backward"],
        ]:
            assert (
                len(i) == datalength
            ), "lengths of batch_idx, logprobs and states must match"

    def __to_np__(self, x, dtype):
        """
        Convert to np array with given dtype.

        Parameters
        ----------
        x: np.ndarray | torch.Tensor
        dtype:
            expected dtype
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x.astype(dtype, copy=False)

    def write_to_db(self, compute_graph=True):
        """Write the data of the current block to the database.

        Should be called every m iterations.

        Parameters
        ----------
        compute_graph: bool
            whether to compute the graph when writing to the db.
            When true, the graph dbs neccessary to view the DAG gets computed.
            As this takes some time, memory and computation power,
            this makes sense only if you want to view the Graph during running training.
            When false you will need to call compute_graph() after training is done.
            If you write the samples to the db in batches instead of all at once,
            write all batches first with compute_graph = False and call
            compute_graph() afterwards.
        """
        self.__check_data__()
        data = pd.DataFrame(columns=self.cols)

        # calculate the step for each state of a trajectory and if a state is final
        _, data["final_id"] = np.unique(self.current["batch_idx"], return_inverse=True)

        change = self.current["batch_idx"][1:] != self.current["batch_idx"][:-1]
        step = np.arange(len(self.current["batch_idx"])) - np.maximum.accumulate(
            np.r_[0, np.flatnonzero(change) + 1].repeat(
                np.diff(
                    np.r_[0, np.flatnonzero(change) + 1, len(self.current["batch_idx"])]
                )
            )
        )
        last = np.r_[step[1:] == 0, True]
        data["step"] = step
        data["final_object"] = last

        # expand scalar iteration and add iteration and logprobs
        datalength = len(self.current["batch_idx"])
        data["iteration"] = np.array([self.current["iteration"]] * datalength)
        data["logprobs_forward"] = self.current["logprobs_forward"]
        data["logprobs_backward"] = self.current["logprobs_backward"]

        # expand arrays of length batchsize to db size
        # (total_reward, loss, provided additional metrics)
        nan_prefill = np.full(len(self.current["batch_idx"]), np.nan)
        metric_list = ["total_reward", "loss"]
        if self.metrics is not None:
            metric_list += [i for i in self.metrics]
        for m in metric_list:
            ar = nan_prefill.copy()
            ar[last] = self.current[m]
            data[m] = ar

        # add provided features
        if self.features is not None:
            data["features_valid_provided"] = self.current["features_valid_provided"]
            for i in self.features:
                data[i] = self.current[i]

        # compute texts
        data["text"] = self.fn_state_to_text(self.current["states"])

        # add states temporarily to allow feature computation only for neccessary ones
        data["states"] = self.current["states"]

        # delete s0 or update step to start with 1
        if self.s0_included:
            data = data[data["step"] != 0]
        else:
            data["step"] += 1

        # collapse consecutive identical texts of each trajectory
        # (take last row and sum logprobs)
        data = data.sort_values(["final_id", "text", "step"])
        logprob_sums = data.groupby(["final_id", "text"], as_index=False)[
            ["logprobs_forward", "logprobs_backward"]
        ].sum()
        last_rows = (
            data.groupby(["final_id", "text"], as_index=False)
            .last()
            .drop(columns=["logprobs_forward", "logprobs_backward"])
        )
        data = last_rows.merge(logprob_sums, on=["final_id", "text"], how="left")

        # compute features
        feature_cols = None
        if self.fn_compute_features is not None:
            features, features_valid = self.fn_compute_features(list(data["states"]))
            data["features_valid_computed"] = features_valid
            feature_cols = [f"computed_features_{i}" for i in range(features.shape[1])]
            features_df = pd.DataFrame(features, columns=feature_cols, index=data.index)
            data = pd.concat([data, features_df], axis=1)

        # combine features_valid
        if self.features is None:
            data["features_valid_provided"] = True
        if "features_valid_computed" not in data.columns:
            data["features_valid_computed"] = True
        data["features_valid"] = (
            data["features_valid_provided"] & data["features_valid_computed"]
        )
        column_list = [
            "final_id",
            "text",
            "step",
            "final_object",
            "iteration",
            "total_reward",
            "loss",
            *(self.metrics if self.metrics is not None else []),
            "logprobs_forward",
            "logprobs_backward",
            "features_valid",
            *(self.features if self.features is not None else []),
            *(feature_cols if feature_cols is not None else []),
        ]
        data = data[column_list]

        # create db if not existing, shift final ids and save to db
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()
        query = (
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trajectories'"
        )
        cur.execute(query)
        table_exists = cur.fetchone() is not None
        if table_exists:
            query = "SELECT COALESCE(MAX(final_id), 0) AS max FROM trajectories"
            offset = pd.read_sql_query(query, conn)["max"][0]
            data["final_id"] = data["final_id"] + offset + 1
        data.to_sql("trajectories", conn, if_exists="append", index=False)

        # indexing
        if not table_exists:
            cur.execute("CREATE INDEX idx_points_finalid ON trajectories(final_id)")
            cur.execute("CREATE INDEX idx_points_text ON trajectories(text)")
            cur.execute("CREATE INDEX idx_points_iteration ON trajectories(iteration)")
            cur.execute("CREATE INDEX idx_points_reward ON trajectories(total_reward)")
            cur.execute("CREATE INDEX idx_points_loss ON trajectories(loss)")

        # compute graphs and save nodes and edges db
        if compute_graph:
            create_graph_dbs(conn)
        conn.close()

        # reset current
        self.current = {
            "batch_idx": None,
            "states": None,
            "total_reward": None,
            "iteration": None,
            "logprobs_forward": None,
            "logprobs_backward": None,
        }
        self.current.update({name: None for name in self.metrics or []})
        self.current.update({name: None for name in self.features or []})

    def compute_graph(self):
        """Add nodes and edges tables to db by computing the graph."""
        conn = sqlite3.connect(self.db)
        create_graph_dbs(conn)
        conn.close()

    def create_and_append_testset(
        self,
        texts: list | None = None,
        total_reward: np.ndarray | torch.Tensor | None = None,
        metrics: dict | None = None,
        features: np.ndarray | torch.Tensor | None = None,
        features_valid: np.ndarray | torch.Tensor | None = None,
    ):
        """Create the testset in the expected format.

        Expects final states, their reward and their features.
        Provide the same features as in the logged training data.
        Allows for passing the whole data at once or in chunks.
        If passed in chunks just call the function repeatedly.
        It then appends to the created testset each time.

        Parameters
        ----------
        texts:
            array of shape (chunksize,),
            containing unique string representations of states.
            Use the same function to generate texts as in logging
        total_reward:
            total rewards of the states, shape (chunksize,)
        metrics:
            Optional: If there are more metrics of the final states specify them here.
            Expects a dict with the title of the metric as key and the metric for all
            states as array, tensor or list.
        features:
            array or tensor of shape (len(features), chunksize) for the downprojections.
            Dtype should be int or float.
        features_valid:
            bool array, tensor or list of shape (chunksize,),
            Specifying if the features of a state are valid.
        """
        assert texts is not None, "Specify text representation of states"
        assert total_reward is not None, "Specify rewards of the objects"
        assert (
            features is not None and features_valid is not None
        ), "Specify features of the objects"
        assert (
            len(texts) == len(total_reward) == features.shape[1] == len(features_valid)
        ), "lengths do not match"

        df = pd.DataFrame({"text": texts, "total_reward": total_reward})
        if metrics is not None:
            for k, v in metrics.items():
                df[k] = v
        df["features_valid"] = features_valid
        new_cols = {f"f_{i}": f for i, f in enumerate(features)}
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        df.insert(0, "id", -df.index - 1)

        # create db or append to it
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='testset'"
        )
        table_exists = cur.fetchone() is not None
        if table_exists:
            query = "SELECT COALESCE(MIN(id), 0) AS min FROM testset"
            offset = pd.read_sql_query(query, conn)["min"][0]
            df["id"] = df["id"] + offset
        df.to_sql("testset", conn, if_exists="append", index=False)

        # create indices the first time
        if not table_exists:
            cur.execute("CREATE INDEX idx_testset_text ON testset(text)")
            cur.execute("CREATE INDEX idx_testset_reward ON testset(total_reward)")
        conn.close()
