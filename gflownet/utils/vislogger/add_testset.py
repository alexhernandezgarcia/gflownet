import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from gflownet.utils.common import gflownet_from_config


def add_testset(logdir, states):
    """
    Adds the states as testset to the database. Uses the same env and proxy as in the
    logged run.

    Parameters
    ----------
    logdir: path to the logging directory
    states: States in environment format
    """

    def create_and_append_testset(
        db,
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
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='testset'"
        )
        table_exists = cur.fetchone() is not None
        if table_exists:
            existing_cols = {
                row[1] for row in cur.execute("PRAGMA table_info(testset)")
            }
            for col in df.columns:
                if col not in existing_cols:
                    dtype = df[col].dtype
                    if pd.api.types.is_integer_dtype(dtype):
                        sql_type = "INTEGER"
                    elif pd.api.types.is_float_dtype(dtype):
                        sql_type = "REAL"
                    else:
                        sql_type = "TEXT"
                    cur.execute(f'ALTER TABLE testset ADD COLUMN "{col}" {sql_type}')
            conn.commit()
            query = "SELECT COALESCE(MIN(id), 0) AS min FROM testset"
            offset = pd.read_sql_query(query, conn)["min"][0]
            df["id"] = df["id"] + offset
        df.to_sql("testset", conn, if_exists="append", index=False)

        # create indices the first time
        if not table_exists:
            cur.execute("CREATE INDEX idx_testset_text ON testset(text)")
            cur.execute("CREATE INDEX idx_testset_reward ON testset(total_reward)")
        conn.close()

    path = Path(logdir)
    cfg = OmegaConf.load(path / ".hydra" / "config.yaml")
    cfg["logger"]["vislogger"]["use"] = False
    gfn = gflownet_from_config(cfg)
    rewards = gfn.proxy(gfn.env.states2proxy(states))
    features, features_valid = gfn.env.vis_states2features(states)

    create_and_append_testset(
        db=path / "visdata" / "data.db",
        texts=gfn.env.vis_states2text(states.tolist()),
        total_reward=rewards,
        features=features.T,
        features_valid=features_valid,
    )
