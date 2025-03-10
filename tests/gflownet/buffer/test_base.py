import ast
import pickle
import shutil
import tempfile
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from gflownet.buffer.base import BaseBuffer
from gflownet.envs.ctorus import ContinuousTorus
from gflownet.proxy.torus import Torus
from gflownet.utils.common import tbool, tfloat, tlong


@pytest.fixture
def env_ctorus():
    return ContinuousTorus(n_dim=2, length_traj=3, n_comp=2)


@pytest.fixture
def proxy_ctorus():
    return Torus(normalize=True, alpha=1.0, beta=1.0, device="cpu", float_precision=32)


@pytest.fixture
def tmp_local():
    tmp = Path("./tmp")
    tmp.mkdir(exist_ok=True)
    return tmp


# TODO: add test for "random" type
@pytest.mark.parametrize(
    "train_type, train_path, test_type, test_path, n_train, n_test",
    [
        ("grid", None, "grid", None, 36, 16),
        ("uniform", None, "uniform", None, 10000, 10000),
        (
            "pkl",
            "./tests/data/buffer/ctorus_train.pkl",
            "pkl",
            "./tests/data/buffer/ctorus_test.pkl",
            36,
            16,
        ),
        (
            "csv",
            "./tests/data/buffer/ctorus_train.csv",
            "csv",
            "./tests/data/buffer/ctorus_test.csv",
            36,
            16,
        ),
    ],
)
def test__buffer_init_ctorus(
    env_ctorus,
    proxy_ctorus,
    tmp_local,
    train_type,
    train_path,
    test_type,
    test_path,
    n_train,
    n_test,
):
    env = env_ctorus
    proxy = proxy_ctorus
    proxy.setup(env)
    replay_capacity = 10

    temp_dir = Path(tempfile.mkdtemp(prefix="test__buffer_init_ctorus", dir=tmp_local))
    print(f"Created temporary folder: {temp_dir}")

    train_config = OmegaConf.create(
        {
            "type": train_type,
            "path": train_path,
            "n": n_train,
            "samples_column": "samples",
        }
    )

    test_config = OmegaConf.create(
        {
            "type": test_type,
            "path": test_path,
            "n": n_test,
            "samples_column": "samples",
        }
    )

    buffer = BaseBuffer(
        env=env,
        proxy=proxy,
        replay_capacity=replay_capacity,
        train=train_config,
        test=test_config,
        datadir=temp_dir,
    )

    # check replay
    assert (temp_dir / "replay.csv").exists()
    assert temp_dir / "replay.csv" == buffer.replay_csv
    replay = buffer.load_replay_from_path(temp_dir / "replay.csv")
    assert len(replay) == len(buffer.replay) == 0

    train_samples = np.array(buffer.train.samples.values.tolist())
    test_samples = np.array(buffer.test.samples.values.tolist())

    # check csv type
    def check_csv_loaded_from_path(buffer_df, path):
        type(buffer_df.samples[0]) == list
        expected_df = pd.read_csv(
            path, index_col=0, converters={"samples": ast.literal_eval}
        )

        identical_columns = ["samples", "samples_readable"]
        assert buffer_df[identical_columns].equals(expected_df[identical_columns])
        assert np.allclose(expected_df.scores.values, buffer_df.scores.values)
        assert np.allclose(expected_df.rewards.values, buffer_df.rewards.values)

    if train_type == "csv":
        check_csv_loaded_from_path(buffer.train, train_path)
    if test_type == "csv":
        check_csv_loaded_from_path(buffer.test, test_path)

    # check pkl type
    def check_pkl_loaded_from_path(buffer_df, path):
        with open(path, "rb") as f:
            data_dict = pickle.load(f)
        samples = np.array(data_dict["samples"])
        scores = np.array(data_dict["scores"])
        rewards = np.array(data_dict["rewards"])

        assert np.allclose(samples, np.array(buffer_df.samples.values.tolist()))
        assert np.allclose(scores, buffer_df.scores.values)
        assert np.allclose(rewards, buffer_df.rewards.values)

    if train_type == "pkl":
        check_pkl_loaded_from_path(buffer.train, train_path)
    if test_type == "pkl":
        check_pkl_loaded_from_path(buffer.test, test_path)

    # check output files are created
    for name in ["train.pkl", "test.pkl", "train.csv", "test.csv"]:
        if name is not None:
            assert (temp_dir / name).exists()

    # check output pkl files are correct
    for name, buffer_samples in zip(
        ["train.pkl", "test.pkl"], [buffer.train, buffer.test]
    ):
        if name is not None:
            check_pkl_loaded_from_path(buffer_samples, temp_dir / name)

    # check output csv files are correct
    for name, buffer_df in zip(
        ["train.csv", "test.csv"], [buffer.train, buffer.test]
    ):
        if name is not None:
            check_csv_loaded_from_path(buffer_df, temp_dir / name)

    # check samples
    assert train_samples.shape[0] == train_config.n
    assert test_samples.shape[0] == test_config.n
    assert test_samples.shape[1] == train_samples.shape[1]

    def check_grid(samples, n):
        deltas = (samples[1:, :-1] - samples[:-1, :-1]) % (2 * np.pi)
        values = deltas[deltas != 0]
        mean = np.mean(values)
        std = np.std(values)
        assert np.allclose(mean, 2 * np.pi / np.sqrt(n), atol=1e-4)
        assert np.allclose(std, 0, atol=1e-4)

    def check_uniform(samples):
        # remove step
        samples = samples[:, :-1]
        # expected stats
        exp_mean = np.pi
        exp_std = np.sqrt(np.pi**2 / 3)
        mean = np.mean(samples)
        std = np.std(samples)
        assert np.allclose(mean, exp_mean, atol=1e-1)
        assert np.allclose(std, exp_std, atol=1e-1)

    if train_config.type == "grid":
        check_grid(train_samples, train_config.n)
    if test_config.type == "grid":
        check_grid(test_samples, test_config.n)

    if train_config.type == "uniform":
        check_uniform(train_samples)
    if test_config.type == "uniform":
        check_uniform(test_samples)

    shutil.rmtree(temp_dir)
    print(f"\nDeleted temporary folder: {temp_dir}")


def test__replay_add_ctorus(env_ctorus, proxy_ctorus, tmp_local):
    env = env_ctorus
    env.state_space_atol = 1.0  # make it higher for easier testing
    proxy = proxy_ctorus
    proxy.setup(env)

    n_states = 30
    replay_capacity = 10

    temp_dir = Path(tempfile.mkdtemp(prefix="test__replay_add_ctorus_", dir=tmp_local))
    print(f"Created temporary folder: {temp_dir}")

    train_config = OmegaConf.create(
        {
            "type": "grid",
            "path": None,
            "n": 100,
            "output_pkl": None,
            "output_csv": None,
        }
    )

    test_config = OmegaConf.create(
        {
            "type": "grid",
            "path": None,
            "n": 100,
            "output_pkl": None,
            "output_csv": None,
        }
    )

    buffer = BaseBuffer(
        env=env,
        proxy=proxy,
        replay_capacity=replay_capacity,
        train=train_config,
        test=test_config,
        datadir=temp_dir,
    )

    # sample trajectories
    states = []
    actions_traj = []

    for _ in range(n_states):
        env = env.reset()
        st, atrj = env.trajectory_random()
        states.append(copy(st))
        actions_traj.append(copy(atrj))

    states_proxy = env.states2proxy(states)
    logrewards = proxy.rewards(states_proxy, log=True, return_proxy=False).tolist()

    buffer.add(states, actions_traj, logrewards, 1, buffer="replay")

    assert len(buffer.replay) <= replay_capacity

    # check saved replay file is the same as attribute
    replay_from_file = buffer.load_replay_from_path()

    identical_columns = ["state", "traj", "iter", "state_readable", "traj_readable"]

    for col in identical_columns:
        assert np.all(replay_from_file[col] == buffer.replay[col]), col

    assert np.allclose(replay_from_file["reward"], buffer.replay["reward"], atol=1e-6)

    n_close = 0

    for _ in range(400):
        env = env.reset()
        st, atrj = env.trajectory_random()

        is_close_to_replay = False
        for rstate in buffer.replay.state.values.tolist():
            if np.allclose(rstate, st, atol=env.state_space_atol):
                is_close_to_replay = True
                n_close += 1
                break

        min_reward = np.min(buffer.replay.reward)

        buffer.add([st], [atrj], [min_reward - 1.0], 1, buffer="replay")
        assert st not in buffer.replay.state.values.tolist()
        assert atrj not in buffer.replay.traj.values.tolist()

        buffer.add([st], [atrj], [min_reward + 1.0], 1, buffer="replay")
        if not is_close_to_replay:
            assert st in buffer.replay.state.values.tolist()
            assert atrj in buffer.replay.traj.values.tolist()
        else:
            assert st not in buffer.replay.state.values.tolist()
            assert atrj not in buffer.replay.traj.values.tolist()

    print(f"{n_close} close states tested successfully")

    shutil.rmtree(temp_dir)
    print(f"\nDeleted temporary folder: {temp_dir}")


@pytest.mark.parametrize("mode", ["permutation", "uniform", "weighted"])
def test__select_ctorus(env_ctorus, proxy_ctorus, tmp_local, mode):
    env = env_ctorus
    proxy = proxy_ctorus
    proxy.setup(env)

    n_states = 100
    replay_capacity = 50

    temp_dir = Path(tempfile.mkdtemp(prefix="test__select_ctorus_", dir=tmp_local))
    print(f"Created temporary folder: {temp_dir}")

    train_config = OmegaConf.create(
        {
            "type": "grid",
            "path": None,
            "n": 100,
            "output_pkl": None,
            "output_csv": None,
        }
    )

    test_config = OmegaConf.create(
        {
            "type": "grid",
            "path": None,
            "n": 100,
            "output_pkl": None,
            "output_csv": None,
        }
    )

    buffer = BaseBuffer(
        env=env,
        proxy=proxy,
        replay_capacity=replay_capacity,
        train=train_config,
        test=test_config,
        datadir=temp_dir,
    )

    # sample trajectories
    states = []
    actions_traj = []

    for _ in range(n_states):
        env = env.reset()
        st, atrj = env.trajectory_random()
        states.append(copy(st))
        actions_traj.append(copy(atrj))

    states_proxy = env.states2proxy(states)
    logrewards = proxy.rewards(states_proxy, log=True, return_proxy=False).tolist()

    buffer.add(states, actions_traj, logrewards, 1, buffer="replay")

    # select
    def test_select(dataset, n_samples, mode):
        selected = buffer.select(dataset, n_samples, mode=mode)
        selected.columns.equals(dataset.columns)
        assert len(selected) == n_samples

        for col in selected.columns:
            assert selected[col].isin(dataset[col]).all()

        if mode == "weighted":
            score = "reward" if "reward" in selected.columns else "rewards"
            assert selected[score].median() >= dataset[score].median()

    for n_samples in np.logspace(4, 10, base=2):
        n_samples = int(n_samples)
        test_select(buffer.replay, n_samples, mode)
        test_select(buffer.test, n_samples, mode)
        test_select(buffer.train, n_samples, mode)

    shutil.rmtree(temp_dir)
    print(f"\nDeleted temporary folder: {temp_dir}")
