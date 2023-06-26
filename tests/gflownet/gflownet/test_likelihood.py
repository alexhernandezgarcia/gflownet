import hydra
import pytest
import torch
import yaml
from hydra import compose, initialize

from gflownet.envs.grid import Grid
from gflownet.envs.tetris import Tetris


@pytest.fixture
def grid2d():
    return Grid(n_dim=2, length=5, cell_min=-1.0, cell_max=1.0)


@pytest.fixture
def tetris6x4():
    return Tetris(width=4, height=6)


def _test__sample_backwards__returns_valid_trajectories(
    env, state_term, states, trajectories, n_trajectories
):
    # Load config
    with initialize(version_base="1.1", config_path="../../../config", job_name="xxx"):
        config = compose(config_name="tests")
    # Logger
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)
    # Proxy
    proxy = hydra.utils.instantiate(
        config.proxy, device=config.device, float_precision=config.float_precision
    )
    # Set proxy in env
    env.proxy = proxy
    # No buffer
    config.env.buffer.train = None
    config.env.buffer.test = None
    # GFlowNet agent
    gflownet = hydra.utils.instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env=env,
        buffer=config.env.buffer,
        logger=logger,
    )
    # Sample backward trajectories from state_term
    #     states_term = env.get_uniform_terminating_states(2, seed=1)
    batch = gflownet.sample_backwards([state_term], n_trajectories)
    for traj_idx in batch:
        assert batch[traj_idx]["states"] in states
        assert batch[traj_idx]["actions"] in trajectories


@pytest.mark.parametrize(
    "state_term, states, trajectories",
    [
        (
            [0, 0],
            [[[0, 0], [0, 0]]],
            [[(0, 0)]],
        ),
        (
            [1, 1],
            ([[1, 1], [1, 1], [1, 0], [0, 0]], [[1, 1], [1, 1], [0, 1], [0, 0]]),
            ([(0, 0), (0, 1), (1, 0)], [(0, 0), (1, 0), (0, 1)]),
        ),
    ],
)
def test__sample_backwards__returns_valid_trajectories_grid2d(
    grid2d, state_term, states, trajectories
):
    n_trajectories = 3 * len(trajectories)
    _test__sample_backwards__returns_valid_trajectories(
        grid2d, state_term, states, trajectories, n_trajectories
    )


@pytest.mark.parametrize(
    "state_term, states, trajectories",
    [
        (
            [
                [100, 000, 000, 000],
                [100, 000, 000, 000],
                [100, 000, 000, 000],
                [100, 000, 000, 000],
                [400, 400, 401, 401],
                [400, 400, 401, 401],
            ],
            [
                [
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [400, 400, 401, 401],
                    [400, 400, 401, 401],
                ],
                [
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [400, 400, 401, 401],
                    [400, 400, 401, 401],
                ],
                [
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [400, 400, 401, 401],
                    [400, 400, 401, 401],
                ],
                [
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 401, 401],
                    [000, 000, 401, 401],
                ],
                [
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ],
            ],
            [[(0, 0, 0), (1, 0, 0), (4, 0, 0), (4, 0, 2)]],
        ),
        (
            [
                [100, 000, 000, 000],
                [100, 000, 000, 000],
                [100, 000, 000, 000],
                [100, 000, 000, 000],
                [400, 400, 401, 401],
                [400, 400, 401, 401],
            ],
            [
                [
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [400, 400, 401, 401],
                    [400, 400, 401, 401],
                ],
                [
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [400, 400, 401, 401],
                    [400, 400, 401, 401],
                ],
                [
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [400, 400, 401, 401],
                    [400, 400, 401, 401],
                ],
                [
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [400, 400, 000, 000],
                    [400, 400, 000, 000],
                ],
                [
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ],
            ],
            [[(0, 0, 0), (1, 0, 0), (4, 0, 2), (4, 0, 0)]],
        ),
        (
            [
                [100, 000, 000, 000],
                [100, 000, 000, 000],
                [100, 000, 000, 000],
                [100, 000, 000, 000],
                [400, 400, 401, 401],
                [400, 400, 401, 401],
            ],
            [
                [
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [400, 400, 401, 401],
                    [400, 400, 401, 401],
                ],
                [
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [400, 400, 401, 401],
                    [400, 400, 401, 401],
                ],
                [
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [100, 000, 000, 000],
                    [400, 400, 000, 000],
                    [400, 400, 000, 000],
                ],
                [
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [400, 400, 000, 000],
                    [400, 400, 000, 000],
                ],
                [
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                    [000, 000, 000, 000],
                ],
            ],
            [[(0, 0, 0), (4, 0, 2), (1, 0, 0), (4, 0, 0)]],
        ),
    ],
)
def test__sample_backwards__returns_valid_trajectories_tetris6x4(
    tetris6x4, state_term, states, trajectories
):
    n_trajectories = 3 * len(trajectories)
    _test__sample_backwards__returns_valid_trajectories(
        tetris6x4, state_term, states, trajectories, n_trajectories
    )
