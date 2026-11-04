import pytest
import torch
from utils_for_tests import load_base_test_config

from gflownet.losses.detailedbalance import DetailedBalance
from gflownet.losses.subtrajectorybalance import SubTrajectoryBalance
from gflownet.losses.trajectorybalance import TrajectoryBalance

# Number of forward and backward (from the train set) trajectories
N_FORWARD_N_TRAIN = [(5, 0), (0, 5), (5, 5)]


# Replaces the config of conftest.py, so that gflownet_for_tests uses SubTB
@pytest.fixture
def config_for_tests():
    return load_base_test_config(
        overrides=[
            "env=grid",
            "env.length=4",
            "proxy=box/corners",
            "gflownet=subtrajectorybalance",
            "loss=subtb",
            "buffer.train.type=all",
            "gflownet.optimizer.n_train_steps=5",
        ]
    )


def get_batch(gfn, n_forward, n_train):
    batch, _ = gfn.sample_batch(
        n_forward=n_forward,
        n_train=n_train,
        collect_forwards_masks=True,
        collect_backwards_masks=True,
    )
    batch.set_proxy(gfn.proxy)
    return batch


@pytest.mark.parametrize("n_forward, n_train", N_FORWARD_N_TRAIN)
def test__compute_losses_of_batch__returns_one_loss_per_trajectory(
    gflownet_for_tests, n_forward, n_train
):
    gfn = gflownet_for_tests
    batch = get_batch(gfn, n_forward, n_train)
    losses = gfn.loss.compute_losses_of_batch(batch)
    assert losses.shape == (len(batch.trajectories),)
    assert torch.all(torch.isfinite(losses))
    assert torch.all(losses >= 0)


@pytest.mark.parametrize("n_forward, n_train", N_FORWARD_N_TRAIN)
def test__compute_losses_of_batch__small_lambda_is_mean_of_db_losses(
    gflownet_for_tests, n_forward, n_train
):
    gfn = gflownet_for_tests
    batch = get_batch(gfn, n_forward, n_train)
    db = DetailedBalance(
        forward_policy=gfn.forward_policy,
        backward_policy=gfn.backward_policy,
        state_flow=gfn.state_flow,
    )
    db_losses = db.compute_losses_of_batch(batch)
    db_means = torch.stack(
        [db_losses[idxs].mean() for idxs in batch.trajectories.values()]
    )
    gfn.loss.lambda_ = 1e-8
    losses = gfn.loss.compute_losses_of_batch(batch)
    assert torch.allclose(losses, db_means, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("n_forward, n_train", N_FORWARD_N_TRAIN)
def test__compute_losses_of_batch__large_lambda_is_tb_loss(
    gflownet_for_tests, n_forward, n_train
):
    gfn = gflownet_for_tests
    batch = get_batch(gfn, n_forward, n_train)
    # TB with log Z equal to the log-flow of the source state
    tb = TrajectoryBalance(
        forward_policy=gfn.forward_policy, backward_policy=gfn.backward_policy
    )
    tb.set_log_z(gfn.state_flow(gfn.env.states2policy([gfn.env.source])))
    tb_losses = tb.compute_losses_of_batch(batch)
    gfn.loss.lambda_ = 1e8
    losses = gfn.loss.compute_losses_of_batch(batch)
    assert torch.allclose(losses, tb_losses, rtol=1e-4, atol=1e-6)


def test__train__runs(gflownet_for_tests):
    assert isinstance(gflownet_for_tests.loss, SubTrajectoryBalance)
    gflownet_for_tests.train()
