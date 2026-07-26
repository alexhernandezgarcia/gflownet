"""
Unit tests for gflownet.regularizers.flowregularization.FlowRegularization
"""

import pytest
import torch
from omegaconf import OmegaConf
from utils_for_tests import load_base_test_config

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.envs.grid import Grid
from gflownet.losses.detailedbalance import DetailedBalance
from gflownet.losses.flowmatching import FlowMatching
from gflownet.losses.regularized import RegularizedLoss
from gflownet.losses.trajectorybalance import TrajectoryBalance
from gflownet.losses.vargrad import VarGrad
from gflownet.policy.base import Policy
from gflownet.policy.state_flow import StateFlow
from gflownet.regularizers.flowregularization import FlowRegularization
from gflownet.utils.batch import Batch
from gflownet.utils.common import gflownet_from_config

### Utils for basic tests ###


@pytest.fixture
def env(request):
    """
    Indirect fixture: dispatches to the right environment based on the
    string passed via parametrize's `env` argname.
    """
    if request.param == "grid":
        return Grid(n_dim=3, length=5, cell_min=-1.0, cell_max=1.0)
    elif request.param == "ctorus":
        return ContinuousTorus(n_dim=2, length_traj=3)
    else:
        raise ValueError(f"Unknown env param: {request.param}")


@pytest.fixture
def flow_reg():
    return FlowRegularization()


def make_gflownet(env, loss_name):
    if loss_name == "tb":
        overrides = ["gflownet=trajectorybalance", "loss=trajectorybalance"]
    elif loss_name == "vargrad":
        overrides = ["gflownet=vargrad", "loss=vargrad"]
    config = load_base_test_config(
        overrides=[
            "gflownet.optimizer.batch_size.forward=4",
            "gflownet.optimizer.n_train_steps=1",
            "buffer.replay_capacity=10",
        ]
        + overrides
    )

    # Initialize a GFlowNet agent from the configuration file
    gflownet = gflownet_from_config(config, env=env)

    return gflownet


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------


class TestFlowRegularizationInit:
    def test_default_attributes(self, flow_reg):
        assert flow_reg.name == "Flow Regularization"
        assert flow_reg.acronym == "FR"
        assert flow_reg.id == "flowreg"
        assert flow_reg.gamma == 1.0
        assert flow_reg.use_log is True
        assert flow_reg._requires_log_z is False

    def test_custom_gamma_and_use_log(self):
        reg = FlowRegularization(gamma=2.5, use_log=False)
        assert reg.gamma == 2.5
        assert reg.use_log is False


class TestFlowRegularizationMetadataMethods:
    def test_aggregates_over(self, flow_reg):
        assert flow_reg.aggregates_over() == "trajectories"

    def test_requires_backward_policy(self, flow_reg):
        assert flow_reg.requires_backward_policy() is False

    def test_requires_state_flow_model(self, flow_reg):
        assert flow_reg.requires_state_flow_model() is False

    def test_is_defined_for_continuous(self, flow_reg):
        assert flow_reg.is_defined_for_continuous() is True


# ---------------------------------------------------------------------------
# On a batch
# ---------------------------------------------------------------------------


class TestOnABatch:
    @pytest.mark.parametrize(
        "env, loss_name",
        [
            ("ctorus", "tb"),
            ("grid", "tb"),
            ("ctorus", "vargrad"),
            ("grid", "vargrad"),
        ],
        indirect=["env"],
    )
    def test_output_format_is_correct(self, env, loss_name, flow_reg):

        gflownet = make_gflownet(env, loss_name)
        n_traj = 4
        batch = Batch(
            env=env,
            proxy=gflownet.proxy,
            device=gflownet.device,
            float_type=gflownet.float,
        )
        batch.merge(gflownet.sample_batch(n_forward=n_traj, train=True)[0])

        result_batch = flow_reg.compute_losses_of_batch(batch)
        result = flow_reg.aggregate_losses_of_batch(result_batch, batch, gflownet.loss)

        assert f"{flow_reg.id} all" in result.keys()
        assert result_batch.shape == (n_traj,)

    @pytest.mark.parametrize(
        "env, loss_name",
        [
            ("ctorus", "tb"),
            ("ctorus", "vargrad"),
        ],
        indirect=["env"],
    )
    def test_output_values_are_correct_ctorus(self, env, loss_name, flow_reg):

        gflownet = make_gflownet(env, loss_name)
        n_traj = 4
        batch = Batch(
            env=env,
            proxy=gflownet.proxy,
            device=gflownet.device,
            float_type=gflownet.float,
        )
        batch.merge(gflownet.sample_batch(n_forward=n_traj, train=True)[0])

        result_batch = flow_reg.compute_losses_of_batch(batch)
        result = flow_reg.aggregate_losses_of_batch(result_batch, batch, gflownet.loss)

        # Uniform proxy gives logreward == 0 and in ctorus logprob of EOS == 0.
        # because traj length is fixed, therefore flowreg should be 0.
        assert result["flowreg all"] == 0.0
        assert torch.all(result_batch == 0.0)

        flow_reg.use_log = False
        result_batch = flow_reg.compute_losses_of_batch(batch)
        result = flow_reg.aggregate_losses_of_batch(result_batch, batch, gflownet.loss)

        assert result["flowreg all"] == 1.0
        assert torch.all(result_batch == 1.0)

    @pytest.mark.parametrize(
        "env, loss_name",
        [
            ("ctorus", "tb"),
            ("ctorus", "vargrad"),
        ],
        indirect=["env"],
    )
    def test_output_values_are_correct_ctorus_bakward(self, env, loss_name, flow_reg):
        gflownet = make_gflownet(env, loss_name)
        # sample batch forward
        batch = Batch(
            env=env,
            proxy=gflownet.proxy,
            device=gflownet.device,
            float_type=gflownet.float,
        )
        batch.merge(gflownet.sample_batch(n_forward=10, n_replay=0, train=True)[0])

        # Add to replay buffer
        states_term = batch.get_terminating_states(sort_by="trajectory")
        actions_trajectories = batch.get_actions_trajectories()
        rewards = batch.get_terminating_rewards(sort_by="trajectory")
        gflownet.buffer.add(
            states_term,
            actions_trajectories,
            rewards,
            0,
            buffer="replay",
        )

        # sample batch backward
        batch = Batch(
            env=env,
            proxy=gflownet.proxy,
            device=gflownet.device,
            float_type=gflownet.float,
        )
        batch.merge(gflownet.sample_batch(n_forward=0, n_replay=4, train=True)[0])

        # need to ocompute loss first to set all logprobs in the batch to valid
        loss = gflownet.loss.compute(batch)

        result_batch = flow_reg.compute_losses_of_batch(batch)
        result = flow_reg.aggregate_losses_of_batch(result_batch, batch, gflownet.loss)

        assert result["flowreg all"] == 0.0
        assert torch.all(result_batch == 0.0)

        flow_reg.use_log = False
        result_batch = flow_reg.compute_losses_of_batch(batch)
        result = flow_reg.aggregate_losses_of_batch(result_batch, batch, gflownet.loss)

        assert result["flowreg all"] == 1.0
        assert torch.all(result_batch == 1.0)

    @pytest.mark.parametrize(
        "env, loss_name",
        [
            ("grid", "tb"),
            ("grid", "vargrad"),
        ],
        indirect=["env"],
    )
    def test_output_values_are_correct_grid(self, env, loss_name, flow_reg):

        gflownet = make_gflownet(env, loss_name)
        n_traj = 4
        batch = Batch(
            env=env,
            proxy=gflownet.proxy,
            device=gflownet.device,
            float_type=gflownet.float,
        )
        batch.merge(gflownet.sample_batch(n_forward=n_traj, train=True)[0])

        result_batch = flow_reg.compute_losses_of_batch(batch)
        result = flow_reg.aggregate_losses_of_batch(result_batch, batch, gflownet.loss)

        # Uniform proxy gives logreward == 0 and in grid logprob of EOS <= 0.
        # because traj length is not fixed, therefore flowreg should be >= 0.
        assert result["flowreg all"] > 0.0
        assert torch.all(result_batch >= 0.0)

        flow_reg.use_log = False
        result_batch_exp = flow_reg.compute_losses_of_batch(batch)
        result_exp = flow_reg.aggregate_losses_of_batch(
            result_batch, batch, gflownet.loss
        )

        assert result_exp["flowreg all"] > 1.0
        assert torch.all(result_batch_exp == torch.exp(result_batch))

    @pytest.mark.parametrize(
        "env, loss_name",
        [
            ("grid", "tb"),
            ("grid", "vargrad"),
        ],
        indirect=["env"],
    )
    def test_output_values_are_correct_grid_backward(self, env, loss_name, flow_reg):
        gflownet = make_gflownet(env, loss_name)
        # sample batch forward
        batch = Batch(
            env=env,
            proxy=gflownet.proxy,
            device=gflownet.device,
            float_type=gflownet.float,
        )
        batch.merge(gflownet.sample_batch(n_forward=10, n_replay=0, train=True)[0])

        # Add to replay buffer
        states_term = batch.get_terminating_states(sort_by="trajectory")
        actions_trajectories = batch.get_actions_trajectories()
        rewards = batch.get_terminating_rewards(sort_by="trajectory")
        gflownet.buffer.add(
            states_term,
            actions_trajectories,
            rewards,
            0,
            buffer="replay",
        )

        # sample batch backward
        batch = Batch(
            env=env,
            proxy=gflownet.proxy,
            device=gflownet.device,
            float_type=gflownet.float,
        )
        batch.merge(gflownet.sample_batch(n_forward=0, n_replay=4, train=True)[0])

        # need to ocompute loss first to set all logprobs in the batch to valid
        loss = gflownet.loss.compute(batch)

        result_batch = flow_reg.compute_losses_of_batch(batch)
        result = flow_reg.aggregate_losses_of_batch(result_batch, batch, gflownet.loss)

        # Uniform proxy gives logreward == 0 and in grid logprob of EOS <= 0.
        # because traj length is not fixed, therefore flowreg should be >= 0.
        assert result["flowreg all"] > 0.0
        assert torch.all(result_batch >= 0.0)

        flow_reg.use_log = False
        result_batch_exp = flow_reg.compute_losses_of_batch(batch)
        result_exp = flow_reg.aggregate_losses_of_batch(
            result_batch, batch, gflownet.loss
        )

        assert result_exp["flowreg all"] > 1.0
        assert torch.all(result_batch_exp == torch.exp(result_batch))
