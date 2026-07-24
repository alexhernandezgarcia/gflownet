"""
Unit tests for gflownet.losses.regularized_loss.RegularizedLoss
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

# TODO:create env (ctorus and grid), policies for them to be able
# to create the losses

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


def make_policy(env):
    config = OmegaConf.create()
    config.type = "mlp"
    config.n_hid = 4
    config.n_layers = 2
    device = env.device
    float_precision = env.float
    return Policy(config, env, device, float_precision)


def make_state_flow(env):
    config = OmegaConf.create()
    config.type = "mlp"
    config.n_hid = 4
    config.n_layers = 2
    device = env.device
    float_precision = env.float
    return StateFlow(config, env, device, float_precision)


def make_loss(name, env):
    forward_policy = make_policy(env)
    backward_policy = make_policy(env)
    state_flow = make_state_flow(env)
    if name == "tb":
        return TrajectoryBalance(
            forward_policy=forward_policy, backward_policy=backward_policy
        )
    if name == "db":
        return DetailedBalance(
            forward_policy=forward_policy,
            backward_policy=backward_policy,
            state_flow=state_flow,
        )
    if name == "vargrad":
        return VarGrad(forward_policy=forward_policy, backward_policy=backward_policy)
    if name == "fm":
        return FlowMatching(forward_policy=forward_policy)


#### Basic tests #####
@pytest.mark.parametrize(
    "env",
    [
        "ctorus",
        "grid",
    ],
    indirect=["env"],
)
def test_compatible_when_aggregation_matches(env, flow_reg):
    loss_tb = make_loss("tb", env)
    loss_fm = make_loss("fm", env)
    loss_vargrad = make_loss("vargrad", env)
    loss_db = make_loss("db", env)

    assert RegularizedLoss.are_compatible(loss_tb, [flow_reg])
    assert RegularizedLoss.are_compatible(loss_vargrad, [flow_reg])
    assert not RegularizedLoss.are_compatible(loss_fm, [flow_reg])
    assert not RegularizedLoss.are_compatible(loss_db, [flow_reg])


@pytest.mark.parametrize(
    "env",
    [
        "ctorus",
        "grid",
    ],
    indirect=["env"],
)
def test_raises_when_incompatible(env, flow_reg):
    loss_fm = make_loss("fm", env)
    loss_db = make_loss("db", env)

    with pytest.raises(Exception):
        RegularizedLoss(loss_fm, [flow_reg])

    with pytest.raises(Exception):
        RegularizedLoss(loss_db, [flow_reg])


@pytest.mark.parametrize(
    "env",
    [
        "ctorus",
        "grid",
    ],
    indirect=["env"],
)
def test_sets_loss_and_regularizers(env, flow_reg):
    loss_tb = make_loss("tb", env)

    reg_loss = RegularizedLoss(loss_tb, [flow_reg, flow_reg])

    assert reg_loss.loss is loss_tb
    assert reg_loss.regularizers == [flow_reg, flow_reg]


def make_gflownet(env, loss_name):
    if loss_name == "tb":
        overrides = ["gflownet=trajectorybalance", "loss=trajectorybalance"]
    elif loss_name == "vargrad":
        overrides = ["gflownet=vargrad", "loss=vargrad"]
    config = load_base_test_config(
        overrides=[
            "gflownet.optimizer.batch_size.forward=4",
            "gflownet.optimizer.n_train_steps=1",
        ]
        + overrides
    )

    # Initialize a GFlowNet agent from the configuration file
    gflownet = gflownet_from_config(config, env=env)

    return gflownet


### Test on batch ###


class TestComputeLossesOfBatch:
    @pytest.mark.parametrize(
        "env, loss_name",
        [
            ("ctorus", "tb"),
            ("grid", "tb"),
            ("ctorus", "vargrad"),
            ("grid", "vargrad"),
        ],
        indirect=["env"],  # only `env` needs indirection; loss_name is a plain value
    )
    def test_combines_loss_and_regularizer_outputs(self, env, loss_name, flow_reg):

        gflownet = make_gflownet(env, loss_name)
        # import ipdb; ipdb.set_trace()
        batch = Batch(
            env=env,
            proxy=gflownet.proxy,
            device=gflownet.device,
            float_type=gflownet.float,
        )
        batch.merge(gflownet.sample_batch(n_forward=4, train=True)[0])

        reg_loss = RegularizedLoss(gflownet.loss, [flow_reg])

        result = reg_loss.compute_losses_of_batch(batch)

        expected_loss = gflownet.loss.compute_losses_of_batch(batch)
        expected_reg = flow_reg.compute_losses_of_batch(batch)

        assert torch.equal(result["loss"], expected_loss)
        assert len(result["regularizers"]) == 1
        assert torch.equal(result["regularizers"][0], expected_reg)
