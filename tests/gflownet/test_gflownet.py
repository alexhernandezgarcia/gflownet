from copy import copy

import pytest
import torch
from utils_for_tests import ch_tmpdir, load_base_test_config

from gflownet.utils.common import gflownet_from_config


@pytest.fixture
def gfn_ccube():
    exp_name = "+experiments=/ccube/corners"
    config = load_base_test_config(overrides=[exp_name])
    # Change batch size
    config.gflownet.optimizer.batch_size.forward = 10
    with ch_tmpdir() as tmpdir:
        print(f"Current GFlowNetAgent execution directory: {tmpdir}")
        gfn = gflownet_from_config(config)
    return gfn


def test__compute_logprobs_trajectories__logprobs_from_batch_are_same_as_computed_cube(
    gfn_ccube,
):
    gfn = gfn_ccube

    collect_backwards_masks = gfn.loss in [
        "trajectorybalance",
        "detailedbalance",
        "forwardlooking",
    ]

    batch, times = gfn.sample_batch(
        n_forward=gfn.batch_size.forward,
        n_train=gfn.batch_size.backward_dataset,
        n_replay=gfn.batch_size.backward_replay,
        collect_forwards_masks=True,
        collect_backwards_masks=collect_backwards_masks,
    )
    batch_no_lp = copy(batch)
    batch_no_lp.logprobs_forward = [None] * len(batch)

    assert batch.logprobs_forward != batch_no_lp.logprobs_forward

    lp_fw = gfn.compute_logprobs_trajectories(batch, False)
    lp_bw = gfn.compute_logprobs_trajectories(batch, True)

    lp_fw_no = gfn.compute_logprobs_trajectories(batch_no_lp, False)
    lp_bw_no = gfn.compute_logprobs_trajectories(batch_no_lp, True)

    masks_f = batch_no_lp.get_masks_forward(of_parents=True)
    parents_policy = batch_no_lp.get_parents(policy=True)
    actions = batch_no_lp.get_actions()
    parents_policy = batch_no_lp.get_parents(policy=True)
    parents = batch_no_lp.get_parents(policy=False)
    policy_output_f = gfn.forward_policy(parents_policy)
    logprobs_states_fw = gfn.env.get_logprobs(
        policy_output_f, actions, masks_f, parents, False
    )

    logpobs_fw_from_batch = batch.get_logprobs()

    traj_idx = torch.tensor(batch.traj_indices)
    for tit in range(len(lp_fw)):
        if not torch.allclose(lp_fw[tit], lp_fw_no[tit]):
            lps_rc = logprobs_states_fw[traj_idx == tit]
            lps_b = logpobs_fw_from_batch[traj_idx == tit]
            print(torch.isclose(lps_rc, lps_b))
            print(torch.sum(torch.logical_not(torch.isclose(lps_rc, lps_b))))

    # import ipdb; ipdb.set_trace()
    assert torch.allclose(logprobs_states_fw, logpobs_fw_from_batch, atol=1e-3)
    assert torch.allclose(lp_fw, lp_fw_no, atol=1e-3)
    assert torch.allclose(lp_bw, lp_bw_no, atol=1e-3)


@pytest.fixture
def gfn_grid():
    exp_name = "+experiments=/grid/corners"
    config = load_base_test_config(overrides=[exp_name])
    # Change batch size
    config.gflownet.optimizer.batch_size.forward = 6
    with ch_tmpdir() as tmpdir:
        print(f"Current GFlowNetAgent execution directory: {tmpdir}")
        gfn = gflownet_from_config(config)
    return gfn


def test__compute_logprobs_trajectories__logprobs_from_batch_are_same_as_computed_grid(
    gfn_grid,
):
    gfn = gfn_grid

    collect_backwards_masks = gfn.loss in [
        "trajectorybalance",
        "detailedbalance",
        "forwardlooking",
    ]

    batch, times = gfn.sample_batch(
        n_forward=gfn.batch_size.forward,
        n_train=gfn.batch_size.backward_dataset,
        n_replay=gfn.batch_size.backward_replay,
        collect_forwards_masks=True,
        collect_backwards_masks=collect_backwards_masks,
    )
    batch_no_lp = copy(batch)
    batch_no_lp.logprobs_forward = [None] * len(batch)

    assert batch.logprobs_forward != batch_no_lp.logprobs_forward

    lp_fw = gfn.compute_logprobs_trajectories(batch, False)
    lp_bw = gfn.compute_logprobs_trajectories(batch, True)

    lp_fw_no = gfn.compute_logprobs_trajectories(batch_no_lp, False)
    lp_bw_no = gfn.compute_logprobs_trajectories(batch_no_lp, True)

    assert torch.allclose(lp_fw, lp_fw_no, atol=1e-6)
    assert torch.allclose(lp_bw, lp_bw_no, atol=1e-6)
