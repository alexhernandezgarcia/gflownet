from copy import copy

import pytest
import torch
from utils_for_tests import ch_tmpdir, load_base_test_config

from gflownet.utils.batch import compute_logprobs_trajectories
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


@pytest.mark.parametrize(
    "do_grid, n_forward, n_train, n_replay, collect_reversed_logprobs",
    [
        (False, 100, 0, 0, False),
        (False, 100, 0, 0, True),
        (True, 100, 0, 0, False),
        (True, 100, 0, 0, True),
    ],
)
def test__compute_logprobs_trajectories__logprobs_from_batch_are_same_as_computed_cube_and_grid(
    gfn_ccube,
    gfn_grid,
    do_grid,
    n_forward,
    n_train,
    n_replay,
    collect_reversed_logprobs,
):
    if do_grid:
        gfn = gfn_grid
    else:
        gfn = gfn_ccube
    gfn.collect_reversed_logprobs = collect_reversed_logprobs

    collect_backwards_masks = gfn.loss in [
        "trajectorybalance",
        "detailedbalance",
        "forwardlooking",
    ]

    batch, times = gfn.sample_batch(
        n_forward=n_forward,
        n_train=n_train,
        n_replay=n_replay,
        collect_forwards_masks=True,
        collect_backwards_masks=collect_backwards_masks,
    )
    batch_no_lp = copy(batch)
    batch_no_lp.logprobs_forward = [None] * len(batch)
    batch_no_lp.logprobs_backward = [None] * len(batch)

    if n_forward > 0 or (collect_reversed_logprobs and (n_train > 0 or n_replay > 0)):
        assert batch.logprobs_forward != batch_no_lp.logprobs_forward
    if n_train > 0 or n_replay > 0 or (collect_reversed_logprobs and n_forward > 0):
        assert batch.logprobs_backward != batch_no_lp.logprobs_backward

    lp_fw = compute_logprobs_trajectories(batch, None, gfn.forward_policy, None, False)
    lp_bw = compute_logprobs_trajectories(batch, None, None, gfn.backward_policy, True)

    lp_fw_no = compute_logprobs_trajectories(
        batch_no_lp, None, gfn.forward_policy, None, False
    )
    lp_bw_no = compute_logprobs_trajectories(
        batch_no_lp, None, None, gfn.backward_policy, True
    )

    masks_f = batch_no_lp.get_masks_forward(of_parents=True)
    parents_policy = batch_no_lp.get_parents(policy=True)
    actions = batch_no_lp.get_actions()
    parents_policy = batch_no_lp.get_parents(policy=True)
    parents = batch_no_lp.get_parents(policy=False)
    policy_output_f = gfn.forward_policy(parents_policy)
    logprobs_states_fw = gfn.env.get_logprobs(
        policy_output_f, actions, masks_f, parents, False
    )

    states = batch.get_states(policy=False)
    states_policy = batch.get_states(policy=True)
    masks_b = batch.get_masks_backward()
    policy_output_b = gfn.backward_policy(states_policy)
    logprobs_states_bw = gfn.env.get_logprobs(
        policy_output_b, actions, masks_b, states, True
    )

    logpobs_fw_from_batch, logprobs_fw_valid = batch.get_logprobs()
    logpobs_bw_from_batch, logprobs_bw_valid = batch.get_logprobs(backward=True)

    if n_forward > 0 and collect_reversed_logprobs:
        assert torch.all(logprobs_bw_valid)
    if n_forward > 0 and n_train == 0 and n_replay == 0:
        assert torch.all(logprobs_fw_valid)

    traj_idx = torch.tensor(batch.traj_indices)
    for tit in range(len(lp_fw)):
        if not torch.allclose(lp_fw[tit], lp_fw_no[tit]):
            lps_rc = logprobs_states_fw[traj_idx == tit]
            lps_b = logpobs_fw_from_batch[traj_idx == tit]
            print(torch.isclose(lps_rc, lps_b))
            print(torch.sum(torch.logical_not(torch.isclose(lps_rc, lps_b))))

    for tit in range(len(lp_bw)):
        if not torch.allclose(lp_bw[tit], lp_bw_no[tit]):
            lps_rc = logprobs_states_bw[traj_idx == tit]
            lps_b = logpobs_bw_from_batch[traj_idx == tit]
            print(f"Mistake in trajj: {tit}")
            print(torch.isclose(lps_rc, lps_b))
            print(f"Recomp lps: {lps_rc}")
            print(f"Batch lps: {lps_b}")

    assert torch.allclose(logprobs_states_fw, logpobs_fw_from_batch, atol=1e-3)
    assert torch.allclose(lp_fw, lp_fw_no, atol=1e-3)
    if n_forward > 0 and collect_reversed_logprobs:
        assert torch.allclose(logprobs_states_bw, logpobs_bw_from_batch, atol=1e-3)
    assert torch.allclose(lp_bw, lp_bw_no, atol=1e-3)

    assert lp_bw.requires_grad
    assert lp_fw.requires_grad


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

    lp_fw = compute_logprobs_trajectories(batch, None, gfn.forward_policy, None, False)
    lp_bw = compute_logprobs_trajectories(batch, None, None, gfn.backward_policy, True)

    lp_fw_no = compute_logprobs_trajectories(
        batch_no_lp, None, gfn.forward_policy, None, False
    )
    lp_bw_no = compute_logprobs_trajectories(
        batch_no_lp, None, None, gfn.backward_policy, True
    )

    assert torch.allclose(lp_fw, lp_fw_no, atol=1e-6)
    assert torch.allclose(lp_bw, lp_bw_no, atol=1e-6)
