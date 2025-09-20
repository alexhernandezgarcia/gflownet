from copy import copy

import pytest
import torch
from utils_for_tests import ch_tmpdir, load_base_test_config

from gflownet.utils.batch import compute_logprobs_trajectories
from gflownet.utils.common import gflownet_from_config, tfloat


@pytest.fixture
def gfn_ccube():
    exp_name = "+experiments=/ccube/corners"
    config = load_base_test_config(overrides=[exp_name])
    # Change train
    config.buffer.train.n = 100
    config.buffer.train.type = "grid"
    with ch_tmpdir() as tmpdir:
        print(f"Current GFlowNetAgent execution directory: {tmpdir}")
        gfn = gflownet_from_config(config)
    return gfn


@pytest.fixture
def gfn_grid():
    exp_name = "+experiments=/grid/corners"
    config = load_base_test_config(overrides=[exp_name])
    # Change train
    config.buffer.train.type = "all"
    with ch_tmpdir() as tmpdir:
        print(f"Current GFlowNetAgent execution directory: {tmpdir}")
        gfn = gflownet_from_config(config)
    return gfn


@pytest.mark.parametrize(
    "gfn",
    [
        "gfn_grid",
        "gfn_ccube",
    ],
)
@pytest.mark.parametrize(
    "n_forward, n_train, collect_reversed_logprobs",
    [
        (100, 0, False),
        (100, 0, True),
        (100, 0, False),
        (100, 0, True),
        (0, 100, False),
        (0, 100, False),
        (0, 100, True),
        (0, 100, True),
        (100, 100, False),
        (100, 100, True),
    ],
)
def test__compute_logprobs_trajectories__logprobs_from_batch_are_same_as_computed(
    gfn,
    n_forward,
    n_train,
    collect_reversed_logprobs,
    request,
):
    gfn = request.getfixturevalue(gfn)
    gfn.collect_reversed_logprobs = collect_reversed_logprobs

    collect_backwards_masks = gfn.loss in [
        "trajectorybalance",
        "detailedbalance",
        "forwardlooking",
    ]
    # Sample batch
    batch, times = gfn.sample_batch(
        n_forward=n_forward,
        n_train=n_train,
        n_replay=0,
        collect_forwards_masks=True,
        collect_backwards_masks=collect_backwards_masks,
    )
    # Create a copy with placeholder logprobs and unavailable logprobs
    batch_no_lp = copy(batch)
    batch_no_lp.logprobs_forward = [
        tfloat(2.0, device=gfn.device, float_type=gfn.float)
    ] * len(batch)
    batch_no_lp.logprobs_forward_avail = [False] * len(batch)
    batch_no_lp.logprobs_backward = [
        tfloat(2.0, device=gfn.device, float_type=gfn.float)
    ] * len(batch)
    batch_no_lp.logprobs_backward_avail = [False] * len(batch)

    if n_forward > 0 or (collect_reversed_logprobs and n_train > 0):
        assert batch.logprobs_forward != batch_no_lp.logprobs_forward
    if n_train > 0 or (collect_reversed_logprobs and n_forward > 0):
        assert batch.logprobs_backward != batch_no_lp.logprobs_backward

    # Compute logprobs of the trajectories
    lp_fw = compute_logprobs_trajectories(batch, None, gfn.forward_policy, None, False)
    lp_bw = compute_logprobs_trajectories(batch, None, None, gfn.backward_policy, True)

    lp_fw_no = compute_logprobs_trajectories(
        batch_no_lp, None, gfn.forward_policy, None, False
    )
    lp_bw_no = compute_logprobs_trajectories(
        batch_no_lp, None, None, gfn.backward_policy, True
    )

    assert torch.allclose(lp_fw, lp_fw_no, atol=1e-3)
    assert torch.allclose(lp_bw, lp_bw_no, atol=1e-3)
    assert lp_bw.requires_grad
    assert lp_fw.requires_grad

    # Compute logprobs of each state manualy: relevant only for debugging

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

    if (n_train > 0 and n_forward == 0) or collect_reversed_logprobs:
        assert torch.all(logprobs_bw_valid)
    if n_forward > 0 and n_train == 0:
        assert torch.all(logprobs_fw_valid)
    if n_train > 0 and collect_reversed_logprobs:
        assert not torch.all(logprobs_fw_valid)

    traj_idx = torch.tensor(batch.traj_indices)
    for tit in range(len(lp_fw)):
        if not torch.allclose(lp_fw[tit], lp_fw_no[tit]):
            lps_rc = logprobs_states_fw[traj_idx == tit]
            lps_b = logpobs_fw_from_batch[traj_idx == tit]
            print(f"Mistake in trajj: {tit}")
            print(torch.isclose(lps_rc, lps_b))
            print(f"Recomp lps: {lps_rc}")
            print(f"Batch lps: {lps_b}")

    for tit in range(len(lp_bw)):
        if not torch.allclose(lp_bw[tit], lp_bw_no[tit]):
            lps_rc = logprobs_states_bw[traj_idx == tit]
            lps_b = logpobs_bw_from_batch[traj_idx == tit]
            print(f"Mistake in trajj: {tit}")
            print(torch.isclose(lps_rc, lps_b))
            print(f"Recomp lps: {lps_rc}")
            print(f"Batch lps: {lps_b}")

    if (n_train > 0 and collect_reversed_logprobs) or n_forward > 0:
        assert torch.allclose(
            logprobs_states_fw[logprobs_fw_valid],
            logpobs_fw_from_batch[logprobs_fw_valid],
            atol=1e-3,
        )

    if n_train > 0 or (n_forward > 0 and collect_reversed_logprobs):
        assert torch.allclose(logprobs_states_bw, logpobs_bw_from_batch, atol=1e-3)


@pytest.mark.parametrize(
    "gfn",
    [
        "gfn_grid",
        "gfn_ccube",
    ],
)
@pytest.mark.parametrize(
    "n_forward, n_train, collect_reversed_logprobs",
    [
        (100, 0, False),
        (10, 0, True),
        (100, 0, False),
        (100, 0, True),
        (0, 100, False),
        (0, 100, False),
        (0, 100, True),
        (0, 100, True),
        (100, 100, False),
        (100, 100, True),
    ],
)
def test__logprobs_validity(
    gfn,
    n_forward,
    n_train,
    collect_reversed_logprobs,
    request,
):
    gfn = request.getfixturevalue(gfn)
    gfn.collect_reversed_logprobs = collect_reversed_logprobs

    collect_backwards_masks = gfn.loss in [
        "trajectorybalance",
        "detailedbalance",
        "forwardlooking",
    ]
    # Sample batch
    batch, times = gfn.sample_batch(
        n_forward=n_forward,
        n_train=n_train,
        n_replay=0,
        collect_forwards_masks=True,
        collect_backwards_masks=collect_backwards_masks,
    )

    logpobs_fw_from_batch, logprobs_fw_valid = batch.get_logprobs()
    logpobs_bw_from_batch, logprobs_bw_valid = batch.get_logprobs(backward=True)
    actions = batch.get_actions()

    # Check that non-valid logprobs are 2.0, and that logprob_bw(eos) == 0 and valid
    for lp, val in zip(logpobs_fw_from_batch.tolist(), logprobs_fw_valid.tolist()):
        if not val:
            assert lp == 2.0
    for lp, val, act in zip(
        logpobs_bw_from_batch.tolist(), logprobs_bw_valid.tolist(), actions
    ):
        if not val:
            assert lp == 2.0
        if act == gfn.env.eos:
            assert lp == 0.0
            assert val
