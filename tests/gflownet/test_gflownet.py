from copy import copy

import pytest
import torch

from gflownet.utils.common import gflownet_from_config
from gflownet.utils.utils_for_tests import ch_tmpdir, load_base_test_config


@pytest.fixture
def gfn_ccube():
    exp_name = "+experiments=/ccube/corners"
    config = load_base_test_config(overrides=[exp_name])
    with ch_tmpdir() as tmpdir:
        print(f"Current GFlowNetAgent execution directory: {tmpdir}")
        gfn = gflownet_from_config(config)
    return gfn


def test__compute_logprobs_trajectories__logprobs_from_batch_are_same_as_computed(
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

    assert torch.allclose(lp_fw, lp_fw_no, atol=1e-6)
    assert torch.allclose(lp_bw, lp_bw_no, atol=1e-6)
