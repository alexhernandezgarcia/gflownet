"""
S3GFN loss variants.
"""

import math

import torch

from gflownet.losses.trajectorybalance import TrajectoryBalance
from gflownet.utils.batch import Batch, compute_logprobs_trajectories


class S3TrajectoryBalance(TrajectoryBalance):
    """
    Trajectory Balance with an S3GFN contrastive auxiliary loss.
    """

    uses_s3gfn_aux = True

    def __init__(self, aux_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.aux_weight = aux_weight
        self.name = "S3 Trajectory Balance"
        self.acronym = "S3TB"
        self.id = "s3gfn"

    def compute(self, batch: Batch, get_sublosses: bool = False):
        losses = self.compute_losses_of_batch(batch)
        aggregated = self.aggregate_losses_of_batch(losses, batch)
        if get_sublosses:
            return aggregated
        return aggregated["all"]

    def aggregate_losses_of_batch(self, losses, batch: Batch) -> dict[str, float]:
        tb_loss = losses.mean()
        return {
            "tb": tb_loss,
            "all": tb_loss,
        }

    def compute_replay_loss(self, pos_batch: Batch, neg_batch: Batch):
        zero = torch.zeros((), device=self.device, dtype=self.float)
        if (
            pos_batch is None
            or neg_batch is None
            or len(pos_batch) == 0
            or len(neg_batch) == 0
        ):
            return {
                "replay_tb": zero,
                "aux": zero,
                "all": zero,
            }

        replay_tb = self.compute_losses_of_batch(pos_batch).mean()
        aux_loss = self._compute_aux_loss(pos_batch, neg_batch)
        return {
            "replay_tb": replay_tb,
            "aux": aux_loss,
            "all": replay_tb + self.aux_weight * aux_loss,
        }

    def _compute_aux_loss(self, pos_batch: Batch, neg_batch: Batch):
        seq_logprobs = compute_logprobs_trajectories(
            pos_batch, forward_policy=self.forward_policy, backward=False
        )
        neg_seq_logprobs = compute_logprobs_trajectories(
            neg_batch, forward_policy=self.forward_policy, backward=False
        )
        if seq_logprobs.numel() == 0 or neg_seq_logprobs.numel() == 0:
            return torch.zeros((), device=self.device, dtype=self.float)

        pos_seq_logprobs = seq_logprobs
        neg_log_sum = torch.logsumexp(neg_seq_logprobs, dim=0) - math.log(
            max(neg_seq_logprobs.numel(), 1.0)
        )
        aux_loss = -(
            pos_seq_logprobs - torch.logaddexp(pos_seq_logprobs, neg_log_sum)
        ).mean()
        return aux_loss
