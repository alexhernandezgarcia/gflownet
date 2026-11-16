"""
Sub-Trajectory Balance loss or objective for training GFlowNets.

The Sub-Trajectory Balance (SubTB) loss or objective was defined by Madan et al.
(2023):

    .. _a link: https://arxiv.org/abs/2209.12782
"""

import math

import torch
from torchtyping import TensorType

from gflownet.losses.detailedbalance import DetailedBalance
from gflownet.utils.batch import Batch


class SubTrajectoryBalance(DetailedBalance):
    def __init__(self, lambda_: float = 0.9, **kwargs):
        """
        Initialization method for the Sub-Trajectory Balance loss class.

        Parameters
        ----------
        lambda_ : float
            Weight of the sub-trajectories: a sub-trajectory of length k has weight
            lambda_^k. Small values give more weight to short sub-trajectories (closer
            to Detailed Balance) and large values to long sub-trajectories (closer to
            Trajectory Balance).

        Attributes
        ----------
        name : str
            The name of the loss or objective function: Sub-Trajectory Balance
        acronym : str
            The acronym of the loss or objective function: SubTB
        id : str
            The identifier of the loss or objective function: subtrajectorybalance
        """
        super().__init__(**kwargs)

        assert lambda_ > 0
        self.lambda_ = lambda_

        self.name = "Sub-Trajectory Balance"
        self.acronym = "SubTB"
        self.id = "subtrajectorybalance"

    def compute_losses_of_batch(self, batch: Batch) -> TensorType["batch_size"]:
        """
        Computes the Sub-Trajectory Balance loss for each trajectory of the input batch.

        The Sub-Trajectory Balance (SubTB) loss or objective is computed in this method
        as is defined in Equation 11 of Madan et al. (2023).

        .. _a link: https://arxiv.org/abs/2209.12782

        Parameters
        ----------
        batch : Batch
            A batch of trajectories.

        Returns
        -------
        tensor
            The loss of each trajectory in the batch.
        """
        assert batch.is_valid()

        # Get necessary tensors from batch
        states = batch.get_states(policy=False)
        states_policy = batch.get_states(policy=True)
        actions = batch.get_actions()
        parents = batch.get_parents(policy=False)
        parents_policy = batch.get_parents(policy=True)
        done = batch.get_done()
        logrewards = batch.get_terminating_rewards(log=True, sort_by="insertion")

        # Get logprobs
        masks_f = batch.get_masks_forward(of_parents=True)
        policy_output_f = self.forward_policy(parents_policy)
        logprobs_f = batch.readonly_env.get_logprobs(
            policy_output_f, actions, masks_f, parents, is_backward=False
        )
        masks_b = batch.get_masks_backward()
        policy_output_b = self.backward_policy(states_policy)
        logprobs_b = batch.readonly_env.get_logprobs(
            policy_output_b, actions, masks_b, states, is_backward=True
        )

        # Get logflows
        logflows_states = self.state_flow(states_policy)
        logflows_states[done.eq(1)] = logrewards
        # TODO: Optimise by reusing logflows_states and batch.get_parent_indices
        logflows_parents = self.state_flow(parents_policy)

        # Sub-trajectory balance loss of each trajectory
        losses = []
        for batch_indices in batch.trajectories.values():
            # Log-flows of the states of the trajectory, including the source state
            logflows = torch.cat(
                (logflows_parents[batch_indices[:1]], logflows_states[batch_indices])
            )
            # Running sum of log P_F - log P_B along the trajectory, starting at 0
            logratios = logprobs_f[batch_indices] - logprobs_b[batch_indices]
            logratios_cumsum = torch.cat(
                (torch.zeros_like(logratios[:1]), torch.cumsum(logratios, dim=0))
            )
            # diffs[i, j] = log F(s_i) + sum(log P_F) - log F(s_j) - sum(log P_B), over
            # the sub-trajectory from state i to state j
            a = logflows - logratios_cumsum
            diffs = a.unsqueeze(1) - a.unsqueeze(0)
            # Length of each sub-trajectory (only i < j)
            positions = torch.arange(len(a), device=self.device)
            lengths = positions.unsqueeze(0) - positions.unsqueeze(1)
            is_subtrajectory = lengths > 0
            # Normalized weights lambda^(j - i) (softmax avoids overflow)
            weights = torch.softmax(
                lengths[is_subtrajectory] * math.log(self.lambda_), dim=0
            )
            # Weighted average of the squared terms
            losses.append((weights * diffs[is_subtrajectory].pow(2)).sum())
        return torch.stack(losses)

    def aggregate_losses_of_batch(
        self, losses: TensorType["batch_size"], batch: Batch
    ) -> dict[str, float]:
        """
        Aggregates the losses computed from a batch to obtain the overall average loss.

        The result is returned as a dictionary with the following items:
        - 'all': Overall average loss

        Parameters
        ----------
        losses : tensor
            The loss of each trajectory in the batch.
        batch : Batch
            A batch of trajectories.

        Returns
        -------
        loss_dict : dict
            A dictionary of loss aggregations.
        """
        return {
            "all": losses.mean(),
        }
