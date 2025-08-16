"""
Flow Matching loss or objective for training GFlowNets.

The Flow Matching (FM) loss or objective was defined by Bengio et al. (2021):

    .. _a link: https://arxiv.org/abs/2106.04399
"""

from typing import Union

from gflownet.losses.base import BaseLoss
from gflownet.utils.batch import Batch


class FlowMatching(BaseLoss):
    def __init__(self):
        """
        Initialization method for the Flow Matching loss class.

        Attributes
        ----------
        name : str
            The name of the loss or objective function: Flow Matching
        acronym : str
            The acronym of the loss or objective function: FM
        id : str
            The identifier of the loss or objective function: flowmatching
        """
        super().__init__()
        self.name = "Flow Matching"
        self.acronym = "FM"
        self.id = "flowmatching"

    def is_defined_for_continuous(self) -> bool:
        """
        Returns True if the loss function is well defined for continuous GFlowNets,
        that is continuous environments, or False otherwise.

        The Flow Matching loss is currently not well defined for continuous GFlowNets,
        therefore this method returns False.

        Returns
        -------
        False
        """
        return False

    # TODO: consider using epsilon
    def compute_losses_of_batch(self, batch: Batch) -> TensorType["batch_size"]:
        """
        Computes the Flow Matching loss for each state of the input batch.

        The Flow Matching (FM) loss or objective is computed in this method as is
        defined in Equation 12 of Bengio et al. (2021), except that the outer sum is
        ommited here:

        .. _a link: https://arxiv.org/abs/2106.04399

        Parameters
        ----------
        batch : Batch
            A batch of states.

        Returns
        -------
        losses : tensor
            The loss of each state in the batch.
        """
        assert batch.is_valid()

        # Get necessary tensors from batch
        states = batch.get_states(policy=True)
        parents, parents_actions, parents_state_idx = batch.get_parents_all(policy=True)
        done = batch.get_done()
        masks_sf = batch.get_masks_forward()
        parents_a_idx = self.env.actions2indices(parents_actions)
        logrewards = batch.get_rewards(log=True)

        # Compute in-flows
        inflow_logits = torch.full(
            (states.shape[0], self.env.policy_output_dim),
            -torch.inf,
            dtype=self.float,
            device=self.device,
        )
        inflow_logits[parents_state_idx, parents_a_idx] = self.forward_policy(parents)[
            torch.arange(parents.shape[0]), parents_a_idx
        ]
        inflow = torch.logsumexp(inflow_logits, dim=1)

        # Compute out-flows
        outflow_logits = self.forward_policy(states)
        outflow_logits[masks_sf] = -torch.inf
        outflow = torch.logsumexp(outflow_logits, dim=1)
        outflow[done] = logrewards[done]

        # Compute and return the flow matching loss for each state
        return (inflow - outflow).pow(2)
