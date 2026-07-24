"""
Flow regularization for training non-acyclic continuous gflownets

This regularization wwas defined by Korolev et al. (2026):

    .. _a link: https://arxiv.org/pdf/2606.16073 (equation 22)
"""

import torch
from torchtyping import TensorType

from gflownet.losses.base import BaseLoss
from gflownet.utils.batch import Batch
from gflownet.utils.common import tlong


class BaseRegularization(BaseLoss):
    def __init__(self, **kwargs):
        """
        Initialization method for the Base Regularization class.

        Attributes
        ----------
        name : str
            The name of the regulariser
        acronym : str
            The acronym of the regulariser
        id : str
            The identifier of the regulariser
        """
        super().__init__(**kwargs)

        self.name = "Base Regularization"
        self.acronym = "BR"
        self.id = "basereg"

    def aggregate_losses_of_batch(
        self, losses: TensorType["batch_size"], batch: Batch, loss_instance: BaseLoss
    ) -> dict[str, float]:
        """
        Aggregates the regularizer values computed from a batch to obtain the
        overall average regularizer.

        Parameters
        ----------
        losses : tensor
            The regularizer values of each trajectory / state in the batch.
        batch : Batch
            A batch of trajectories / states.
        loss_instance : BaseLoss or its child
            A loss instance which aggregation function is used

        Returns
        -------
        dict
            A dictionary of regulariser aggregations.
        """
        losses_dict = loss_instance.aggregate_losses_of_batch(losses, batch)
        result = {f"{self.id} {key}": value for key, value in losses_dict.items()}
        return result


class FlowRegularization(BaseRegularization):
    def __init__(self, gamma=1.0, use_log=True, **kwargs):
        """
        Initialization method for the Flow Regularization loss class.

        Attributes
        ----------
        name : str
            The name of the regulariser: Flow Regularization
        acronym : str
            The acronym of the regulariser: FR
        id : str
            The identifier of the regulariser: flowreg
        gamma : float
            Multiplier to control the magnitude of regularization
        use_log : bool
            If True, the regulariser with be taken under the logarithm, otherwise
            the original regularizer from the paper is used (w/o the logarithm)
        """
        super().__init__(**kwargs)
        self._requires_log_z = False

        self.name = "Flow Regularization"
        self.acronym = "FR"
        self.id = "flowreg"
        self.gamma = gamma
        self.use_log = use_log

    def aggregates_over(self) -> str:
        """
        Returns a label indentifying over which objects in the batch
        aggregation happens.

        Returns
        -------
        str
            "trajectories"
        """
        return "trajectories"

    def requires_backward_policy(self) -> bool:
        """
        Returns True if the regulariser requires a backward policy.

        The Flow Regularization does not require a backward policy model, hence False is
        returned.

        Returns
        -------
        False
        """
        return False

    def requires_state_flow_model(self) -> bool:
        """
        Returns True if the regulariser requires a state flow model.

        The Flow Regularization does not require a state flow model, hence False is
        returned.

        Returns
        -------
        False
        """
        return False

    def is_defined_for_continuous(self) -> bool:
        """
        Returns True if the regulariser is well defined for continuous GFlowNets,
        that is continuous environments, or False otherwise.

        The Flow Regularization is well defined for continuous GFlowNets, therefore
        this method returns True.

        Returns
        -------
        True
        """
        return True

    def compute_losses_of_batch(self, batch: Batch) -> TensorType["batch_size"]:
        """
        Computes the Flow Regularization for each trajectory of the input batch.

        The Flow Regularization is computed in this method as it is
        defined in the last term of the Equation 22 of Korolev et al. (2026):

        .. _a link: https://arxiv.org/pdf/2606.16073 (equation 22)

        Parameters
        ----------
        batch : Batch
            A batch of trajectories.

        Returns
        -------
        tensor
            The regulariser value for each trajectory in the batch.
        """
        # Get terminal logrewards from batch
        logrewards_term = batch.get_terminating_rewards(log=True, sort_by="trajectory")
        # Get terminal logprobs
        logprobs, valids = batch.get_logprobs(backward=False)
        # TODO: that will work only with forward sampling, extend it to
        # make it work with backward
        assert torch.all(valids)
        term_indices = tlong(
            batch.get_terminating_indices(sort_by="trajectory"), device=self.device
        )
        logprobs_term = logprobs[term_indices]

        loss = logrewards_term - logprobs_term

        if not self.use_log:
            loss = torch.exp(loss)

        return self.gamma * loss
