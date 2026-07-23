"""
Flow regularization for training non-acyclic continuous gflownets

This regularization wwas defined by Korolev et al. (2026):

    .. _a link: https://arxiv.org/pdf/2606.16073 (equation 22)
"""

from torchtyping import TensorType

from gflownet.losses.base import BaseLoss
from gflownet.utils.batch import Batch, compute_logprobs_trajectories
from gflownet.utils.common import tlong


class FlowRegularization(BaseLoss):
    def __init__(self, gamma=1.0, use_log=True, **kwargs):
        """
        Initialization method for the Flow Regularization loss class.

        Attributes
        ----------
        name : str
            The name of the loss or objective function: Flow Regularization
        acronym : str
            The acronym of the loss or objective function: FR
        id : str
            The identifier of the loss or objective function: flowreg
        gamma : float
            Multiplier to control the magnitude of regularization
        use_log : bool
            If True, the regulariser with be taken under the logarithm, othervise
            the original regularizer from the paper is used (w/o the logarithm)
        """
        super().__init__(**kwargs)
        self._requires_log_z = False

        self.name = "Flow Regularization"
        self.acronym = "FR"
        self.id = "flowreg"
        self.gamma = gamma
        self.use_log = use_log

    def requires_backward_policy(self) -> bool:
        """
        Returns True if the loss function requires a backward policy.

        The Flow Regularization loss does not require a backward policy model, hence False is
        returned.

        Returns
        -------
        False
        """
        return False

    def requires_state_flow_model(self) -> bool:
        """
        Returns True if the loss function requires a state flow model.

        The Flow Regularization loss does not require a state flow model, hence False is
        returned.

        Returns
        -------
        False
        """
        return False

    def is_defined_for_continuous(self) -> bool:
        """
        Returns True if the loss function is well defined for continuous GFlowNets,
        that is continuous environments, or False otherwise.

        The Flow Regularization loss is well defined for continuous GFlowNets, therefore
        this method returns True.

        Returns
        -------
        True
        """
        return True

    def compute_losses_of_batch(self, batch: Batch) -> TensorType["batch_size"]:
        """
        Computes the Flow Regularization loss for each trajectory of the input batch.

        The Flow Regularization (TB) loss or objective is computed in this method as is
        defined in the last term of the Equation 22 of Korolev et al. (2026):

        .. _a link: https://arxiv.org/pdf/2606.16073 (equation 22)

        Parameters
        ----------
        batch : Batch
            A batch of trajectories.

        Returns
        -------
        tensor
            The loss of each trajectory in the batch.
        """
        # Get terminal logrewards from batch
        logrewards_term = batch.get_terminating_rewards(log=True, sort_by="trajectory")
        # Get terminal logprobs
        logprobs = batch.get_logprobs(backward=False)
        term_indicies = tlong(
            batch.get_terminating_indices(sort_by="trajectory"), device=self.device
        )
        logprobs_term = logprobs[term_indicies]

        loss = logrewards_term - logprobs_term

        if not self.use_log:
            loss = torch.exp(loss)

        return self.gamma * loss

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
