"""
Trajectory Balance loss or objective for training GFlowNets.

The Trajectory Balance (TB) loss or objective was defined by Malkin et al. (2022):

    .. _a link: https://arxiv.org/abs/2201.13259
"""

from torchtyping import TensorType

from gflownet.losses.base import BaseLoss
from gflownet.utils.batch import Batch, compute_logprobs_trajectories


class TrajectoryBalance(BaseLoss):
    def __init__(self, **kwargs):
        """
        Initialization method for the Trajectory Balance loss class.

        Attributes
        ----------
        name : str
            The name of the loss or objective function: Trajectory Balance
        acronym : str
            The acronym of the loss or objective function: TB
        id : str
            The identifier of the loss or objective function: trajectorybalance
        """
        super().__init__(**kwargs)

        assert self.forward_policy is not None
        assert self.backward_policy is not None

        # Attribute to indicate that logZ is required in the computation of the loss
        self._requires_log_z = True

        self.name = "Trajectory Balance"
        self.acronym = "TB"
        self.id = "trajectorybalance"

    def requires_backward_policy(self) -> bool:
        """
        Returns True if the loss function requires a backward policy.

        The Trajectory Balance loss does require a backward policy model, hence True is
        returned.

        Returns
        -------
        True
        """
        return True

    def requires_state_flow_model(self) -> bool:
        """
        Returns True if the loss function requires a state flow model.

        The Trajectory Balance loss does not require a state flow model, hence False is
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

        The Trajectory Balance loss is well defined for continuous GFlowNets, therefore
        this method returns True.

        Returns
        -------
        True
        """
        return True

    def compute_losses_of_batch(self, batch: Batch) -> TensorType["batch_size"]:
        """
        Computes the Trajectory Balance loss for each trajectory of the input batch.

        The Trajectory Balance (TB) loss or objective is computed in this method as is
        defined in Equation 14 of Malkin et al. (2022).

        .. _a link: https://arxiv.org/abs/2201.13259

        Parameters
        ----------
        batch : Batch
            A batch of trajectories.

        Returns
        -------
        tensor
            The loss of each trajectory in the batch.
        """
        # Get logprobs of forward and backward transitions
        logprobs_f = compute_logprobs_trajectories(
            batch, forward_policy=self.forward_policy, backward=False
        )
        logprobs_b = compute_logprobs_trajectories(
            batch, backward_policy=self.backward_policy, backward=True
        )
        # Get rewards from batch
        logrewards = batch.get_terminating_rewards(log=True, sort_by="trajectory")

        # Trajectory balance loss
        return (self.logZ.sum() + logprobs_f - logprobs_b - logrewards).pow(2)

    # TODO: extend with loss over the different types of trajectories (forward, replay
    # buffer, training set...)
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
