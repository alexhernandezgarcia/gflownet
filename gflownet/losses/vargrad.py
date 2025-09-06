"""
VarGrad loss or objective for training GFlowNets.

The VarGrad (VG) loss or objective was defined by https://arxiv.org/abs/2010.10436 and https://arxiv.org/abs/2005.05409 , 
Then it was rediscovered for GFNs.

    .. . _a link (for rediscovering it for GFNs): https://arxiv.org/abs/2302.05446
"""

from torchtyping import TensorType

from gflownet.losses.trajectorybalance import TrajectoryBalance
from gflownet.utils.batch import Batch, compute_logprobs_trajectories


class VarGrad(TrajectoryBalance):
    def __init__(self, **kwargs):
        """
        Initialization method for the VarGrad loss class.

        Attributes
        ----------
        name : str
            The name of the loss or objective function: VarGrad
        acronym : str
            The acronym of the loss or objective function: vg
        id : str
            The identifier of the loss or objective function: vargrad
        """
        super().__init__(**kwargs)

        assert self.forward_policy is not None
        assert self.backward_policy is not None

        # Attribute to indicate that logZ is required in the computation of the loss
        self._requires_log_z = False

        self.name = "VarGrad"
        self.acronym = "VG"
        self.id = "vargrad"

    def is_defined_for_continuous(self) -> bool:
        """
        Returns True if the loss function is well defined for continuous GFlowNets,
        that is continuous environments, or False otherwise.

        The VarGrad loss is well defined for continuous GFlowNets, therefore
        this method returns True.

        Returns
        -------
        True
        """
        return True

    def compute_losses_of_batch(self, batch: Batch) -> TensorType["batch_size"]:
        """
        Computes the VarGrad loss for each trajectory of the input batch.

        The VarGrad loss or objective is computed in this method as is
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

        # Get the LogZ for VarGrad Loss . TODO check that doing .mean(dim=0) is correct? 
        if logprobs_f.requires_grad or logprobs_b.requires_grad:
            logZ = (logrewards + logprobs_b - logprobs_f).detach().mean(dim=0) # average over the batch
        else:
            logZ = (logrewards + logprobs_b - logprobs_f).mean(dim=0)

        # VarGrad loss
        return (logZ.sum() + logprobs_f - logprobs_b - logrewards).pow(2)

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
