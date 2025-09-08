"""
MLE loss or objective for training GFlowNets.

The MLE loss was first defined by Kolya Malkin, and then was implemented in Dinghuai Zhang's paper: 

    .. _a link: https://arxiv.org/abs/2209.02606
"""

from torchtyping import TensorType

from gflownet.losses.base import BaseLoss
from gflownet.utils.batch import Batch, compute_logprobs_trajectories


class MLELoss(BaseLoss):
    def __init__(self, **kwargs):
        """
        Initialization method for the MLE loss class.

        Attributes
        ----------
        name : str
            The name of the loss or objective function: Maximum Likelihood Estimation
        acronym : str
            The acronym of the loss or objective function: MLE
        id : str
            The identifier of the loss or objective function: mle
        """
        super().__init__(**kwargs)

        assert self.forward_policy is not None
        assert self.backward_policy is not None

        # Attribute to indicate that logZ is required in the computation of the loss
        self._requires_log_z = True

        self.name = "MLE"
        self.acronym = "MLE"
        self.id = "mle"

    def requires_backward_policy(self) -> bool:
        """
        Returns True if the loss function requires a backward policy.

        The MLE loss does require a backward policy model, hence True is
        returned.

        Returns
        -------
        True
        """
        return True

    def requires_state_flow_model(self) -> bool:
        """
        Returns True if the loss function requires a state flow model.

        The MLE loss does not require a state flow model, hence False is
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

        The MLE loss is well defined for continuous GFlowNets, therefore
        this method returns True.

        Returns
        -------
        True
        """
        return True

    def compute_losses_of_batch(self, batch: Batch, batch_cons: Batch = None,) -> TensorType["batch_size"]:
        """
        Computes the MLE loss for each trajectory of the input batch.

        The MLE loss is computed as MLE Loss = KL_loss + consistency_loss.

        Parameters
        ----------
        batch : Batch
            A batch of trajectories.

        batch_cons: Batch
            The twin-brother of Batch: Same terminal states, but different trajectories!

        Returns
        -------
        tensor
            The loss of each trajectory in the batch Batch.
        """
        # Get logprobs of forward and backward transitions
        logprobs_f = compute_logprobs_trajectories(
            batch, forward_policy=self.forward_policy, backward=False
        )
        logprobs_b = compute_logprobs_trajectories(
            batch, backward_policy=self.backward_policy, backward=True
        )


        # Get logprobs of forward and backward transitions for the second batch, if not None

        if batch_cons is not None:
            logprobs_f_cons = compute_logprobs_trajectories(
                batch_cons, forward_policy=self.forward_policy, backward=False
            )
            logprobs_b_cons = compute_logprobs_trajectories(
                batch_cons, backward_policy=self.backward_policy, backward=True
            )


        # KL loss for training P_F
        KL_loss = - (logprobs_f - logprobs_b.detach()).mean() 

        # Consistency loss for training P_B 
        consistency_loss = - (logprobs_f.detach() -  logprobs_b  -  (logprobs_f_cons.detach() - logprobs_b_cons) ).pow(2).mean() if batch_cons is not None else 0.0


        return KL_loss +  consistency_loss 

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
