"""
MLE loss or objective for training GFlowNets.

The MLE loss or objective was defined by Malkin et al. (2022):

    .. _a link: https://arxiv.org/abs/2201.13259
"""

from torchtyping import TensorType

from gflownet.losses.base import BaseLoss
from gflownet.utils.batch import Batch, compute_logprobs_trajectories


class MLE(BaseLoss):
    def __init__(self, **kwargs):
        """
        Initialization method for the MLE loss class.

        Attributes
        ----------
        name : str
            The name of the loss or objective function: MLE
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

        self.name = "MLE"
        self.acronym = "mle"
        self.id = "mlelosa"

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

    def compute_losses_of_batch(self, batch: Batch, batch_cons: Batch = None, use_consistency_loss: bool = False, alpha: float = 0.1) -> TensorType["batch_size"]:
        """
        Computes the MLE loss for each trajectory of the input batch.

        The MLE (mle) loss or objective is computed in this method as is
        defined in

        

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
        

        # MLE loss to train P_F
        mle_loss =  - ( logprobs_f - logprobs_b.detach() ) if logprobs_b.requires_grad else - ( logprobs_f - logprobs_b)

        
        # Optional: consistency loss to train P_B
        if use_consistency_loss: 
            assert batch_cons is not None, "Error: no consistency batch provided for consistency loss!"
            logprobs_f_cons = compute_logprobs_trajectories(
            batch_cons, forward_policy=self.forward_policy, backward=False
            )
            logprobs_f_cons = compute_logprobs_trajectories(
                batch_cons, backward_policy=self.backward_policy, backward=True
            )
            if logprobs_f.requires_grad: 
                consistency_loss = torch.pow( ( logprobs_f.detach() - logprobs_b ) - ( logprobs_f.detach() - logprobs_b ), 2 )
            else: 
                consistency_loss = torch.pow( ( logprobs_f - logprobs_b ) - ( logprobs_f - logprobs_b ), 2 )
        else: 
            consistency_loss = 0 
        
        return mle_loss + alpha * consistency_loss 



    
    
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
