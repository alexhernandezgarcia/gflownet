"""
MLE loss or objective for training GFlowNets.

The MLE loss or objective was defined by Zhang et al. (2023):

    .. _a link: https://arxiv.org/pdf/2209.02606
"""

import torch
from torchtyping import TensorType

import wandb
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
            The acronym of the loss or objective function: mle
        id : str
            The identifier of the loss or objective function: mleloss
        """
        super().__init__(**kwargs)

        assert self.forward_policy is not None
        assert self.backward_policy is not None

        # Attribute to indicate that logZ is required in the computation of the loss
        self._requires_log_z = False

        self.name = "MLE"
        self.acronym = "mle"
        self.id = "mleloss"

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

    def compute_losses_of_batch(
        self,
        batch: Batch,
        use_consistency_loss: bool = True,
        alpha: float = 1.0,
    ) -> TensorType["batch_size"]:
        """
        Computes the MLE loss for each trajectory of the input batch.

        The MLE loss or objective is computed in this method as is defined in algorithm 1 in Zhang et al. (2023).
        The MLE loss is used to learn P_F, and the consistency loss is used to learn P_B.
        For the consistency loss: we are generalizing eq. (33) (where 2 trajectories only are sampled backwards per terminal state) to any number of trajectories sampled backwards per terminal state.
        To generalise this, we are using VarGrad loss on these trajectories. The reward gets cancelled as these trajectories lead to the same terminal state.
        More formally:
        Given a terminal state s_f, and trajectories (tau_1, ... tau_k) that end in s_f, the generalised consistency loss is defined as:
        Consistency_loss(s_f, tau_1, ... tau_k)= Var( \log ( P_F(\tau) /(R(s_f)P_B(tau)))) = Var( \log (P_F(\tau) /P_B(tau)))).
        For the case k=2, we recover the original consistency loss defined in eq. (33) modulo a constant.


        Parameters
        ----------
        batch : Batch
            A batch of trajectories.

        use_consistency_loss: Bool
            Whether or not to use the consistency loss for P_B.

        alpha: float
            A hyperparameter such that the total loss is MLE_loss + alpha * consistency_loss

        Returns
        -------
        tensor
            A tuple of:
            - the MLE loss (one for each trajectory in the batch),
            - The consistency loss (one for each **unique** terminal state in the batch, as there may be several trajectories that are sampled backwards from the same terminal state).
        """
        terminating_states = batch.get_terminating_states(proxy=True)
        # Get logprobs of forward and backward transitions
        logprobs_f = compute_logprobs_trajectories(
            batch, forward_policy=self.forward_policy, backward=False
        )
        logprobs_b = compute_logprobs_trajectories(
            batch, backward_policy=self.backward_policy, backward=True
        )

        # MLE loss to train P_F
        mle_loss = (
            -(logprobs_f - logprobs_b.detach())
            if logprobs_b.requires_grad
            else -(logprobs_f - logprobs_b)
        )

        # Optional: consistency loss to train P_B
        consistency_loss = 0
        n_duplicates = batch.n_duplicates_batch_train
        bs = len(logprobs_f)
        if use_consistency_loss:
            logprobs_f_cons = logprobs_f.reshape(bs // n_duplicates, n_duplicates)
            logprobs_b_cons = logprobs_b.reshape(bs // n_duplicates, n_duplicates)
            if logprobs_f.requires_grad:
                consistency_loss = torch.var(
                    logprobs_f_cons.detach() - logprobs_b_cons, dim=1
                )  # shape bs//n_duplicates
            else:
                consistency_loss = torch.var(logprobs_f_cons - logprobs_b_cons, dim=1)

        consistency_loss = alpha * consistency_loss
        return mle_loss, consistency_loss

    def aggregate_losses_of_batch(
        self,
        losses: TensorType["batch_size"],
        batch: Batch,
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
        alpha: Float
            A float to balance between the

        Returns
        -------
        loss_dict : dict
            A dictionary of loss aggregations.
        """
        mle_loss, consistency_loss = losses[0], losses[1]
        return {
            "all": mle_loss.mean() + consistency_loss.mean(),
            "mle_loss": mle_loss.mean(),
            "consistency_loss": consistency_loss.mean(),
        }
