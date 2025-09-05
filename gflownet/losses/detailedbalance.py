"""
Detailed Balance loss or objective for training GFlowNets.

The Detailed Balance (DB) loss or objective was defined by Malkin et al. (2022):

    .. _a link: https://arxiv.org/abs/2201.13259
"""

from torchtyping import TensorType

from gflownet.losses.base import BaseLoss
from gflownet.utils.batch import Batch


class DetailedBalance(BaseLoss):
    def __init__(self, **kwargs):
        """
        Initialization method for the Detailed Balance loss class.

        Attributes
        ----------
        name : str
            The name of the loss or objective function: Detailed Balance
        acronym : str
            The acronym of the loss or objective function: DB
        id : str
            The identifier of the loss or objective function: detailedbalance
        """
        super().__init__(**kwargs)

        assert self.forward_policy is not None
        assert self.backward_policy is not None
        assert self.state_flow is not None

        self.name = "Detailed Balance"
        self.acronym = "DB"
        self.id = "detailedbalance"

    def requires_backward_policy(self) -> bool:
        """
        Returns True if the loss function requires a backward policy.

        The Detailed Balance loss does require a backward policy model, hence True is
        returned.

        Returns
        -------
        True
        """
        return True

    def requires_state_flow_model(self) -> bool:
        """
        Returns True if the loss function requires a state flow model.

        The Detailed Balance loss does require a state flow model, hence True is
        returned.

        Returns
        -------
        True
        """
        return True

    def is_defined_for_continuous(self) -> bool:
        """
        Returns True if the loss function is well defined for continuous GFlowNets,
        that is continuous environments, or False otherwise.

        The Detailed Balance loss is well defined for continuous GFlowNets, therefore
        this method returns True.

        Returns
        -------
        True
        """
        return True

    def compute_losses_of_batch(self, batch: Batch) -> TensorType["batch_size"]:
        """
        Computes the Detailed Balance loss for each state of the input batch.

        The Detailed Balance (DB) loss or objective is computed in this method as is
        defined in Equation 11 of Malkin et al. (2022).

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

        # Detailed balance loss
        return (logflows_parents + logprobs_f - logflows_states - logprobs_b).pow(2)

    def aggregate_losses_of_batch(
        self, losses: TensorType["batch_size"], batch: Batch
    ) -> dict[str, float]:
        """
        Aggregates the losses computed from a batch to obtain the overall average loss
        and the average loss over terminating states and intermediate states.

        The result is returned as a dictionary with the following items:
        - 'all': Overall average loss
        - 'Loss (terminating)': Average loss over terminating states
        - 'Loss (non-term.)': Average loss over non-terminating (intermediate) states

        Parameters
        ----------
        losses : tensor
            The loss of each state in the batch.
        batch : Batch
            A batch of states.

        Returns
        -------
        loss_dict : dict
            A dictionary of loss aggregations.
        """
        done = batch.get_done()
        # Loss of terminating states
        loss_term = losses[done].mean()
        contrib_term = done.eq(1).to(self.float).mean()
        # Loss of non-terminating states
        loss_interm = losses[~done].mean()
        contrib_interm = done.eq(0).to(self.float).mean()
        # Overall loss
        loss_overall = contrib_term * loss_term + contrib_interm * loss_interm

        return {
            "all": loss_overall,
            "Loss (terminating)": loss_term,
            "Loss (non-term.)": loss_interm,
        }
