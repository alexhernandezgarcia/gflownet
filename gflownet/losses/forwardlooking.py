"""
Forward Looking loss or objective for training GFlowNets.

The Forward Looking (FL) loss or objective was defined by Pan et al. (2023):

    .. _a link: https://arxiv.org/abs/2302.01687
"""

from torchtyping import TensorType

from gflownet.losses.detailedbalance import DetailedBalance
from gflownet.utils.batch import Batch


class ForwardLooking(DetailedBalance):
    def __init__(self, **kwargs):
        """
        Initialization method for the Forward Looking loss class.

        Attributes
        ----------
        name : str
            The name of the loss or objective function: Forward Looking
        acronym : str
            The acronym of the loss or objective function: FL
        id : str
            The identifier of the loss or objective function: forwardlooking
        """
        super().__init__(**kwargs)

        self.name = "Forward Looking"
        self.acronym = "FL"
        self.id = "forwardlooking"

    def compute_losses_of_batch(self, batch: Batch) -> TensorType["batch_size"]:
        """
        Computes the Forward Looking loss for each state of the input batch.

        The Forward Looking (FL) loss or objective is computed in this method as is
        defined in Equation 11 of Pan et al. (2023).

        .. _a link: https://arxiv.org/abs/2302.01687

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
        logrewards_states = batch.get_rewards(log=True, do_non_terminating=True)
        logrewards_parents = batch.get_rewards_parents(log=True)
        done = batch.get_done()

        # Get logprobs
        masks_f = batch.get_masks_forward(of_parents=True)
        policy_output_f = self.forward_policy(parents_policy)
        logprobs_f = self.env.get_logprobs(
            policy_output_f, actions, masks_f, parents, is_backward=False
        )
        masks_b = batch.get_masks_backward()
        policy_output_b = self.backward_policy(states_policy)
        logprobs_b = self.env.get_logprobs(
            policy_output_b, actions, masks_b, states, is_backward=True
        )

        # Get FL logflows
        logflflows_states = self.state_flow(states_policy)
        # Log FL flow of terminal states is 0 (eq. 9 of paper)
        logflflows_states[done.eq(1)] = 0.0
        # TODO: Optimise by reusing logflows_states and batch.get_parent_indices
        logflflows_parents = self.state_flow(parents_policy)

        # Get energies transitions
        energies_transitions = logrewards_parents - logrewards_states

        # Forward-looking loss
        return (
            logflflows_parents
            - logflflows_states
            + logprobs_f
            - logprobs_b
            + energies_transitions
        ).pow(2)
