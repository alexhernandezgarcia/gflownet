from copy import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.distributions import Bernoulli
from torchtyping import TensorType

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.utils.common import tbool, tfloat
from gflownet.utils.metrics import angles_allclose

"""
Non-acyclic continuous hyper-torus environment.

This environment extends :class:`~gflownet.envs.ctorus.ContinuousTorus` to
support trajectories that do not have a fixed length and are not
constrained to be acyclic: unlike the parent ctorus environment, states here
do not encode a timestep, so the same state (a point on the torus) can be
reached via multiple different action sequences and revisited, allowing
cycles in the underlying transition graph.

At each non-source state, the policy jointly decides (via a Bernoulli
distribution over an additional "end" logit) whether to terminate the
trajectory (EOS forward / back-to-source backward) or to continue by
sampling an angle increment per dimension from a mixture of von Mises
distributions, as in the parent class. From the source state, termination
is not allowed and only a continuous increment can be taken.

States are represented as a list/vector of angles (in radians, within
$[0, 2\\pi)$) for each of the `n_dim` dimensions. The source state is 
represented as a list of ``None`` values, distinguishing it from any valid 
angle state. The fist step from the source is always uniform over the whole
torus.
"""


class NonAcyclicContinuousTorus(ContinuousTorus):
    r"""
    Non-acyclic continuous hyper-torus environment.

    The action space consists of the increment of the angle $\theta_i$ of each
    dimension $i$. Increments of any magnitude and any sign are allowed

    Trajectories can have various lengths and the time step is NOT included in
    the state. This can create cycles.

    States are represented by the angles (in radians and within
    $[0, 2\pi]$) for all dimensions.

    The increments of the angles are sampled from a mixture of von Mises distributions.
    """

    def __init__(
        self,
        n_dim: int = 2,
        n_comp: int = 1,
        policy_encoding_dim_per_angle: int = None,
        fixed_distr_params: dict = None,
        random_distr_params: dict = None,
        vonmises_min_concentration: float = 1e-3,
        exp_vonmises_concentration: bool = True,
        state_space_atol=1e-6,
        **kwargs,
    ):
        """
        Initializes a NonAcyclicContinuousTorus environent.

        Parameters
        ----------
        n_dim : int
            Dimensionality of the torus
        n_comp : int
           Number of components in the mixture of distributions used to
           sample angle increments.
        policy_encoding_dim_per_angle : int
            Dimensionality of the policy encodings of the angles.
        fixed_distr_params : dict
            Dictionary of parameters of the Bernoulli and the von Mises distributions
            that defines the fixed policy of the environment. It must contain the following keys
            with float values: ``vonmises_mean``, ``vonmises_concentration``,
            ``bernoulli_logit``.
        random_distr_params : dict
            Dictionary of parameters of the Bernoulli and the von Mises distributions
            that defines the random policy of the environment. It must contain the following keys
            with float values: ``vonmises_mean``, ``vonmises_concentration``,
            ``bernoulli_logit``.
        vonmises_min_concentration : float
            Minimum value allowed for the concentration parameter of the von Mises
            distributions.
        exp_vonmises_concentration: bool
            A flag indicating whether to exponentiate concentrations for von Mises distribution.
            Default is True
        state_space_atol: float
            Tolerance for comparing states similarity.
        """
        distr_type = "von_mises"
        if fixed_distr_params is None:
            fixed_distr_params = {
                "vonmises_mean": 0.0,
                "vonmises_concentration": 0.5,
                "bernoulli_logit": 0.0,  # prob to sample 1 is 0.5
            }
        if random_distr_params is None:
            random_distr_params = {
                "vonmises_mean": 0.0,
                "vonmises_concentration": 0.001,
                "bernoulli_logit": 0.0,  # prob to sample 1 is 0.5
            }
        super().__init__(
            distr_type=distr_type,
            n_dim=n_dim,
            length_traj=1,  # ? maybe np.inf?
            n_comp=n_comp,
            policy_encoding_dim_per_angle=policy_encoding_dim_per_angle,
            fixed_distr_params=fixed_distr_params,
            random_distr_params=random_distr_params,
            vonmises_min_concentration=vonmises_min_concentration,
            exp_vonmises_concentration=exp_vonmises_concentration,
            state_space_atol=state_space_atol,
            start_uniform=True,
            **kwargs,
        )
        # remove step dimention from the source
        self.source = [None] * n_dim
        self.reset()

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        The action space is continuous, thus the mask is not of invalid actions as
        in discrete environments, but an indicator of "special cases", for example
        states from which only certain actions are possible.

        The "mask" has 2 elements: the first one identifies if continious inclement
        is invalid, the second one identifies if EOS is invalid. There're three cases

        - If done is True, then the mask is [True, True], everything is invalid.
        - If state is source, then only continious action is valid and EOS is invalid, i.e.
          the mask is [False, True]
        - Otherwise, both the continuous action and EOS are valid and the mask is [False, False]
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [True] * 2
        elif self.is_source(state):
            return [False, True]
        else:
            return [False] * 2

    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        """
        The action is space is continuous, thus the mask is not of invalid actions as
        in discrete environments, but an indicator of "special cases", for example
        states from which only certain actions are possible.

        The "mask" has 2 elements - to match the mask of forward actions - but only
        one is needed for backward actions, thus both elements take the same value,
        according to the following:

        - if done is True, then the mask is True, meaning that only EOS action is valid
        - othervise the mask is False, meaning that EOS is invalid while a continious
        increment or a back-to-source actions are valid.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [True, True]
        else:
            return [False, False]

    def get_valid_actions(
        self,
        mask: Optional[List] = None,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        backward: Optional[bool] = False,
    ) -> List[Tuple]:
        """
        Returns the list of non-invalid (valid, for short) according to the mask of
        invalid actions.

        As a continuous environment, the returned actions are "representatives", that
        is the actions represented in the action space.

        Parameters
        ----------
        mask : list (optional)
            The mask of a state. If None, it is computed in place.
        state : list (optional)
            A state in GFlowNet format. If None, self.state is used.
        done : bool (optional)
            Whether the trajectory is done. If None, self.done is used.
        backward : bool
            True if the transtion is backwards; False if forward.

        Returns
        -------
        list
            The list of representatives of the valid actions.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if mask is None:
            mask = self.get_mask(state, done, backward)
        if backward:
            if mask[0]:
                return [self.eos]
            else:
                return [self.representative_action]
        else:
            actions = []
            if not mask[0]:
                actions.append(self.representative_action)
            if not mask[1]:
                actions.append(self.eos)
            return actions

    def get_end_logits(self, policy_outputs):
        return policy_outputs[:, -1]

    def get_end_distr(
        self,
        end_logits: TensorType["n_states"],
    ):
        distr_end = Bernoulli(logits=end_logits)
        return distr_end

    def _get_timesteps(self, states: List):
        """
        Extract the timestep component from a batch of states.

        In non-acyclis ctorus the states do not contain info about timeteps, so
        this method returns None

        Parameters
        ----------
        states : list
            Batch of states

        Returns
        -------
            None
        """

        return None

    def isclose(
        self,
        state_x: List,
        state_y: List,
        atol: Optional[float] = None,
        do_equal: bool = False,
    ) -> bool:
        """
        Check if two states are close in the state space.

        The states are compared using :func:`angles_allclose`, which accounts
        for the periodicity of the torus (i.e., angles differing by integer
        multiples of :math:`2\pi` are treated as equivalent) and floating-point
        numerical precision.

        Parameters
        ----------
        state_x : list
            First state to compare
        state_y : list
            Second state to compare
        atol : float
            Maximum absolute tolerance threshold for numeric values.
        do_equal : bool
            If True, comparisons are by equality instead of closeness and
            ``atol`` is ignored.

        Returns
        -------
        bool or iterable
            True if the two states are close, False otherwise.
        """
        if not do_equal:
            if atol is None:
                atol = self.state_space_atol
            return angles_allclose(state_x, state_y, atol=atol)
        else:
            return state_x == state_y

    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: TensorType["n_states", "mask_dim"],
        states_from: List,
        is_backward: Optional[bool] = False,
        random_action_prob: Optional[float] = 0.0,
        temperature_logits: Optional[float] = 1.0,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs. First,
        a discrete action is sampled determining whether the trajectory will be terminated
        (from a Bernoulli distribution).
        Then, if it is not terminated, the angle increments are sampled from a mixture
        of Von Mises distributions.

        A distinction between forward and backward actions is made and specified by the
        argument is_backward, in order to account for the following specifics:

        Forward:

        - If state is source, EOS is not possible, only continuous increment is valid.
        - The termination action is EOS

        Backward:

        - if done is True, only EOS is valid
        - Te termination action is back-to-source

        Parameters
        ----------
        policy_outputs : tensor
            The output of the GFlowNet policy model.
        mask : tensor
            The mask containing information about special cases.
        states_from : list
            The states originating the actions, in GFlowNet format.
        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default).
        random_action_prob : float, optional
            The probability of sampling a random action.
        temperature_logits : float, optional
            A scalar by which the model outputs are divided to temper the sampling
            distribution.
        """
        # Sample end of the trajectory (EOS or back-to-source) where it is allowed by mask
        if not is_backward:
            end_is_possible = ~mask[:, 1]
        else:
            end_is_possible = torch.all(~mask, dim=1)

        # randomise policy outputs first
        logits_sampling = self.randomize_and_temper_sampling_distribution(
            policy_outputs.clone().detach(), random_action_prob, temperature_logits
        )
        # extract end_logits for discretesampling
        end_logits_sampling = self.get_end_logits(logits_sampling[end_is_possible])

        disr_end = self.get_end_distr(end_logits_sampling)
        end_is_sampled = tbool(disr_end.sample(), device=self.device)

        if not is_backward:
            mask_incremet_invalid = mask[:, 0]
            # invalidate increments where eos is sampled
            mask_incremet_invalid[end_is_possible] = end_is_sampled
            # broadcast the mask to match the format of the ctorus env mask
            mask_super = torch.stack(
                [mask_incremet_invalid, mask_incremet_invalid], dim=1
            )
        else:
            n_states = policy_outputs.shape[0]
            do_bts = torch.full((n_states,), False, device=self.device)
            do_bts[end_is_possible] = end_is_sampled

            do_eos = torch.all(mask, dim=1)
            mask_super = torch.stack([do_bts, do_eos], dim=1)

        # Sample increments using the parent method
        actions = super().sample_actions_batch(
            policy_outputs=logits_sampling[
                :, :-1
            ],  # pass already randomised policy outputs
            mask=mask_super,
            states_from=states_from,
            is_backward=is_backward,
            random_action_prob=0.0,  # no randomisation
            temperature_logits=1.0,  # no randomisation
        )
        return actions

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: Union[List, TensorType["n_states", "action_dim"]],
        mask: TensorType["n_states", "mask_dim"],
        states_from: Optional[List] = None,
        is_backward: bool = False,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.

        Parameters
        ----------
        policy_outputs : tensor
            The output of the GFlowNet policy model.
        actions : list or tensor
            The actions (angle increments) from each state in the batch for which to
            compute the log probability.
        mask : tensor
            The mask containing information special cases.
        states_from : list
            The states originating the actions, in GFlowNet format. Used to determine
            the log probability of the first step if start from uniform.
        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default).
        """
        n_states = policy_outputs.shape[0]
        logprobs = torch.zeros(n_states, dtype=self.float, device=self.device)

        if not is_backward:
            end_is_possible = ~mask[:, 1]
        else:
            end_is_possible = torch.all(~mask, dim=1)

        end_is_sampled = tbool(torch.zeros(end_is_possible.sum()), device=self.device)

        if torch.any(end_is_possible):
            end_logits_sampling = self.get_end_logits(policy_outputs[end_is_possible])
            disr_end = self.get_end_distr(end_logits_sampling)
            actions = tfloat(actions, float_type=self.float, device=self.device)
            if not is_backward:
                eos_tensor = tfloat(self.eos, float_type=self.float, device=self.device)
                end_is_sampled = torch.all(
                    actions[end_is_possible] == eos_tensor, dim=1
                )
            else:
                source_angles = tfloat(
                    self.source_angles, float_type=self.float, device=self.device
                )
                states_from_angles = tfloat(
                    states_from, float_type=self.float, device=self.device
                )[end_is_possible]
                expected_actions_bts = (states_from_angles - source_angles) % (
                    2 * torch.pi
                )
                end_is_sampled = angles_allclose(
                    expected_actions_bts,
                    actions[end_is_possible],
                    atol=self.state_space_atol,
                )
            logprobs[end_is_possible] = disr_end.log_prob(
                tfloat(end_is_sampled, float_type=self.float, device=self.device)
            )

        if not is_backward:
            mask_incremet_invalid = mask[:, 0]
            # invalidate increments where eos is sampled
            mask_incremet_invalid[end_is_possible] = end_is_sampled
            # broadcast the mask to match the format of the ctorus env mask
            mask_super = torch.stack(
                [mask_incremet_invalid, mask_incremet_invalid], dim=1
            )
        else:
            do_bts = torch.full((n_states,), False, device=self.device)
            do_bts[end_is_possible] = end_is_sampled

            do_eos = torch.all(mask, dim=1)
            mask_super = torch.stack([do_bts, do_eos], dim=1)

        logprobs_increments = super().get_logprobs(
            policy_outputs=policy_outputs[:, :-1],
            actions=actions,
            mask=mask_super,
            states_from=states_from,
            is_backward=is_backward,
        )
        return logprobs + logprobs_increments

    def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy.

        For each dimension d of the torus and component c of the mixture, the
        output of the policy should return:
          1) the weight of the component in the mixture
          2) the location of the von Mises distribution to sample the angle increment
          3) the log concentration of the von Mises distribution to sample the angle
          increment
        The last element of the policy output vector is the logit for the bernouilli distribution
        defining the probability to end the trajectory.

        Therefore, the output of the policy model has dimensionality D x C x n_params_per_dim + 1,
        where D is the number of dimensions (self.n_dim) and C is the number of components
        (self.n_comp). The first n_params_per_dim x C entries in the policy output correspond to the
        first dimension, and so on
        """
        policy_output = torch.ones(
            self.n_dim * self.n_comp * self.n_params_per_dim + 1,
            dtype=self.float,
            device=self.device,
        )
        policy_output[1 :: self.n_params_per_dim] = params["vonmises_mean"]
        policy_output[2 :: self.n_params_per_dim] = params["vonmises_concentration"]
        policy_output[-1] = params["bernoulli_logit"]
        return policy_output

    def _step(
        self,
        action: Tuple[float],
        backward: bool,
    ) -> Tuple[List[float], Tuple[float], bool]:
        """
        Updates self.state given a non-EOS action. This method is called by both step()
        and step_backwards(), with the corresponding value of argument backward.

        Parameters
        ----------
        action : tuple
            Action to be executed. An action is a vector where the value at position d
            indicates the increment in the angle at dimension d.

        backward : bool
            If True, perform backward step. Otherwise, perform forward step.
        """
        # if source, set the values of the state to the source angles first,
        # then apply actions
        if self.is_source():
            self.state = copy(self.source_angles)

        for dim, angle in enumerate(action):
            if backward:
                self.state[int(dim)] -= angle
            else:
                self.state[int(dim)] += angle
            self.state[int(dim)] = self.state[int(dim)] % (2 * np.pi)

        if backward:
            if self.isclose(self.state, self.source_angles, atol=1e-5):
                self.state = copy(self.source)

    def step(
        self, action: Tuple[float], skip_mask_check: bool = False
    ) -> Tuple[List[float], Tuple[float], bool]:
        """
        Executes forward step given an action, if possible (i.e. the action is
        valid in the current state)

        Parameters
        ----------
        action : tuple
            Action to be executed. An action is a vector where the value at position d
            indicates the increment in the angle at dimension d.

        skip_mask_check : bool
            Ignored because the action space space is fully continuous, therefore there
            is nothing to check.

        Returns
        -------
        self.state : list
            The state after executing the action, if it was valid. Othervise,
            the current state

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state
        """
        # check validity of the action
        valid, self.state, _ = self._pre_step(
            self.action2representative(action),
            skip_mask_check=skip_mask_check,
            backward=False,
        )
        if valid:
            self.n_actions += 1
            if action == self.eos:
                self.done = True
                return self.state, action, valid
            self._step(action, backward=False)
        return self.state, action, valid

    def step_backwards(
        self, action: Tuple[float], skip_mask_check: bool = False
    ) -> Tuple[List[float], Tuple[float], bool]:
        """
        Executes backward step given an action, if possible (i.e. the action is
        valid in the current state)

        Parameters
        ----------
        action : tuple
            Action to be executed. An action is a vector where the value at position d
            indicates the increment in the angle at dimension d.

        skip_mask_check : bool
            Ignored because the action space space is fully continuous, therefore there
            is nothing to check.

        Returns
        -------
        self.state : list
            The sequence after executing the action if it was valid. Othervise,
            the current state

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        # check validity of the action
        valid, self.state, _ = self._pre_step(
            self.action2representative(action),
            skip_mask_check=skip_mask_check,
            backward=True,
        )
        if valid:
            self.n_actions += 1
            if self.done:
                self.done = False
            else:
                self._step(action, backward=True)
        return self.state, action, valid

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment, including the EOS
        action (used in the base env). As this env is non-acyclic,
        the max trajectory should be infintite, but it leads to errors in samplig
        random trajectories, so I set it just to a big number
        """
        return 100

    def states2policy(
        self,
        states: Union[List, TensorType["batch", "state_dim"]],
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: if
        policy_encoding_dim_per_angle >= 2, then the state (angles) is encoded using
        trigonometric components.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = self._source_to_angles(states)
        return super().states2policy(states, encode_step=False)

    def statse2proxy(
        self, states: Union[List[List], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in "environment format" for the proxy: each state is
        a vector of length n_dim where each value is an angle in radians. The n_actions
        item is removed.

        Parameters
        ----------
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        torch.Tensor
            A tensor of states in proxy format
        """
        states = self._source_to_angles(states)
        return super().states2proxy(states)

    def _source_to_angles(
        self, states: Union[List[List], TensorType["batch", "state_dim"]]
    ) -> Union[List[List], TensorType["batch", "state_dim"]]:
        """
        Replace source states with the source-angle values.

        Source states are identified using :meth:`is_source_batch` and replaced
        by `self.source_angles`. The input is modified in place.

        Parameters
        ----------
        states : list or torch.Tensor
            States to process


        Returns
        -------
        list or torch.Tensor
            The input states with source states replaced by
            `self.source_angles`.
        """
        states = copy(states)
        is_source = self.is_source_batch(states)
        if isinstance(states, list):
            for idx in range(len(states)):
                if is_source[idx]:
                    states[idx] = copy(self.source_angles)
        else:
            states[is_source] = self.source_angles
        return states

    # TODO: add step2readable, readable2 state
    def get_grid_terminating_states(
        self, n_states: int, n_dim: Optional[int] = None
    ) -> List[List]:
        r"""
        Samples n terminating states by sub-sampling the state space as a grid. See
        ContinuousTorus.get_grid_terminating_states for more details

        Parameters
        ----------
        n_states : int
            The number of terminating states to sample.
        n_dim : int, optional
            The number of dimensions in the state space. If None, the number of
            dimensions of the environment is used.

        Returns
        -------
        states : list
            A list of sampled terminating states.
        """
        samples = super().get_grid_terminating_states(n_states, n_dim)
        # remove the last (step) element from the samples
        for state in samples:
            state.pop()
        return samples

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None, n_dim=None
    ) -> List[List]:
        r"""
        Samples ``n_states`` terminating states uniformly in the state space. See
        ContinuousTorus.get_grid_terminating_states for more details

        Parameters
        ----------
        n_states : int
            The number of terminating states to sample.
        seed : int
            Random seed for the sampling.
        n_dim : int, optional
            The number of dimensions in the state space. If None, the number of
            dimensions of the environment is used.

        Returns
        -------
        states : list
            A list of sampled terminating states.
        """
        samples = super().get_uniform_terminating_states(n_states, n_dim)
        # remove the last (step) element from the samples
        for state in samples:
            state.pop()
        return samples

    def is_source_batch(
        self,
        states: Union[List, TensorType],
        timesteps: Optional[Union[List, TensorType]] = None,
    ) -> TensorType["n_states"]:
        """
        Check which states in a batch correspond to the source state.

        Comparing each state directly against `self.source`.

        Parameters
        ----------
        states: list or torch.Tensor
            Batch of states
        timesteps: torch.Tensor, optional
            Timestep for each state in the batch (always None for this env)

        Returns
        --------
        torch.Tensor :
            Boolean tensor of shape (batch_size,) indicating which
            states are source states.
        """
        if isinstance(states, list):
            return tbool([self.is_source(st) for st in states], device=self.device)
        else:
            # relying on the fact that source is a list of nans
            return torch.isnan(states).all(dim=1)
