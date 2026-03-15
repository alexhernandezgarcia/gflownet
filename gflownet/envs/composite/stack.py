"""
Base class for Stack environments.

Stack environments are environments which consist of sequence of multiple
sub-environments in a fixed order.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.composite.base import CompositeBase
from gflownet.utils.common import copy, tfloat


class Stack(CompositeBase):
    """
    Base class to create new environments by stacking multiple sub-environments.

    This class imposes the order specified in the creation, such that the trajectory of
    the first sub-environment needs to be completed (reach done) before the actions of
    the second sub-environment become valid, and so on and so forth.

    The Stack is a sub-class of the CompositeBase, and it thus enables the
    incorporation of constraints across sub-environments via the
    :py:meth:`~gflownet.envs.composite.base.CompositeBase._apply_constraints` method.
    In order to implement the application of constraints, Stack sub-classes must
    override:
    - :py:meth:`~gflownet.envs.composite.base.CompositeBase._apply_constraints_forward`
    - :py:meth:`~gflownet.envs.composite.base.CompositeBase._apply_constraints_backward`
    """

    def __init__(
        self,
        subenvs: Sequence[GFlowNetEnv],
        **kwargs,
    ):
        """
        Parameters
        ----------

        subenvs : Sequence[GFlowNetEnv]
            A sequence containing the ordered list of the sub-environments to be
            stacked.
        """
        self.subenvs = subenvs
        self.n_subenvs = len(self.subenvs)

        # Determine the unique environments
        (
            self.envs_unique,
            _,
            self.unique_indices,
        ) = self._get_unique_environments(self.subenvs)

        # States are represented as a dictionary with the following keys and values:
        # - Meta-data about the Stack
        #   - "_active":  The index of the currently active sub-environment, starting
        #   from 0 at the source state, up to the total number of sub-environments.
        #   - "_envs_unique": A list of indices identifying the unique environment
        #   corresponding to each subenv.
        # - States of the sub-environments, with keys the indices of the subenvs.
        # Note: "_dones" is not included because it can be inferred from the active
        # sub-environment.
        self.source = {
            "_active": -1,
            "_envs_unique": self.unique_indices,
        }
        self.source.update(
            {idx: subenv.source for idx, subenv in enumerate(self.subenvs)}
        )

        # TODO: review if needed
        # Get action dimensionality by computing the maximum action length among all
        # sub-environments, and adding 1 to indicate the sub-environment.
        self.action_dim = max([len(subenv.eos) for subenv in self.subenvs]) + 1

        # The EOS action of the Stack is EOS action of the last sub-environment
        self.eos = self._pad_action(
            self.subenvs[-1].eos, idx_unique=self.unique_indices[-1]
        )

        # TODO: review
        # Policy distributions parameters
        kwargs["fixed_distr_params"] = [
            subenv.fixed_distr_params for subenv in self.subenvs
        ]
        kwargs["random_distr_params"] = [
            subenv.random_distr_params for subenv in self.subenvs
        ]
        # Base class init
        super().__init__(**kwargs)

        # TODO: this should be turned into a property
        # The stack is continuous if any subenv is continuous
        self.continuous = any([subenv.continuous for subenv in self.subenvs])

    def _compute_mask_dim(self):
        """
        Calculates the mask dimensionality of the Stack environment.

        The mask consists of:
            - A one-hot encoding of the index of the active sub-environment.
            - The mask of the active sub-environment.

        Therefore, the dimensionality is the number of sub-environments, plus the
        maximum dimensionality of the mask of all sub-environments.

        Returns
        -------
        int
            The number of elements in the Stack masks.
        """
        mask_dim_envs_unique = [env.mask_dim for env in self.envs_unique]
        return max(mask_dim_envs_unique) + self.n_subenvs

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment, including the EOS
        action.

        Returns
        -------
        int
            The maximum trajectory length.
        """
        return sum([subenv.max_traj_length for subenv in self.subenvs])

    # TODO: review
    def get_policy_output(self, params: list[dict]) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model.

        The policy output is the concatenation of the policy outputs of all the
        sub-environments.
        """
        return torch.cat(
            [
                subenv.get_policy_output(params_subenv)
                for subenv, params_subenv in zip(self.subenvs, params)
            ]
        )

    # TODO: review
    def _get_policy_outputs_of_subenv(
        self, policy_outputs: TensorType["n_states", "policy_output_dim"], stage: int
    ):
        """
        Returns the columns of the policy outputs that correspond to the
        sub-environment indicated by stage.

        Args
        ----
        policy_outputs : tensor
            A tensor containing a batch of policy outputs. It is assumed that all the
            rows in the this tensor correspond to the same stage.

        stage : int
            Index of the sub-environment of which the corresponding columns of the
            policy outputs are to be extracted.
        """
        init_col = 0
        for stg, subenv in self.subenvs.items():
            end_col = init_col + subenv.policy_output_dim
            if stg == stage:
                return policy_outputs[:, init_col:end_col]
            init_col = end_col

    def get_mask_invalid_actions_forward(
        self, state: Optional[Dict] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the forward actions mask of the state.

        The mask of the Stack environment is the mask of the active sub-environment,
        preceded by a one-hot encoding of the index of the subenv and padded with False
        up to mask_dim. Including only the relevant mask saves memory and computation.

        If a state is passed as an argument (not ``None``) and the Stack has
        constraints, the constraints are applied before computing the mask and reset
        thereafter. This is necessary because otherwise the sub-environments may not
        have the correct attributes necessary to calculate the mask.
        """
        do_constraints = state is not None and self.has_constraints
        state = self._get_state(state)
        done = self._get_done(done)

        # Apply constraints based on the input state
        if do_constraints:
            # TODO: _apply_constraints could return a boolean variable if constraints
            # are applied
            self._apply_constraints(state=state)

        # Get active sub-environment, substate and unique environment
        active_subenv = self._get_active_subenv(state)
        state_subenv = self._get_substate(state, active_subenv)
        subenv = self._get_unique_env_of_subenv(active_subenv, state)

        # Obtain mask of substate
        mask = subenv.get_mask_invalid_actions_forward(state_subenv, done)

        # Reset constraints for self.state
        if do_constraints:
            self._apply_constraints(state=self.state)

        return self._format_mask(mask, active_subenv)

    def get_mask_invalid_actions_backward(
        self, state: Optional[Dict] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the backward actions mask of the state.

        The mask of the Stack environment is the mask of the relevant sub-environment,
        preceded by a one-hot encoding of the index of the subenv and padded with False
        up to ``mask_dim``. Including only the relevant mask saves memory and
        computation.

        The relevant sub-environment regarding the backward mask is the current
        sub-environment except if the state of the sub-environment is the subenv's
        source, in which case the mask must be the one of the preceding
        sub-environment, so as to sample its EOS action.

        There are two exceptions to the above case:
            - If ``done`` is True, in which case the current sub-environment is the last
              one and the EOS action must come from itself, not the preceding subenv.
            - If the current stage is the first sub-environment, in which case there is
              no preceding subenv.

        If a state is passed as an argument (not ``None``) and the Stack has
        constraints, the constraints are applied before computing the mask and reset
        thereafter. This is necessary because otherwise the sub-environments may not
        have the correct attributes necessary to calculate the mask.
        """
        do_constraints = state is not None and self.has_constraints
        state = self._get_state(state)
        done = self._get_done(done)

        # Apply constraints based on the input state
        if do_constraints:
            # TODO: _apply_constraints could return a boolean variable if constraints
            # are applied
            self._apply_constraints(state=state)

        # Get active sub-environment, substate and unique environment
        active_subenv = self._get_active_subenv(state)
        state_subenv = self._get_substate(state, active_subenv)
        subenv = self._get_unique_env_of_subenv(active_subenv, state)

        # Change the relevant sub-environment and set done to True if the substate is
        # the source of an intermediate sub-environment
        if active_subenv > 0 and not done and subenv.is_source(state_subenv):
            relevant_subenv = active_subenv - 1
            state_subenv = self._get_substate(state, relevant_subenv)
            subenv = self._get_unique_env_of_subenv(relevant_subenv, state)
            done = True
        else:
            relevant_subenv = active_subenv

        # Obtain mask of substate
        mask = subenv.get_mask_invalid_actions_backward(state_subenv, done)

        # Reset constraints for self.state
        if do_constraints:
            self._apply_constraints(state=self.state)

        return self._format_mask(mask, relevant_subenv)

    # TODO: rethink whether padding should be True (invalid) instead.
    def _format_mask(self, mask: List[bool], idx_subenv: int):
        """
        Formats the mask of a sub-environment into a Stack mask.

        The output format is the input mask, which corresponds to a sub-environment,
        preceded by a one-hot encoding of the index of active sub-environment and with
        False up to ``self.mask_dim``.

        Parameters
        ----------
        mask : List[bool]
            The mask of a sub-environment
        idx_subenv : int
            The index of the sub-environment to be one-hot encoded.
        """
        idx_onehot = [False] * self.n_subenvs
        idx_onehot[idx_subenv] = True
        padding = [False] * (self.mask_dim - (len(mask) + self.n_subenvs))
        return idx_onehot + mask + padding

    def get_valid_actions(
        self,
        mask: Optional[bool] = None,
        state: Optional[Dict] = None,
        done: Optional[bool] = None,
        backward: Optional[bool] = False,
    ) -> List[Tuple]:
        """
        Returns the list of non-invalid (valid, for short) actions.

        This method is overridden because the mask of a Stack of environments does not
        cover the entire action space, but only the relevant sub-environment. Therefore,
        this method calls the ``get_valid_actions()`` method of the currently relevant
        sub-environment and returns the padded actions.

        If a state is passed as an argument (not ``None``) and the Stack has
        constraints, the constraints are applied before computing the mask and reset
        thereafter. This is necessary because otherwise the sub-environments may not
        have the correct attributes necessary to calculate the mask.
        """
        do_constraints = state is not None and self.has_constraints
        state = self._get_state(state)
        done = self._get_done(done)

        # Apply constraints based on the input state
        if do_constraints:
            # TODO: _apply_constraints could return a boolean variable if constraints
            # are applied
            self._apply_constraints(state=state)

        # Get active sub-environment, substate and unique environment
        active_subenv = self._get_active_subenv(state)
        state_subenv = self._get_substate(state, active_subenv)
        subenv = self._get_unique_env_of_subenv(active_subenv, state)

        # Change the relevant sub-environment and set done to True if the substate is
        # the source of an intermediate sub-environment
        if (
            backward
            and active_subenv > 0
            and not done
            and subenv.is_source(state_subenv)
        ):
            relevant_subenv = active_subenv - 1
            state_subenv = self._get_substate(state, relevant_subenv)
            subenv = self._get_unique_env_of_subenv(relevant_subenv, state)
            done = True
        else:
            relevant_subenv = active_subenv

        if mask is not None:
            # Extract the part of the mask corresponding to the sub-environment
            # TODO: consider writing a method to do this
            mask = mask[env.n_subenvs : env.n_subenvs + subenv.mask_dim]

        # Obtain valid actions
        valid_actions = [
            env._pad_action(action, relevant_subenv)
            for action in subenv.get_valid_actions(mask, state_subenv, done, backward)
        ]

        # Reset constraints for self.state
        if do_constraints:
            self._apply_constraints(state=self.state)

        return valid_actions

    # TODO: review
    def mask_conditioning(
        self, mask: Union[List[bool], TensorType["mask_dim"]], env_cond, backward: bool
    ):
        """
        Conditions the input mask based on the restrictions imposed by a conditioning
        environment, env_cond.

        This method is overriden because the base mask_conditioning would change the
        mask unaware of the special Stack format. Therefore, this method calls the
        mask_conditioning() method of the currently relevant sub-environment and
        returns the mask with the correct Stack format.
        """
        stage = self._get_stage()
        subenv = self.subenvs[stage]
        # Extract the part of the mask corresponding to the sub-environment
        # TODO: consider writing a method to do this
        mask = mask[self.n_subenvs : self.n_subenvs + subenv.mask_dim]
        env_cond = env_cond.subenvs[stage]
        mask = subenv.mask_conditioning(mask, env_cond, backward)
        return self._format_mask(mask, stage, subenv.mask_dim)

    def get_parents(
        self,
        state: Optional[Dict] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to the input state.

        If a state is passed as an argument (not ``None``) and the Stack has
        constraints, the constraints are applied before computing the mask and reset
        thereafter. This is necessary because otherwise the sub-environments may not
        have the correct attributes necessary to calculate the mask.

        Parameters
        ----------
        state : list
            State in environment format. If not, self.state is used.
        done : bool
            Whether the trajectory is done. If None, self.done is used.
        action : tuple
            Ignored.

        Returns
        -------
        parents : list
            List of parents in state format
        actions : list
            List of actions that lead to state for each parent in parents
        """
        do_constraints = state is not None and self.has_constraints
        state = self._get_state(state)
        done = self._get_done(done)

        # If done is True, the only parent is the state itself with action EOS.
        if done:
            return [state], [self.eos]

        # Apply constraints based on the input state
        if do_constraints:
            # TODO: _apply_constraints could return a boolean variable if constraints
            # are applied
            self._apply_constraints(state=state)

        # Get active sub-environment, substate and unique environment
        active_subenv = self._get_active_subenv(state)
        state_subenv = self._get_substate(state, active_subenv)
        subenv = self._get_unique_env_of_subenv(active_subenv, state)

        # Change the relevant sub-environment and set done to True if the substate is
        # the source of an intermediate sub-environment
        if (
            backward
            and active_subenv > 0
            and not done
            and subenv.is_source(state_subenv)
        ):
            relevant_subenv = active_subenv - 1
            state_subenv = self._get_substate(state, relevant_subenv)
            subenv = self._get_unique_env_of_subenv(relevant_subenv, state)
            done = True
        else:
            relevant_subenv = active_subenv

        # Get parents of the relevant sub-environment
        parents_subenv, parent_actions = subenv.get_parents(state_subenv, done)

        # Convert subenv parents to Stack states
        parents = []
        for parent_subenv in parents_subenv:
            parent = copy(state)
            parent = self._set_active_subenv(relevant_subenv, parent)
            parent = self._set_substate(relevant_subenv, parent_subenv, parent)
            parents.append(parent)
        # Pad actions
        parent_actions = [
            self._pad_action(action, relevant_subenv) for action in parent_actions
        ]

        return parents, parent_actions

    # TODO: Review
    def _update_state(self, stage: int):
        """
        Updates the global state based on the states of the sub-environments and the
        stage passed as an argument.
        """
        return [stage] + [subenv.state for subenv in self.subenvs.values()]

    def step(
        self, action: Tuple, skip_mask_check: bool = False
    ) -> Tuple[List, Tuple, bool]:
        """
        Executes forward step given an action.

        The action is performed by the corresponding sub-environment and then the
        global state is updated accordingly. If the action is the EOS of the
        sub-environment, the stage is advanced and constraints are set on the
        subsequent sub-environment.

        Parameters
        ----------
        action : tuple
            Action to be executed. The input action is global, that is padded.

        Returns
        -------
        self.state : Dict
            The state after executing the action.
        action : int
            Action executed.
        valid : bool
            False, if the action is not allowed for the current state. True otherwise.
        """
        # If done, exit immediately
        if self.done:
            return self.state, action, False

        # Get active sub-environment, subenv and action of subenv
        active_subenv = self._get_active_subenv(state)
        subenv = self.subenvs[active_subenv]
        action_subenv = self._depad_action(action, active_subenv)

        # Perform pre-step from subenv - if it was done from the Stack env there could
        # be a mismatch between mask and action space due to continuous subenvs.
        action_to_check = subenv.action2representative(action_subenv)
        # Skip mask check if stage is continuous
        if subenv.continuous:
            skip_mask_check = True
        do_step, _, _ = subenv._pre_step(
            action_to_check,
            skip_mask_check=(skip_mask_check or self.skip_mask_check),
        )
        if not do_step:
            return self.state, action, False

        # Call step of active sub-environment
        _, action_subenv, valid = subenv.step(action_subenv)

        # If action is invalid, exit immediately.
        if not valid:
            return self.state, action, False

        # Otherwise, increment number of actions and go on
        self.n_actions += 1

        # Check if action is EOS of subenv
        if action_subenv == subenv.eos:
            # If it is global EOS set done to True
            if action == self.eos:
                self.done = True
            else:
                # Increment active subenv and apply constraints
                self._set_active_subenv(active_subenv + 1)
                self._apply_constraints(action=action, is_backward=False)
        else:
            # Update substate
            self._set_substate(active_subenv, subenv.state)

        return self.state, action, valid

    def step_backwards(
        self, action: Tuple, skip_mask_check: bool = False
    ) -> Tuple[List, Tuple, bool]:
        """
        Executes backward step given an action.

        The action is performed by the corresponding sub-environment and then the
        global state is updated accordingly. If the updated state of the
        sub-environment becomes its source, the stage is decreased.

        Parameters
        ----------
        action : tuple
            Action to be executed. The input action is global, that is padded.

        Returns
        -------
        self.state : list
            The state after executing the action.
        action : int
            Action executed.
        valid : bool
            False, if the action is not allowed for the current state. True otherwise.
        """
        # Get relevant subenv index from action (not from state), subenv and action of
        # subenv
        relevant_subenv = action[0]
        subenv = self.subenvs[relevant_subenv]
        action_subenv = self._depad_action(action, relevant_subenv)

        # If subenv of action and state are different, action must be EOS of subenv
        if relevant_subenv != self._get_active_subenv(self.state):
            assert action_subenv == subenv.eos

        # Perform pre-step from subenv - if it was done from the "superenv" there could
        # be a mismatch between mask and action space due to continuous subenvs.
        action_to_check = subenv.action2representative(action_subenv)
        # Skip mask check if stage is continuous
        if subenv.continuous:
            skip_mask_check = True
        do_step, _, _ = subenv._pre_step(
            action_to_check,
            backward=True,
            skip_mask_check=(skip_mask_check or self.skip_mask_check),
        )
        if not do_step:
            return self.state, action, False

        # Call step of current subenvironment
        _, _, valid = subenv.step_backwards(action_subenv)

        # If action is invalid, exit immediately
        if not valid:
            return self.state, action, False

        # Otherwise, increment number of actions and go on
        self.n_actions += 1

        # If action was from done, set done False
        if self.done:
            assert action == self.eos
            self.done = False

        # If action is EOS of subenv, apply backward constraints
        if action_subenv == subenv.eos:
            self._apply_constraints(action=action, is_backward=True)

        # Update substate
        self._set_substate(active_subenv, subenv.state)
        self._set_active_subenv(relevant_subenv)
        return self.state, action, valid

    # TODO: remove
    def _apply_constraints(
        self,
        action: Tuple = None,
        state: Union[List, torch.Tensor] = None,
        dones: List[bool] = None,
        is_backward: bool = None,
    ):
        """
        Applies constraints across sub-environments.

        This method is called from the methods that can modify the state, namely:
            - :py:meth:`~gflownet.envs.base.GFlowNetEnv.step()`
            - :py:meth:`~gflownet.envs.base.GFlowNetEnv.step_backwards()`
            - :py:meth:`~gflownet.envs.base.GFlowNetEnv.set_state()`
            - :py:meth:`~gflownet.envs.base.GFlowNetEnv.reset()`

        This method simply calls
        :py:meth:`~gflownet.envs.composite.stack.Stack._apply_constraints_forward`
        and/or
        :py:meth:`~gflownet.envs.composite.stack.Stack._apply_constraints_backward`.

        This method should in general not be overriden. Instead, classes inheriting the
        Stack class may override:
            - :py:meth:`~gflownet.envs.composite.stack.Stack._apply_constraints_forward`
            - :py:meth:`~gflownet.envs.composite.stack.Stack._apply_constraints_backward`

        Parameters
        ----------
        action : tuple (optional)
            An action, which can be used to determine whether which constraints should
            be applied and which should not, since the computations may be intensive.
            If the call of the method is initiated by ``set_state()`` or ``reset()``,
            then the action will be None.
        state : list or tensor (optional)
            A state that can optionally be passed to set in the environment after
            applying the constraints. This may typically be used by ``set_state()``.
        dones : list
            List of boolean values indicating the sub-environments that are done. This
            may be optionally used to set the state together with done after applying
            the constraints. This may typically be used by ``set_state()``.
        is_backward : bool
            Boolean flag to indicate whether the constraint should be applied in the
            backward direction (True), meaning 'undoing' the constraint (this is the
            value when the call method is initiated by ``step_backwards()`` or
            ``reset()``); or in the forward direction, meaning 'applying' the
            constraint (if initiated by ``step()``). If the call of the method is
            initiated by ``set_state()``, then the value is None, since the constraints
            may be applied in any direction, depending on the current state.
        """
        if not self.has_constraints:
            return

        # Forward constraints are applied if the call method is initiated by
        # set_state() (action is None and is_backward is not True) or by step() (action
        # is not None and is_backward is False)
        if (action is None and is_backward is not True) or (
            action is not None and is_backward is False
        ):
            self._apply_constraints_forward(action, state, dones)
        # Backward constraints are applied if the call method is initiated by
        # set_state() or reset() (action is None and is_backward is not False) or by
        # step_backward() (action is not None and is_backward is True)
        if (action is None and is_backward is not False) or (
            action is not None and is_backward is True
        ):
            self._apply_constraints_backward(action)

    # TODO: remove
    def _apply_constraints_forward(
        self,
        action: Tuple = None,
        state: Union[List, torch.Tensor] = None,
        dones: List[bool] = None,
    ):
        """
        Applies constraints across sub-environments in the forward direction.

        This method is called when ``step()`` and ``set_state()`` are called.

        Environments inheriting the Stack may override this method if constraints
        across sub-environments must be applied. The method
        :py:meth:`~gflownet.envs.composite.stack.Stack._do_constraints_for_stage` may
        be used as a helper to determine whether the constraints imposed by a
        sub-environment should be applied depending on the action.

        Parameters
        ----------
        action : tuple (optional)
            An action from the Stack environment. If the call of this method is
            initiated by ``set_state()``, then ``action`` is None.
        state : list or tensor (optional)
            A state from the Stack environment.
        dones : list
            A list indicating the sub-environments that are done.
        """
        pass

    # TODO: remove
    def _apply_constraints_backward(self, action: Tuple = None):
        """
        Applies constraints across sub-environments in the backward direction.

        In the backward direction, in this case, means that the constraints between two
        sub-environments are undone and reset as in the source state.

        This method is called when ``step_backwards()``, ``set_state()`` and
        ``reset()`` are called.

        Environments inheriting the Stack may override this method if constraints
        across sub-environments must be applied. The method
        :py:meth:`~gflownet.envs.composite.stack.Stack._do_constraints_for_stage` may
        be used as a helper to determine whether the constraints imposed by a
        sub-environment should be applied depending on the action.

        Parameters
        ----------
        action : tuple
            An action from the Stack environment.
        """
        pass

    # TODO: remove
    def _do_constraints_for_stage(
        self, stage: int, action: Tuple, is_backward: bool = False
    ) -> bool:
        """
        Returns True if constraints chould be applied given the stage, action and
        direction.

        This environment is meant to be be used by environments inheriting the Stack
        to determine whether the constraints imposed by a particular sub-environment
        should be applied. This depends on whether the environment is done or not,
        whether the constraints are to be done or undone, and whether they would be
        triggered by a transition or by ``set_state()`` or ``reset()``. This method is
        meant to be called from:
        - :py:meth:`~gflownet.envs.composite.stack.Stack._apply_constraints_forward`
        - :py:meth:`~gflownet.envs.composite.stack.Stack._apply_constraints_backward`

        Additionally, Stack environments may include other speciic checks before
        setting inter-environment constraints, besides the output of this method.

        Forward constraints could be applied if:
            - The condition environment is done, and
            - The action is either None or EOS
        Backward constraints could be applied if:
            - The condition environment is not done, and
            - The action is either None or EOS

        Parameters
        ----------
        stage : int
            Index of the sub-environment that would trigger constraints.
        action : tuple (optional)
            The action involved in the transition, or None if there is no transition,
            for example if the application of constraints is initiated by
            ``set_state()`` or ``reset()``.
        is_backward : bool
            Boolean flag to indicate whether the potential constraint is in the
            backward direction (True) or in the forward direction (False).
        """
        # Get relevant sub-environment from the stage and depad the action
        subenv = self.subenvs[stage]
        if action is not None:
            action = self._depad_action(action)

        # For constraints to be applied, either the action is None (meaning the call of
        # this method was initiated by set_state() or reset(), or the action is EOS
        if action is not None and action != subenv.eos:
            return False

        # Backward constraints could only be applied if the sub-environment is not done
        if is_backward:
            return not subenv.done
        # Forward constraints could only be applied if the sub-environment is done
        else:
            return subenv.done

    # TODO: review
    def set_state(self, state: List, done: Optional[bool] = False):
        """
        Sets a state and done.

        The correct state and done of each sub-environment are set too.
        """
        super().set_state(state, done)

        # Set state and done of each sub-environment
        n_done = self._get_stage(state) + int(done)
        dones = (True,) * n_done + (False,) * (self.n_subenvs - n_done)
        for (stage, subenv), done_subenv in zip(self.subenvs.items(), dones):
            subenv.set_state(self._get_substate(self.state, stage), done_subenv)

        # Apply constraints
        self._apply_constraints(state=state, dones=dones, is_backward=None)

        return self

    # TODO: review
    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "policy_output_dim"]] = None,
        states_from: List = None,
        is_backward: Optional[bool] = False,
        random_action_prob: Optional[float] = 0.0,
        temperature_logits: Optional[float] = 1.0,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs.

        This method calls the sample_actions_batch() method of the sub-environment
        corresponding to each state in the batch.

        Note that in order to call sample_actions_batch() of the sub-environments, we
        need to first extract the part of the policy outputs, the masks and the states
        that correspond to the sub-environment.
        """
        # Get the relevant stage of each mask from the one-hot prefix
        stages = torch.where(mask[:, : self.n_subenvs])[1]
        stages_int = stages.tolist()
        states_dict = {stage: [] for stage in self.subenvs.keys()}
        """
        A dictionary with keys equal to the stage indices and the values are the list
        of states in the stage of the key. The states are only the part corresponding
        to the sub-environment.
        """
        for state, stage in zip(states_from, stages_int):
            states_dict[stage].append(self._get_substate(state, stage))

        # Sample actions from each sub-environment
        actions_dict = {}
        for stage, subenv in self.subenvs.items():
            stage_mask = stages == stage
            if not torch.any(stage_mask):
                continue
            actions_dict[stage] = subenv.sample_actions_batch(
                self._get_policy_outputs_of_subenv(policy_outputs[stage_mask], stage),
                mask[stage_mask, self.n_subenvs : self.n_subenvs + subenv.mask_dim],
                states_dict[stage],
                is_backward,
                random_action_prob,
                temperature_logits,
            )

        # Stitch all actions in the right order, with the right padding
        return [
            self._pad_action(actions_dict[stage].pop(0), stage) for stage in stages_int
        ]

    # TODO: review
    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: Union[List, TensorType["n_states", "action_dim"]],
        mask: TensorType["n_states", "mask_dim"],
        states_from: List,
        is_backward: bool,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.

        Parameters
        ----------
        policy_outputs : tensor
            The output of the GFlowNet policy model.
        mask : tensor
            The mask containing information about invalid actions and special cases.
        actions : list or tensor
            The actions (global) from each state in the batch for which to compute the
            log probability.
        states_from : tensor
            The states originating the actions, in environment format.
        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default).
        """
        actions = tfloat(actions, float_type=self.float, device=self.device)
        n_states = policy_outputs.shape[0]
        # Get the relevant stage of each mask from the one-hot prefix
        stages = torch.where(mask[:, : self.n_subenvs])[1]
        stages_int = stages.tolist()
        states_dict = {stage: [] for stage in self.subenvs.keys()}
        """
        A dictionary with keys equal to Stage and the values are the list of states in
        the stage of the key. The states are only the part corresponding to the
        sub-environment.
        """
        for state, stage in zip(states_from, stages_int):
            states_dict[stage].append(self._get_substate(state, stage))

        # Compute logprobs from each sub-environment
        logprobs = torch.empty(n_states, dtype=self.float, device=self.device)
        for stage, subenv in self.subenvs.items():
            stage_mask = stages == stage
            if not torch.any(stage_mask):
                continue
            logprobs[stage_mask] = subenv.get_logprobs(
                self._get_policy_outputs_of_subenv(policy_outputs[stage_mask], stage),
                actions[stage_mask, 1 : 1 + len(subenv.eos)],
                mask[stage_mask, self.n_subenvs : self.n_subenvs + subenv.mask_dim],
                states_dict[stage],
                is_backward,
            )
        return logprobs

    def states2policy(
        self, states: List[List]
    ) -> TensorType["batch", "state_policy_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: simply
        a concatenation of the policy-format states of the sub-environments.

        Args
        ----
        states : list
            A batch of states in environment format.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        return torch.cat(
            [
                env.states2policy([state[idx] for state in states])
                for idx, env in enumerate(self.envs_unique)
            ],
            dim=1,
        )

    # TODO: review
    def states2proxy(self, states: List[List]) -> List[List]:
        """
        Prepares a batch of states in "environment format" for a proxy: simply a
        concatenation of the proxy-format states of the sub-environments.

        Args
        ----
        states : list
            A batch of states in environment format.

        Returns
        -------
        A list of lists, each containing the proxy representation of all the states in
        the Stack, for all the Stacks in the batch..
        """
        states_proxy = []
        for state in states:
            states_proxy.append(
                [
                    subenv.state2proxy(self._get_substate(state, idx))[0]
                    for idx, subenv in self.subenvs.items()
                ]
            )
        return states_proxy

    # TODO: review
    def state2readable(self, state: Optional[List[int]] = None) -> str:
        """
        Converts a state into human-readable representation. It concatenates the
        readable representations of each sub-environment, separated by "; " and
        preceded by "Stage {stage}; ".
        """
        state = self._get_state(state)
        readable = f"Stage {self._get_stage(state)}; " + "".join(
            [
                subenv.state2readable(self._get_substate(state, stage)) + "; "
                for stage, subenv in self.subenvs.items()
            ]
        )
        readable = readable[:-2]
        return readable

    # TODO: review
    def readable2state(self, readable: str) -> List[int]:
        """
        Converts a human-readable representation of a state into the standard format.
        """
        readables = readable.split("; ")
        stage = int(readables[0][-1])
        readables = readables[1:]
        return [stage] + [
            subenv.readable2state(readables[stage])
            for stage, subenv in self.subenvs.items()
        ]

    # TODO: review
    def action2representative(self, action: Tuple) -> int:
        """
        Replaces the part of the action associated with a sub-environment by its
        representative. The part of the action that identifies the sub-environment
        concerned by the action remains unaffected.
        """
        # Get relevant subenv from action (not from state), subenv and action of subenv
        relevant_subenv = action[0]
        subenv = self.subenvs[relevant_subenv]
        action_subenv = self._depad_action(action, relevant_subenv)

        # Obtain the representative from the subenv
        representative_subenv = subenv.action2representative(action_subenv)
        representative = self._pad_action(representative_subenv, relevant_subenv)
        return representative

    # TODO: review
    def is_source(self, state: Optional[List] = None) -> bool:
        """
        Returns True if the environment's state or the state passed as parameter (if
        not None) is the source state of the environment.

        This method is overriden for efficiency (for example, it would return False
        immediately if the stage is not the first stage) and to cover special uses of
        the Stack.

        Parameters
        ----------
        state : list
            None, or a state in environment format.

        Returns
        -------
        bool
            Whether the state is the source state of the environment
        """
        state = self._get_state(state)
        return self._get_stage(state) == 0 and all(
            [
                subenv.is_source(self._get_substate(state, stage))
                for stage, subenv in self.subenvs.items()
            ]
        )
