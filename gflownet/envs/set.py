"""
Classes implementing the family of Set meta-environments, which allow to combine
multiple sub-environments without any specific order.
"""

import uuid
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.composite import CompositeBase
from gflownet.utils.common import copy, tfloat, tlong


class BaseSet(CompositeBase):
    """
    Base class for the SetFlex and the SetFix classes.

    Set environments allow to combine multiple sub-environments of same of different
    type. For example, a new environment could be created by arranging a set of two
    (continuous) Cubes and a Grid.

    The SetFlex implements a Set environment with a variable number of elements
    (sub-environments), up to a pre-defined maximum. That is, trajectories may consist
    of a actions in a variable number of sub-environments from a pre-defined set of
    unique environments.

    The SetFix is a special case of the SetFlex which implements a Set with a fixed
    number of elements (sub-environments). That is, all trajectories consist of actions
    in the same set of pre-defined sub-environments.

    Set environments do not impose any order in the sub-environments, unlike in the
    Stack environment.

    For example, a Set may consist of the following 3 sub-environments:
    - 0: 2D Cube A
    - 1: 2D Cube B
    - 2: 10x10 Grid A

    Two variants are implemented that control how much the actions of sub-environments
    can alternate:

        1. Once a sub-environment is selected, the subsequent actions must be of the
        same sub-environment until its EOS action is performed. This variant is
        selected by setting ``can_alternate_subenvs`` to False.
        2. The actions of the sub-environments can be sampled in any order. In order to
        perform an action of a sub-environment, the sub-environment must be activated
        first with a special action. This variant is selected by setting
        ``can_alternate_subenvs`` to True.

    Therefore, if ``can_alternate_subenvs`` is True, the Set environment alternates
    actions that activate a sub-environment and actions from the active
    sub-environment.

    Additionally, in order to remove the ambiguity of the backward transitions, active
    sub-environments also need to be deactivated or toggled to go back to a state with
    no active sub-environment. This action are needed in the backward transitions in
    order to determine which sub-environment should perform the action.

    Finally, in order to make sure that methods work in their state-less fashion
    (without relying on self.subenvs), the state needs to contain information about
    whether sub-environments are done or not.

    All this implies that the state of a Set environment consists of:
    - The index of the active sub-environment or -1 to indicate that no sub-environment
      is active
    - A flag (toggle) to indicate whether a sub-environment action is expected, or
      whether an action to toggle a sub-environment is expected.
    - A list of flags indicating whether the sub-environments are done (1) or not (0).
    - A dictionary with the states of all the sub-environments

    The flow of actions for each of the two variants is as follows:

    1. Actions of different sub-environments cannot alternate
    (``can_alternate_subenvs`` is False)

    - s0:  (active: -1, toggle: 0, dones: [0, 0]) | action: toggle subenv 1
    - s1:  (active: 1, toggle: 0, dones: [0, 0]   | action: an action of subenv 1
    - s2:  (active: 1, toggle: 1, dones: [0, 0]   | action: an action of subenv 1
    - s3:  (active: 1, toggle: 1, dones: [0, 0]   | action: EOS action of subenv 1
    - s4:  (active: 1, toggle: 0, dones: [0, 1]   | action: toggle subenv 1
    - s5:  (active: -1, toggle: 0, dones: [0, 1]) | action: toggle subenv 0
    - s6:  (active: 0, toggle: 0, dones: [0, 1])  | action: an action of subenv 0
    - s7:  (active: 0, toggle: 1, dones: [0, 1])  | action: an action of subenv 0
    - s8:  (active: 0, toggle: 1, dones: [0, 1]   | action: EOS action of subenv 0
    - s9:  (active: 0, toggle: 0, dones: [1, 1])  | action: toggle subenv 0
    - s10: (active: -1, toggle: 0, dones: [1, 1]) | action: global EOS

    2. Actions of different sub-environments can alternate (``can_alternate_subenvs``
    is True)

    - s0:  (active: -1, toggle: 0, dones: [0, 0]) | action: toggle subenv 1
    - s1:  (active: 1, toggle: 1, dones: [0, 0]   | action: an action of subenv 1
    - s2:  (active: 1, toggle: 0, dones: [0, 0])  | action: toggle subenv 1
    - s3:  (active: -1, toggle: 0, dones: [0, 0]) | action: toggle subenv 1
    - s4:  (active: 1, toggle: 1, dones: [0, 0]   | action: an action of subenv 1
    - s5:  (active: 1, toggle: 0, dones: [0, 0])  | action: toggle subenv 1
    - s6:  (active: -1, toggle: 0, dones: [0, 0]) | action: toggle subenv 0
    - s7:  (active: 0, toggle: 1, dones: [0, 0]   | action: an action of subenv 0
    - s8:  (active: 0, toggle: 0, dones: [0, 0])  | action: toggle subenv 0
    - s9:  (active: -1, toggle: 0, dones: [0, 0]) | action: toggle subenv 1
    - s10: (active: 1, toggle: 1, dones: [0, 0]   | action: EOS action of subenv 1
    - s11: (active: 1, toggle: 0, dones: [0, 1]   | action: toggle subenv 1
    - s12: (active: -1, toggle: 0, dones: [0, 1]) | action: toggle subenv 0
    - s13: (active: 0, toggle: 1, dones: [0, 1]   | action: an action of subenv 0
    - s14: (active: 0, toggle: 0, dones: [0, 1])  | action: toggle subenv 0
    - s15: (active: -1, toggle: 0, dones: [0, 1]) | action: toggle subenv 0
    - s16: (active: 0, toggle: 1, dones: [0, 1]   | action: EOS action of subenv 0
    - s17: (active: 0, toggle: 0, dones: [1, 1]   | action: toggle subenv 0
    - s18: (active: -1, toggle: 0, dones: [1, 1]) | action: global EOS

    A potential alternative implementation would be to keep the active sub-environment
    active until a different sub-environment is selected. However, this would require
    special handling of continuous environment, since in order to calculate the
    probability of an action, we would have to mix the continuous distribution with the
    discrete distribution over the actions to activate a different sub-environment.
    """

    def __init__(
        self,
        can_alternate_subenvs=True,
        **kwargs,
    ):
        """
        Initializes the BaseSet.

        Parameters
        ----------
        can_alternate_subenvs : bool
            If True, actions of different sub-environments can alternate and each
            sub-environment action is preceded and followed by a meta-action to toggle
            the sub-environment. If False, once a sub-environment is activated, only
            actions of that sub-environment can be performed until it gets done (its
            EOS action is performed).
        """
        self.can_alternate_subenvs = can_alternate_subenvs
        # Base class init
        super().__init__(**kwargs)

    # TODO: update by using super().get_action_space(), which will require changing
    # other methods to use the correct indexing of actions
    def get_action_space(self) -> List[Tuple]:
        r"""
        Constructs list with all possible actions, including eos.

        The action space of a Set environment consists of:
            - The actions to activate specific sub-environments.
            - The EOS action.
            - The concatenation of the actions of all unique environments

        In order to make all actions the same length (required to construct batches of
        actions as a tensor), the actions are zero-padded from the back.

        In order to make all actions unique, the unique environment index is added as
        the first element of the action.

        Note that the actions of unique environments are only added once to the action
        space, regardless of how many elements of the unique environment
        (sub-environments) there are in the set. In other words, identical environments
        that are part of the Set share the actions and a given action will have an
        effect on the sub-environment that is active.

        The actions to activate a specific sub-environment are represented as:
        (-1, subenv index, ZERO-PADDING)

        See:
        - :py:meth:`~gflownet.envs.set.Set._pad_action`
        - :py:meth:`~gflownet.envs.set.Set._depad_action`
        """
        action_space = []
        # Actions to activate a sub-environment
        action_space.extend(
            [self._pad_action((idx,), -1) for idx in range(self.max_elements)]
        )
        # EOS action
        action_space += [self.eos]
        # Action space of each unique environment
        for idx in range(self.n_unique_envs):
            action_space.extend(
                [
                    self._pad_action(action, idx)
                    for action in self._get_env_unique(idx).action_space
                ]
            )
        return action_space

    # TODO: make mask prefix indicate the unique environment rather than active subenv
    def get_mask_invalid_actions_forward(
        self, state: Optional[Dict] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the forward actions mask of the state.

        The mask of the Set environment is the concatenation of the following:
        - A one-hot encoding of the index of the subenv (True at the index of the
          active environment). All False if no sub-environment is active.
        - Actual (main) mask of invalid actions:
            - The mask of the actions to activate a sub-environment, OR
            - The mask of the active sub-environment.

        The mask is False-padded from the back up to mask_dim.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        # Get active sub-environment and flag
        active_subenv = self._get_active_subenv(state)
        toggle_flag = self._get_toggle_flag(state)
        dones = self._get_dones(state)

        # Establish the case based on the active sub-environment, the toggle flag and
        # the done flags
        case_a = case_b = case_c = case_d = case_e = False
        if active_subenv == -1:
            # - Case A: no sub-environment is active: the only valid actions are to
            # toggle sub-environments or the global EOS.
            assert toggle_flag == 0
            case_a = True
        elif not self.can_alternate_subenvs and toggle_flag == 0:
            if dones[active_subenv] == 0:
                # Case B: in the variant where sub-environments cannot alternate, the
                # toggle flag is zero and the active sub-environment is not done: this
                # indicates the sub-environment has just been activated and the only
                # valid actions are those of the active sub-environment.
                case_b = True
            else:
                # Case C: in the variant where sub-environments cannot alternate, the
                # toggle flag is zero and the active sub-environment is done: this
                # indicates the sub-environment is done and the only valid action is to
                # toggle (deactivate) the active sub-environment.
                case_c = True
        elif self.can_alternate_subenvs and toggle_flag == 0:
            # Case D: in the variant where sub-environments can alternate, the toggle
            # flag is zero: this indicates a sub-environment action has been performed
            # and the only valid action is to toggle (deactivate) the active
            # sub-environment.
            case_d = True
        elif toggle_flag == 1:
            # Case E: a sub-environment is active and the toggle flag is 1: this
            # indicates that a sub-environment action is to be performed.
            assert not dones[active_subenv]
            case_e = True
        else:
            raise RuntimeError("No forward case could be established")

        # Build the mask based on the case
        if case_a:
            # The main mask is the mask of the meta-actions to toggle a
            # sub-environment. The action to activate a sub-environment is invalid
            # (True) if the sub-environment is done. The global EOS is invalid (True)
            # unless all sub-environments are done.
            mask = [bool(done) for done in dones]
            mask += [not all(mask)]
        elif case_b or case_e:
            # The main mask is the mask of the active sub-environment
            # Get subenv from unique environments. This way computing the mask does not
            # depend on self.subenvs and can be computed without setting the subenvs if
            # the state is passed.
            subenv = self._get_unique_env_of_subenv(active_subenv, state)
            state_subenv = self._get_substate(state, active_subenv)
            mask = subenv.get_mask_invalid_actions_forward(state_subenv, False)
        elif case_c or case_d:
            # The main mask is the mask of the meta-actions to toggle a
            # sub-environment, but the only valid action is to toggle the active
            # sub-environment. The global EOS is invalid (True).
            # active_subenv is set to -1, in order to force the prefix reflect that the
            # state is effectively inactive.
            mask = [True] * self.max_elements
            mask[active_subenv] = False
            mask += [True]
            active_subenv = -1
        else:
            raise RuntimeError("None of the possible forward cases is True")

        # Format mask and return
        return self._format_mask(mask, active_subenv)

    def get_mask_invalid_actions_backward(
        self, state: Optional[Dict] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the backward actions mask of the state.

        The mask of the Set environment is the concatenation of the following:
        - A one-hot encoding of the index of the subenv (True at the index of the
          active environment). All False if no sub-environment is active.
        - Actual (main) mask of invalid actions:
            - The mask of the actions to activate a sub-environment, OR
            - The mask of the active sub-environment.

        The mask is False-padded from the back up to mask_dim.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        # Get active sub-environment and flag
        active_subenv = self._get_active_subenv(state)
        toggle_flag = self._get_toggle_flag(state)
        dones = self._get_dones(state)

        # Establish the case based on the active sub-environment, the toggle flag and
        # the done flags
        case_a = case_b = case_c = case_d = case_e = case_f = False
        if active_subenv == -1:
            # - Case A: no sub-environment is active: the only valid actions are to
            # toggle sub-environments or the global EOS.
            assert toggle_flag == 0
            case_a = True
        elif not self.can_alternate_subenvs and toggle_flag == 0:
            if dones[active_subenv] == 0:
                # Case B: in the variant where sub-environments cannot alternate, the
                # toggle flag is zero and the sub-environment is not done: this
                # indicates the sub-environment is in the source state and the only
                # valid action is to toggle it.
                case_b = True
            else:
                # Case C: in the variant where sub-environments cannot alternate, the
                # toggle flag is zero and the sub-environment is done: this indicates
                # the sub-environment has just been activated (in the backward sense)
                # and the only valid is the EOS of the active sub-environment.
                case_c = True
        elif not self.can_alternate_subenvs and toggle_flag == 1:
            # Case D: in the variant where sub-environments cannot alternate, the toggle
            # flag is one: this indicates a sub-environment action (in the backward
            # sense) must be performed.
            case_d = True
        elif self.can_alternate_subenvs and toggle_flag == 0:
            # Case E: in the variant where sub-environments can alternate, the toggle
            # flag is zero: this indicates a sub-environment action (in the backward
            # sense) must be performed.
            case_e = True
        elif self.can_alternate_subenvs and toggle_flag == 1:
            # Case F: in the variant where sub-environments can alterante, a
            # sub-environment is active and the toggle flag is 1: this indicates a
            # sub-environment action (in the backward sense) has been performed and the
            # only valid action is to toggle (deactivate) the sub-environment.
            case_f = True
        else:
            raise RuntimeError("No backward case could be established")

        # Build the mask based on the case
        if case_a:
            # The main mask is the mask of the meta-actions to activate a
            # sub-environment.  The action to activate a sub-environment is invalid
            # (True) if it is the source state. The global EOS is invalid (True) unless
            # the parent Set environment's done is True. If so, all toggle actions are
            # invalid.
            assert toggle_flag == 0
            mask = [True] * self.max_elements
            if done:
                mask += [False]
            else:
                # Toggling a sub-environment is invalid if the substate is source but
                # the sub-environment is not done.
                indices_unique = self._get_unique_indices(state)
                for idx, (idx_unique, done) in enumerate(zip(indices_unique, dones)):
                    if not done and self._get_env_unique(idx_unique).is_source(
                        self._get_substate(state, idx)
                    ):
                        continue
                    else:
                        mask[idx] = False
                mask += [True]
        elif case_b or case_f:
            # The main mask is the mask of the meta-actions to toggle a
            # sub-environment, but the only valid action is to toggle the active
            # sub-environment. The global EOS is invalid.
            # active_subenv is set to -1, in order to force the prefix reflect that the
            # state is effectively inactive. EOS is invalid from this state.
            mask = [True] * self.max_elements
            mask[active_subenv] = False
            mask += [True]
            active_subenv = -1
        elif case_c or case_d or case_e:
            # The main mask is the mask of the active sub-environment
            # Get subenv from unique environments. This way computing the mask does not
            # depend on self.subenvs and can be computed without setting the subenvs if
            # the state is passed.
            subenv = self._get_unique_env_of_subenv(active_subenv, state)
            state_subenv = self._get_substate(state, active_subenv)
            done_subenv = dones[active_subenv]
            mask = subenv.get_mask_invalid_actions_backward(state_subenv, done_subenv)
        else:
            raise RuntimeError("None of the possible backward cases is True")

        # Format mask and return
        return self._format_mask(mask, active_subenv)

    # TODO
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
        mask = mask[self.max_elements : self.max_elements + subenv.mask_dim]
        env_cond = env_cond.subenvs[stage]
        mask = subenv.mask_conditioning(mask, env_cond, backward)
        return self._format_mask(mask, stage, subenv.mask_dim)

    def step(
        self, action: Tuple, skip_mask_check: bool = False
    ) -> Tuple[Dict, Tuple, bool]:
        """
        Executes forward step given an action.

        Actions may be either sub-environent actions, or set actions. If the former,
        the action is performed by the corresponding sub-environment and then the
        parent state is updated accordingly. If the latter, no sub-environment is
        involved and the changes are in the Set variables (active subenv and toggle
        flag)

        Because the same action may correspond to multiple sub-environments, the action
        will always be performed on the active sub-environment.

        After a sub-environment action, the toggle flag of the state is set to 0.
        After an action to toggle a sub-environment, the active sub-environment is
        updated accordingly.

        Parameters
        ----------
        action : tuple
            Action to be executed. The input action is global, that is padded.

        Returns
        -------
        self.state : dict
            The state after executing the action.
        action : int
            Action executed.
        valid : bool
            False, if the action is not allowed for the current state. True otherwise.
        """
        # If self.subenvs is None, raise an exception
        if self.subenvs is None:
            raise ValueError(
                "self.subenvs of the SetFlex is None. The subenvs must be set before "
                "developing a trajectory."
            )

        # If done, exit immediately
        if self.done:
            return self.state, action, False

        # Case A: the action is EOS or is an action to toggle a sub-environment
        if action[0] == -1:
            assert self._get_toggle_flag(self.state) == 0
            # Skip mask check in pre-step from base environment because the mask would
            # not match the action space
            do_step, _, _ = self._pre_step(
                action,
                skip_mask_check=True,
            )
            # Do mask check of Set actions
            # Note that this relies on the Set actions being placed first in the action
            # space
            if not skip_mask_check and not self.skip_mask_check:
                action_idx = self.action_space.index(action)
                if self._extract_core_mask(
                    self.get_mask_invalid_actions_forward(), idx_unique=-1
                )[action_idx]:
                    do_step = False

            if not do_step:
                return self.state, action, False
            self.n_actions += 1

            # If action is EOS, set done to True and return
            if action == self.eos:
                assert all([env.done for env in self.subenvs])
                self.done = True
                return self.state, action, True

            # Otherwise, it is an action to toggle a sub-environment:
            # - Update the active sub-environment of the parent Set state
            # - Toggle the flag
            # - Return
            toggled_subenv = self._depad_action(action)[0]
            if self._get_active_subenv(self.state) == -1:
                self._set_active_subenv(toggled_subenv)
                self._set_toggle_flag(1)
            else:
                assert self._get_active_subenv(self.state) == toggled_subenv
                self._set_active_subenv(-1)
                self._set_toggle_flag(0)
            return self.state, action, True

        # Case B: the action is an action from a sub-environment
        # Get the sub-environment corresponding to the action and its sub-action
        assert self._get_toggle_flag(self.state) == 1

        # Get active sub-environment and depad action
        active_subenv = self._get_active_subenv(self.state)
        assert active_subenv != -1
        idx_unique = action[0]
        assert self._get_unique_indices(self.state)[active_subenv] == idx_unique
        subenv = self.subenvs[active_subenv]
        action_subenv = self._depad_action(action, idx_unique)

        # Perform pre-step from subenv - if it was done from the Set env there could
        # be a mismatch between mask and action space due to continuous subenvs.
        action_to_check = subenv.action2representative(action_subenv)
        # Skip mask check if active sub-environment is continuous
        if subenv.continuous:
            skip_mask_check = True
        do_step, _, _ = subenv._pre_step(
            action_to_check,
            skip_mask_check=(skip_mask_check or self.skip_mask_check),
        )
        if not do_step:
            return self.state, action, False

        # Call step of current sub-environment
        _, _, valid = subenv.step(action_subenv)

        # If action is invalid, exit immediately. Otherwise increment actions and go on
        if not valid:
            return self.state, action, False
        self.n_actions += 1

        # Update (global) Set state and return
        # Note that the unique indices are not change by performing an action
        self._set_substate(active_subenv, subenv.state)
        self._set_subdone(active_subenv, subenv.done)
        self._set_active_subenv(active_subenv)
        self._set_toggle_flag(0)
        return self.state, action, valid

    def step_backwards(
        self, action: Tuple, skip_mask_check: bool = False
    ) -> Tuple[Dict, Tuple, bool]:
        """
        Executes backward step given an action.

        Actions may be either sub-environent actions, or set actions. If the former,
        the action is performed by the corresponding sub-environment and then the
        parent state is updated accordingly. If the latter, no sub-environment is
        involved and the changes are in the Set variables (active subenv and toggle
        flag)

        Because the same action may correspond to multiple sub-environments, the action
        will always be performed on the active sub-environment.

        After a sub-environment action, the toggle flag of the state, is set to 1,
        contrary to the forward step. After an action to toggle a sub-environment, the
        active sub-environment is updated accordingly.

        Parameters
        ----------
        action : tuple
            Action to be executed. The input action is global, that is padded.

        Returns
        -------
        self.state : dict
            The state after executing the action.
        action : int
            Action executed.
        valid : bool
            False, if the action is not allowed for the current state. True otherwise.
        """
        # If self.subenvs is None, raise an exception
        if self.subenvs is None:
            raise ValueError(
                "self.subenvs of the SetFlex is None. The subenvs must be set before "
                "developing a trajectory."
            )

        # Case A: the action is EOS or is an action to toggle a sub-environment
        if action[0] == -1:
            # If there is an active sub-environment but the toggle flag is 0, the
            # action cannot be a Set action, thus it is invalid
            if self._get_active_subenv() != -1 and self._get_toggle_flag() == 0:
                return self.state, action, False
            # Skip mask check in pre-step from base environment because the mask would
            # not match the action space
            do_step, _, _ = self._pre_step(
                action,
                backward=True,
                skip_mask_check=True,
            )
            # Do mask check of Set actions
            # Note that this relies on the Set actions being placed first in the action
            # space
            if not skip_mask_check and not self.skip_mask_check:
                action_idx = self.action_space.index(action)
                if self._extract_core_mask(
                    self.get_mask_invalid_actions_backward(), idx_unique=-1
                )[action_idx]:
                    do_step = False

            if not do_step:
                return self.state, action, False
            self.n_actions += 1

            # If action is EOS, set done to False and return
            if action == self.eos:
                assert self.done
                assert all([env.done for env in self.subenvs])
                self.done = False
                return self.state, action, True

            # Otherwise, it is an action to toggle a sub-environment:
            # - Update the active sub-environment of the parent Set state
            # - Toggle the flag
            # - Return
            toggled_subenv = self._depad_action(action)[0]
            if self._get_active_subenv(self.state) == -1:
                self._set_active_subenv(toggled_subenv)
                self._set_toggle_flag(0)
            else:
                assert self._get_active_subenv(self.state) == toggled_subenv
                assert self._get_toggle_flag(self.state) == 1
                self._set_active_subenv(-1)
                self._set_toggle_flag(0)
            return self.state, action, True

        # Case B: the action is an action from a sub-environment

        # If the toggle flag is not 0, then it is an invalid action
        if not self._get_toggle_flag(self.state) == 0:
            return self.state, action, False

        # Get active sub-environment and depad action
        active_subenv = self._get_active_subenv(self.state)
        assert active_subenv != -1
        idx_unique = action[0]
        assert self._get_unique_indices(self.state)[active_subenv] == idx_unique
        subenv = self.subenvs[active_subenv]
        action_subenv = self._depad_action(action, idx_unique)

        # Perform pre-step from subenv - if it was done from the Set env there could
        # be a mismatch between mask and action space due to continuous subenvs.
        action_to_check = subenv.action2representative(action_subenv)
        # Skip mask check if active sub-environment is continuous
        if subenv.continuous:
            skip_mask_check = True
        do_step, _, _ = subenv._pre_step(
            action_to_check,
            backward=True,
            skip_mask_check=(skip_mask_check or self.skip_mask_check),
        )
        if not do_step:
            return self.state, action, False

        # Call step of current sub-environment
        _, _, valid = subenv.step_backwards(action_subenv)

        # If action is invalid, exit immediately. Otherwise increment actions and go on
        if not valid:
            return self.state, action, False
        self.n_actions += 1

        # Update (global) Set state and return
        self._set_substate(active_subenv, subenv.state)
        self._set_subdone(active_subenv, subenv.done)
        self._set_active_subenv(active_subenv)
        self._set_toggle_flag(1)
        return self.state, action, valid

    # TODO: Think about the connection with permutation invariance
    def get_parents(
        self,
        state: Optional[Dict] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        Parameters
        ----------
        state : dict
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
        state = self._get_state(state)
        done = self._get_done(done)

        # If done is True, the only parent is the state itself with action EOS.
        if done:
            return [state], [self.eos]

        parents = []
        actions = []

        # Get active sub-environment and flag
        active_subenv = self._get_active_subenv(state)
        toggle_flag = self._get_toggle_flag(state)

        if active_subenv == -1:
            # Case A: no sub-environment is active: the parents of the state correspond
            # to states with the same sub-environment states but with one active
            # sub-environment, unless the sub-environment is at the source state and is
            # not done.
            assert toggle_flag == 0
            dones = self._get_dones(state)
            for idx, env in enumerate(self.subenvs):
                if dones[idx] or not env.is_source(self._get_substate(state, idx)):
                    parent = copy(state)
                    parents.append(self._set_active_subenv(idx, parent))
                    actions.append(self._pad_action((idx,), -1))
        elif toggle_flag == 1:
            # Case B: a sub-environment is active but the toggle flag is 1, indicating
            # that a sub-environment has just been activated. In this case, the only
            # parent is the same state with inactive sub-environments and toggle flag
            # 0.
            parent = copy(state)
            parent = self._set_active_subenv(-1, parent)
            parent = self._set_toggle_flag(0, parent)
            parents.append(parent)
            actions.append(self._pad_action((active_subenv,), -1))
        else:
            # Case C: a sub-environment is active and the toggle flag is 0, indicating
            # that a sub-environment action has just been performed (in the forward
            # sense): The parents are determined by the parents of the active
            # sub-environment.
            assert toggle_flag == 0
            subenv = self.subenvs[active_subenv]
            state_subenv = self._get_substate(state, active_subenv)
            done_subenv = bool(self._get_dones(state)[active_subenv])
            parents_subenv, parent_actions_subenv = subenv.get_parents(
                state_subenv, done_subenv
            )
            for p, p_a in zip(parents_subenv, parent_actions_subenv):
                parent = copy(state)
                parent = self._set_toggle_flag(1, parent)
                parent = self._set_substate(active_subenv, p, parent)
                if p_a == subenv.eos:
                    parent = self._set_subdone(active_subenv, False, parent)
                parents.append(parent)
                actions.append(
                    self._pad_action(p_a, self._get_unique_idx_of_subenv(active_subenv))
                )

        return parents, actions

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
        corresponding to each state in the batch, or samples the actions to activate a
        sub-environment for the environments with no active environment.

        Note that in order to call sample_actions_batch() of the sub-environments, we
        need to first extract the part of the policy outputs, the masks and the states
        that correspond to the sub-environment.
        """
        # Get the states in the batch with and without an active sub-environment
        is_active = torch.any(mask[:, : self.max_elements], axis=1)
        is_set = torch.logical_not(is_active)

        # Sample Set actions (to toggle a sub-environment or EOS).
        # Note that this relies on the Set actions being placed first in the action
        # space, since the super() method will select the actions by indexing the
        # action space, starting from 0.
        if any(is_set):
            actions_set = super().sample_actions_batch(
                self._get_policy_outputs_of_set_actions(policy_outputs[is_set]),
                self._extract_core_mask(mask[is_set], idx_unique=-1),
                None,
                is_backward,
                random_action_prob,
                temperature_logits,
            )

        # Get the active sub-environment of each mask from the one-hot prefix
        active_subenvs = torch.where(mask[is_active, : self.max_elements])[1]

        # If there are no states with active sub-environments, return here
        if len(active_subenvs) == 0:
            assert len(actions_set) == policy_outputs.shape[0]
            return actions_set

        active_subenvs_int = active_subenvs.tolist()
        indices_unique_int = []
        states_dict = {idx: [] for idx in range(self.n_unique_envs)}
        """
        A dictionary with keys equal to the unique environments indices and the values
        are the list of states in the subenv of the key. The states are only the part
        corresponding to the sub-environment.
        """
        idx = 0
        for state, active in zip(states_from, is_active):
            if active:
                active_subenv = active_subenvs_int[idx]
                idx_unique = self._get_unique_indices(state)[active_subenv]
                states_dict[idx_unique].append(self._get_substate(state, active_subenv))
                indices_unique_int.append(idx_unique)
                idx += 1
        indices_unique = tlong(indices_unique_int, device=self.device)

        # Sample actions from each unique environment
        actions_subenvs_dict = {}
        for idx, subenv in enumerate(self.envs_unique):
            indices_unique_mask = indices_unique == idx
            if not torch.any(indices_unique_mask):
                continue
            actions_subenvs_dict[idx] = subenv.sample_actions_batch(
                self._get_policy_outputs_of_subenv(
                    policy_outputs[is_active][indices_unique_mask], idx
                ),
                self._extract_core_mask(
                    mask[is_active][indices_unique_mask], idx_unique=idx
                ),
                states_dict[idx],
                is_backward,
                random_action_prob,
                temperature_logits,
            )
        # Stitch all environment actions in the right order, with the right padding
        actions_subenvs = []
        for idx in indices_unique_int:
            actions_subenvs.append(
                self._pad_action(actions_subenvs_dict[idx].pop(0), idx)
            )

        # Stitch all actions, both Set actions and sub-environment actions
        actions = []
        for action_is_from_subenv in is_active:
            if action_is_from_subenv:
                actions.append(actions_subenvs.pop(0))
            else:
                actions.append(actions_set.pop(0))
        return actions

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

        # Get the states in the batch with and without an active sub-environment
        is_active = torch.any(mask[:, : self.max_elements], axis=1)
        is_set = torch.logical_not(is_active)

        # Get logprobs of Set actions (to toggle a sub-environment or EOS).
        # Note that this relies on the Set actions being placed first in the action
        # space, since the super() method will select the actions by indexing the
        # action space, starting from 0. states_from is ignored so can be None.
        if any(is_set):
            logprobs_set = super().get_logprobs(
                self._get_policy_outputs_of_set_actions(policy_outputs[is_set]),
                actions[is_set],
                self._extract_core_mask(mask[is_set], idx_unique=-1),
                None,
                is_backward,
            )

        # Get the active sub-environment of each mask from the one-hot prefix
        active_subenvs = torch.where(mask[is_active, : self.max_elements])[1]

        # If there are no states with active sub-environments, return here
        if len(active_subenvs) == 0:
            assert logprobs_set.shape[0] == n_states
            return logprobs_set

        active_subenvs_int = active_subenvs.tolist()
        indices_unique_int = []
        states_dict = {idx: [] for idx in range(self.n_unique_envs)}
        """
        A dictionary with keys equal to the unique environment indices and the values
        are the list of states in the subenv of the key. The states are only the part
        corresponding to the sub-environment.
        """
        idx = 0
        for state, active in zip(states_from, is_active):
            if active:
                active_subenv = active_subenvs_int[idx]
                idx_unique = self._get_unique_indices(state)[active_subenv]
                states_dict[idx_unique].append(self._get_substate(state, active_subenv))
                indices_unique_int.append(idx_unique)
                idx += 1
        indices_unique = tlong(indices_unique_int, device=self.device)

        # Compute logprobs from each unique environment
        logprobs_subenvs = torch.empty(
            len(active_subenvs), dtype=self.float, device=self.device
        )
        for idx, subenv in enumerate(self.envs_unique):
            indices_unique_mask = indices_unique == idx
            if not torch.any(indices_unique_mask):
                continue
            logprobs_subenvs[indices_unique_mask] = subenv.get_logprobs(
                self._get_policy_outputs_of_subenv(
                    policy_outputs[is_active][indices_unique_mask], idx
                ),
                actions[is_active][indices_unique_mask, 1 : 1 + len(subenv.eos)],
                self._extract_core_mask(
                    mask[is_active][indices_unique_mask], idx_unique=idx
                ),
                states_dict[idx],
                is_backward,
            )

        # Stitch logprobs of Set actions and environment actions
        logprobs = torch.empty(n_states, dtype=self.float, device=self.device)
        if any(is_set):
            logprobs[is_set] = logprobs_set
        logprobs[is_active] = logprobs_subenvs
        return logprobs

    def _compute_mask_dim(self) -> int:
        """
        Calculates the mask dimensionality of the global Set environment.

        The mask consists of:
           - A one-hot encoding of the index of the active sub-environment.
           - Actual (main) mask of invalid actions:
               - The mask of the Set actions (activate a sub-environment and EOS), OR
               - The mask of the active sub-environment.

        Therefore, the dimensionality is the maximum number of sub-environments, plus
        the maximum dimensionality of the mask of all sub-environments or the number of
        sub-environments plus one (Set actions), whichever is larger.

        Returns
        -------
        int
            The number of elements in the Set masks.
        """
        mask_dim_subenvs = [subenv.mask_dim for subenv in self.envs_unique]
        mask_dim_set_actions = self.max_elements + 1
        return max(mask_dim_subenvs + [mask_dim_set_actions]) + self.max_elements

    def _get_toggle_flag(self, state: Optional[Dict] = None) -> int:
        """
        Returns the value of the toggle flag from the state.

        If no state is passed, ``self.state`` is used.

        The toggle flag is indicated in ``state["_toggle"]``.

        Parameters
        ----------
        state : dict
            A state of the parent Set environment.
        """
        if state is None:
            state = self.state
        return state["_toggle"]

    def _set_toggle_flag(self, toggle_flag: int, state: Optional[Dict] = None) -> Dict:
        """
        Sets the toggle flag.

        If no state is passed, ``self.state`` is used.

        The toggle flag is set in ``state["_toggle"]``.

        Parameters
        ----------
        toggle_flag : int
            Value of the toggle flag to set in the state. Must be 0 or 1.
        state : dict
            A state of the parent Set environment.

        Returns
        -------
        The updated Set state.
        """
        assert toggle_flag in [0, 1]
        if state is None:
            state = self.state
        state["_toggle"] = toggle_flag
        return state

    def action2representative(self, action: Tuple) -> Tuple:
        """
        Replaces the part of the action associated with a sub-environment by its
        representative. The part of the action that identifies the sub-environment
        concerned by the action remains unaffected.

        Parameters
        ----------
        action : tuple
            An action of the Set environment (padded)

        Returns
        -------
        tuple
            A representative of the action, re-padded as a Set action that should be in
            the action space.
        """
        # Get index of unique environmennt from action
        idx_unique = action[0]
        # If the index is -1, it is a Set action, so return
        if idx_unique == -1:
            return action
        # Otherwise, get the unique environment and depad the action
        subenv = self._get_env_unique(idx_unique)
        action_subenv = self._depad_action(action, idx_unique)
        # Obtain the representative from the unique environment
        representative_subenv = subenv.action2representative(action_subenv)
        representative = self._pad_action(representative_subenv, idx_unique)
        return representative

    def _format_mask(self, mask: List[bool], active_subenv: int):
        r"""
        Applies formatting to the mask of a sub-environment.

        The output format is the mask of the input sub-environment, preceded by a
        one-hot encoding of the index of the active sub-environment and padded with
        False up to :py:const:`self.mask_dim`.

        Parameters
        ----------
        mask : list
            The mask of a sub-environment
        active_subenv : int
            The index of the active sub-environment, or -1 if no subenv is active.
        """
        active_subenv_onehot = [False] * self.max_elements
        if active_subenv != -1:
            active_subenv_onehot[active_subenv] = True
        mask = active_subenv_onehot + mask
        padding = [False] * (self.mask_dim - len(mask))
        return mask + padding

    def _extract_core_mask(
        self,
        mask: Tuple[List, TensorType["batch_size", "mask_dim"]],
        idx_unique: int,
    ) -> Tuple[List, TensorType["batch_size", "mask_dim"]]:
        """
        Extracts the core part of the mask, that is without prefix and padding.

        Parameters
        ----------
        mask : list or tensor
            The mask of a state (list) or a batch of masks (tensor). In the latter
            case, it is assumed that all states in the batch of masks correspond to the
            same unique environment or are all in a state with only set actions valid,
            that is idx_unique is -1.
        idx_unique : int
            The index of the unique environment or -1 to indicate that the mask
            corresponds to set actions (toggle and EOS).
        """
        if idx_unique == -1:
            mask_dim = self.max_elements + 1
        else:
            mask_dim = self._get_env_unique(idx_unique).mask_dim
        if isinstance(mask, list):
            return mask[self.max_elements : self.max_elements + mask_dim]
        else:
            assert torch.is_tensor(mask)
            return mask[:, self.max_elements : self.max_elements + mask_dim]

    def get_valid_actions(
        self,
        mask: Optional[bool] = None,
        state: Optional[Dict] = None,
        done: Optional[bool] = None,
        backward: Optional[bool] = False,
    ) -> List[Tuple]:
        """
        Returns the list of non-invalid (valid, for short) according to the mask of
        invalid actions.

        This method is overridden because the mask of a Set of environments does not
        cover the entire action space, but only the relevant sub-environment or the
        toggle actions, depending on the state. Therefore, this method calls the
        get_valid_actions() method of the active sub-environment or retrieves the valid
        toggle actions and returns the padded actions.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        active_subenv = self._get_active_subenv(state)
        toggle_flag = self._get_toggle_flag(state)

        if mask is None:
            mask = self.get_mask(state, done, backward)

        # Set active environment and idx_unique to -1 if the mask contains no active
        # environment
        if not any(mask[: self.max_elements]):
            active_subenv = -1
            idx_unique = -1
        else:
            # Otherwise, get index of unique environment from state
            idx_unique = self._get_unique_indices(state)[active_subenv]

        # Extract core mask
        mask = self._extract_core_mask(mask, idx_unique)

        if active_subenv == -1:
            # Case A: the only valid actions are Set actions.
            # Note that this relies on the Set actions being placed first in the action
            # space, since the super() method will select the actions by indexing the
            # action space, starting from 0.
            return super().get_valid_actions(mask, state, done, backward)

        # Case B: the only valid actions are sub-environment actions, which are
        # retrieved from the active sub-environment and padded before returning them.
        assert active_subenv != -1
        subenv = self._get_unique_env_of_subenv(active_subenv, state)
        state_subenv = self._get_substate(state, active_subenv)
        done = bool(self._get_dones(state)[active_subenv])
        return [
            self._pad_action(action, idx_unique)
            for action in subenv.get_valid_actions(mask, state_subenv, done, backward)
        ]

    def get_policy_output(self, params: list[dict]) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model.

        The policy output is the concatenation of the policy outputs corresponding to
        the Set actions (actions to activate a sub-environment and EOS) and the policy
        outputs of the unique environments.
        """
        policy_outputs_set_actions = torch.ones(
            self.max_elements + 1, dtype=self.float, device=self.device
        )
        policy_outputs_subenvs = torch.cat(
            [
                self._get_env_unique(idx).get_policy_output(params[idx])
                for idx in range(self.n_unique_envs)
            ]
        )
        return torch.cat((policy_outputs_set_actions, policy_outputs_subenvs))

    def _get_policy_outputs_of_set_actions(
        self, policy_outputs: TensorType["n_states", "policy_output_dim"]
    ):
        """
        Returns the columns of the policy outputs that correspond to the Set actions:
        toggle actions and EOS.

        Args
        ----
        policy_outputs : tensor
            A tensor containing a batch of policy outputs. It is assumed that all the
            rows in the this tensor correspond to actions to activate a sub-environemnt.
        """
        return policy_outputs[:, : self.max_elements + 1]

    def _get_policy_outputs_of_subenv(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        idx_unique: int,
    ):
        """
        Returns the columns of the policy outputs that correspond to the
        sub-environment indicated by idx_subenv.

        Args
        ----
        policy_outputs : tensor
            A tensor containing a batch of policy outputs. It is assumed that all the
            rows in the this tensor correspond to the same unique environment.

        idx_unique : int
            Index of the unique environment of which the corresponding columns of the
            policy outputs are to be extracted.
        """
        init_col = self.max_elements + 1
        for idx in range(self.n_unique_envs):
            end_col = init_col + self._get_env_unique(idx).policy_output_dim
            if idx == idx_unique:
                return policy_outputs[:, init_col:end_col]
            init_col = end_col

    def is_source(self, state: Optional[Dict] = None) -> bool:
        """
        Returns True if the environment's state or the state passed as parameter (if
        not None) is the source state of the environment.

        This method is overriden for efficiency (for example, it would return False
        immediately if the meta-data part of the state is not the source's) and to
        cover special uses of the Set.

        Parameters
        ----------
        state : dict
            None, or a state in environment format.

        Returns
        -------
        bool
            Whether the state is the source state of the environment
        """
        state = self._get_state(state)
        substates = self._get_substates(state)
        n_subenvs = len(substates)
        n_left = self.max_elements - n_subenvs
        return (
            self._get_active_subenv(state) == -1
            and self._get_toggle_flag(state) == 0
            and self._get_dones(state) == [0] * n_subenvs + [1] * n_left
            and self._get_unique_indices(state, False)[n_subenvs:] == [-1] * n_left
            and all(
                [
                    self._get_unique_env_of_subenv(idx, state).is_source(substate)
                    for idx, substate in enumerate(substates)
                ]
            )
        )


class SetFix(BaseSet):
    """
    Base class to create new environments by arranging (fixed) sets of multiple
    environments.

    Unlike the SetFlex meta-environment, the SetFix works with a fixed set of
    sub-environments. That is, all trajectories will perform actions from all the
    sub-environments defined at the initialization of the environment.

    A practical use case of the SetFix environment is the creation of an environment
    comprising a set of an arbitrary number N of Cube environments, representing for
    example points in an Euclidean space. The actions of the different Cubes can be
    sampled in any order by first sampling the action that selects the corresponding
    Cube. While the trajectories will be longer because of actions to select the
    specific Cube, all Cubes will share the same action space (and mask, and policy
    output), which is desirable since all of them represent the same kind of object.

    Note that the SetFix (as well as the SetFlex) also admits diverse environments in
    the set (for example, Cubes and Grids).
    """

    def __init__(
        self,
        subenvs: Iterable[GFlowNetEnv],
        **kwargs,
    ):
        """
        Parameters
        ----------
        subenvs : iterable
            An iterable containing the set of the sub-environments.
        """
        self.subenvs = tuple(subenvs)
        self.n_subenvs = len(self.subenvs)
        self.max_elements = self.n_subenvs

        # Determine the unique environments
        (
            self.envs_unique,
            _,
            self.unique_indices,
        ) = self._get_unique_environments(self.subenvs)
        self._n_unique_envs = len(self.envs_unique)

        # States are represented as a dictionary with the following keys and values:
        # - Meta-data about the Set
        #   - "_active":  The index of the currently active sub-environment, or -1 if
        #   none is active.
        #   - "_toggle": A flag indicating whether a sub-environment is active before
        #   (1) or after (0) a sub-environment action (in the forward sense).
        #   - "_done": A list of flags indicating whether the sub-environments are done
        #   (1) or not (0).
        #   - "_envs_unique": A list of indices identifying the unique environment
        #   corresponding to each subenv. -1 if there is no sub-environment in that
        #   position. All -1 in the source.
        # - States of the sub-environments, with keys the indices of the subenvs.
        # The only meta-data key specific to the Set (not part of composite
        # environments by default) is "_toggle".
        self.source = {
            "_active": -1,
            "_toggle": 0,
            "_dones": [0] * self.max_elements,
            "_envs_unique": self.unique_indices,
        }
        self.source.update(
            {idx: subenv.source for idx, subenv in enumerate(self.subenvs)}
        )

        # Get action dimensionality by computing the maximum action length among all
        # sub-environments, and adding 1 to indicate the sub-environment.
        self.action_dim = max([len(subenv.eos) for subenv in self.subenvs]) + 1

        # EOS is a tuple of -1's
        self.eos = (-1,) * self.action_dim

        # Policy distributions parameters
        kwargs["fixed_distr_params"] = [
            subenv.fixed_distr_params for subenv in self.subenvs
        ]
        kwargs["random_distr_params"] = [
            subenv.random_distr_params for subenv in self.subenvs
        ]
        # Base class init
        super().__init__(**kwargs)

        # The set is continuous if any subenv is continuous
        self.continuous = any([subenv.continuous for subenv in self.subenvs])

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment, including the EOS
        action.
        """
        return sum([subenv.max_traj_length for subenv in self.subenvs]) * 3 + 1

    def reset(self, env_id: Union[int, str] = None):
        """
        Resets the environment by resetting the sub-environments.
        """
        for subenv in self.subenvs:
            subenv.reset()
        super().reset(env_id=env_id)
        return self

    def set_state(self, state: List, done: Optional[bool] = False):
        """
        Sets a state and done.

        The correct state and done of each sub-environment are set too.

        Parameters
        ----------
        state : list
            A state of the parent Set environment.

        done : bool
            Whether the trajectory of the environment is done or not.
        """
        # If done is True, then the done flags in the set should all be 1
        dones = [bool(el) for el in self._get_dones(state)]
        if done:
            assert all(dones)

        super().set_state(state, done)
        # Set state and done of each sub-environment
        for idx, (subenv, done_subenv) in enumerate(zip(self.subenvs, dones)):
            subenv.set_state(self._get_substate(self.state, idx), done_subenv)

        return self

    # TODO: The current representation is not permutation invariant. In order to
    # properly let a GFlowNet on a Set environment, the representation should be
    # invariant to the permutation of states from the same environment. For example the
    # state of a Set of M d-dimensional points should be represented such that the
    # permutation of the points is invariant. A dummy representation could rely on the
    # randomisation of the subsets of sub-environments that correspond to the same
    # unique environment.
    def states2policy(
        self, states: List[List]
    ) -> TensorType["batch", "state_policy_dim"]:
        """
        Prepares a batch of states in environment format for the policy model.

        The default policy representation is the concatenation of the following
        elements:
        - One-hot encoding of the active sub-environment
        - Toggle flag
        - Done flag of each sub-environment
        - A concatenation of the policy-format states of the sub-environments

        Parameters
        ----------
        states : list
            A batch of states in environment format.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        n_states = len(states)

        # Obtain torch tensors for the active subenvironments, the toggle flags and
        # the done indicators
        active_subenvs = torch.zeros((n_states, self.n_subenvs), dtype=self.float)
        dones = []
        for idx, state in enumerate(states):
            active_subenv = self._get_active_subenv(state)
            if active_subenv != -1:
                active_subenvs[idx, active_subenv] = 1.0
            dones.append(self._get_dones(state))
        dones = torch.tensor(dones, dtype=self.float)

        # Obtain the torch tensor containing the toggle flags
        toggle_flags = torch.tensor(
            [self._get_toggle_flag(s) for s in states], dtype=self.float
        ).reshape(
            (-1, 1)
        )  # reshape to (n_states, 1)

        # Obtain the torch tensor containing the states2policy of the sub-environments
        substates = []
        for idx_subenv, subenv in enumerate(self.subenvs):
            # Collect all substates for the current subenv
            subenv_states = [self._get_substate(s, idx_subenv) for s in states]

            # Convert the subenv_states to policy format
            substates.append(subenv.states2policy(subenv_states))
        substates = torch.cat(substates, dim=1)

        return torch.cat([active_subenvs, toggle_flags, dones, substates], dim=1)

    def states2proxy(
        self, states: List[List]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in environment format for a proxy.

        The default proxy format is similar to the environment format, except that the
        states of the sub-enviroments in the dictionary are converted into their
        corresponding proxy formats.

        Parameters
        ----------
        states : list
            A batch of states in environment format.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states_proxy = copy(states)
        for state in states_proxy:
            for idx, subenv in enumerate(self.subenvs):
                self._set_substate(
                    idx, subenv.state2proxy(self._get_substate(state, idx))[0], state
                )
        return states_proxy

    def state2readable(self, state: Optional[List[int]] = None) -> str:
        """
        Converts a state into human-readable representation.

        It concatenates the readable representations of each sub-environment, separated
        by "; " and preceded by Set meta-data: active sub-environment and toggle flag.
        If a sub-environment is done, it is indicanted with " | done" after the state.

        Parameters
        ----------
        state : list
            A state in environment format.

        Returns
        -------
        str
            The state in readable format.
        """

        def _done2str(done: bool):
            """
            Converts a boolean done variable into a string suitable for the readable
            representation.

            Parameters
            ----------
            done : bool
                A boolean variable indicating whether a trajectory is done.

            Returns
            -------
            str
                " | done" if done is True; "" otherwise.
            """
            if done:
                return " | done"
            else:
                return ""

        if state is None:
            state = self.state
        dones = self._get_dones(state)
        readable = (
            f"Active subenv {self._get_active_subenv(state)}; "
            + f"Toggle flag {self._get_toggle_flag(state)};\n"
            + "".join(
                [
                    subenv.state2readable(self._get_substate(state, idx))
                    + _done2str(dones[idx])
                    + ";\n"
                    for idx, subenv in enumerate(self.subenvs)
                ]
            )
        )
        readable = readable[:-2]
        return readable

    def readable2state(self, readable: str) -> List[int]:
        """
        Converts a human-readable representation of a state into the standard format.
        """
        state = copy(self.source)
        readables = readable.split(";")
        self._set_active_subenv(int(readables[0].split(" ")[-1]), state)
        self._set_toggle_flag(int(readables[1].split(" ")[-1]), state)
        readables = [readable.strip() for readable in readables[2:]]
        for idx, (subenv, readable) in enumerate(zip(self.subenvs, readables)):
            self._set_substate(
                idx, subenv.readable2state(readable.split(" | ")[0]), state
            )
            self._set_subdone(idx, " | done" in readable, state)
        return state


class SetFlex(BaseSet):
    """
    Base class to create new environments by arranging a variable number (up to a
    maximum) of sub-environments.

    While the (more basic) Set environment is limited to a pre-defined set of
    sub-environments, the SetFlex is a more flexible set. In particular, one can define
    the following properties:
        - The maximum number of elements (sub-environments) in the SetFlex
        - The possible elements (different types of sub-environments) that can be part
          of the SetFlex.

    This flexible implementation enables training a GFlowNet that considers
    environments (SetFlex) whose constituents are a variable number of
    sub-environments.

    For example, we can consider sets of points in an Euclidean space, such as the
    environment Points. Each point is modelled by a ContinuousCube sub-environment. If
    we use the (more basic) Set environment, we need to define the (fixed) number of
    points of the set. Alternatively, we could use the SetFlex to sample a variable
    number of points, from 1 to  self.max_elements.
    """

    def __init__(
        self,
        max_elements: int,
        envs_unique: Iterable[GFlowNetEnv] = None,
        subenvs: Iterable[GFlowNetEnv] = None,
        do_random_subenvs: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        max_elements : int
            The maximum number of enviroments that can be included in the set. Note
            that this number does not refer to the number of unique environments, but
            the number of elements (instances of a sub-environment) that can form a
            set. For example, a SetFlex of up to 10 2D points would contain a single
            unique environment (2D ContinuousCube) with max_elements equal to 10.
        envs_unique : iterable
            An iterable containing the set of unique environments that can make part of
            the set. This iterable is meant to contain unique environments, unique
            meaning that the both the type and the action space are unique.  "Repeated"
            sub-environmens are discarded. If it is None, the unique environments may
            be determined from the argument subenvs.
        subenvs : iterable
            An iterable used to initialize the SetFlex with a specific set of
            sub-environments. This list of environments plays the role of a condition
            for a specific trajectory. If it is None, the set of sub-environments for a
            trajectory may be set via
            :py:meth:`~gflownet.envs.set.SetFlex.set_subenvs`.
        do_random_subenvs : bool
            If True, the environment is initialized with a set of random
            sub-environments. If True, also the reset method will resample the set of
            sub-environments. First, the number of elements is sampled uniformly from 1
            to max_elements, then the set of sub-environments is sampled with
            replacement from the set of unique environments. This is practical for
            testing purposes.
        """
        # If both envs_unique and subenvs are None, the environment cannot be
        # initialized
        if envs_unique is None:
            if subenvs is None:
                raise ValueError(
                    "Both envs_unique and subenvs are None. At least one of the "
                    "two variables must contain a set of environments."
                )
            else:
                # Determine the unique environments from the list of sub-environments
                # to determine a trajectory.
                envs_unique = subenvs
        # Determine the unique environments
        (
            self.envs_unique,
            self.envs_unique_keys,
            _,
        ) = self._get_unique_environments(envs_unique)
        self.max_elements = max_elements
        self._n_unique_envs = len(self.envs_unique)
        self.do_random_subenvs = do_random_subenvs

        # Allocate a cache for env instances of each of the unique environments. These
        # instance pools are used in the event of a call to
        # get_env_instances_by_unique_indices() to avoid performing env copies every
        # time these methods are called.
        self.envs_unique_cache = {idx: [] for idx in range(self.n_unique_envs)}

        # States are represented as a dictionary with the following keys and values:
        # - Meta-data about the Set
        #   - "_active":  The index of the currently active sub-environment, or -1 if
        #   none is active.
        #   - "_toggle": A flag indicating whether a sub-environment is active before
        #   (1) or after (0) a sub-environment action (in the forward sense).
        #   - "_done": A list of flags indicating whether the sub-environments are done
        #   (1) or not (0). The flag of environments that do not correspond to a subenv
        #   is set to 1. In the source state, the list is set to all 1s.
        #   - "_envs_unique": A list of indices identifying the unique environment. -1
        #   if there is no sub-environment in that position. All -1 in the source.
        # - States of the sub-environments, with keys the indices of the subenvs. In
        # the source state, there is not any.
        # The only meta-data key specific to the Set (not part of composite
        # environments by default) is "_toggle".
        self.source = {
            "_active": -1,
            "_toggle": 0,
            "_dones": [1] * self.max_elements,
            "_envs_unique": [-1] * self.max_elements,
        }

        # Set sub-environments
        # - If subenvs is not None, set them as sub-environments
        # - If do_random_subenvs is True, sample a random set of sub-environments
        # - If subenvs is None and do_random_subenvs is False, set the unique
        # environments as sub-environments.
        if subenvs is not None:
            self.set_subenvs(subenvs)
        elif self.do_random_subenvs:
            subenvs = self._sample_random_subenvs()
            self.set_subenvs(subenvs)
        else:
            self.subenvs = None

        # Get action dimensionality by computing the maximum action length among all
        # sub-environments, and adding 1 to indicate the sub-environment.
        self.action_dim = max([len(subenv.eos) for subenv in self.envs_unique]) + 1

        # EOS is a tuple of -1's
        self.eos = (-1,) * self.action_dim

        # Policy distributions parameters
        kwargs["fixed_distr_params"] = [
            subenv.fixed_distr_params for subenv in self.envs_unique
        ]
        kwargs["random_distr_params"] = [
            subenv.random_distr_params for subenv in self.envs_unique
        ]
        # Base class init
        super().__init__(**kwargs)

        # The set is continuous if any subenv is continuous
        self.continuous = any([subenv.continuous for subenv in self.envs_unique])

    def _compute_unique_indices_of_subenvs(
        self, subenvs: Iterable[GFlowNetEnv]
    ) -> List[int]:
        """
        Identifies the unique environment corresponding to each sub-environment in
        subenvs and returns the list of unique indices.

        Parameters
        ----------
        subenvs : iterable
            Set of sub-environments to be matched with the unique environments.

        Returns
        -------
        list
           A list of indices of the unique environments, with the same length as
           subenvs.
        """
        indices_unique = []
        for env in subenvs:
            try:
                indices_unique.append(
                    self.envs_unique_keys.index((type(env), tuple(env.action_space)))
                )
            except:
                raise ValueError(
                    "The list of subenvs contains a sub-environment that could not "
                    "be matched to one of the existing unique environments"
                )
        return indices_unique

    def _sample_random_subenvs(self) -> List[GFlowNetEnv]:
        """
        Samples randomly the unique indices of a set of sub-environments.

        First, the number of elements is sampled uniformly from 1 to self.max_elements.
        Then, the set of unique indices is sampled with replacement from the set of
        unique environments.

        This method can be practical for testing purposes.

        Returns
        -------
        list
            A list of sub-environments, each with a unique id.
        """
        n_subenvs = np.random.randint(low=1, high=self.max_elements + 1)
        indices_unique = np.random.choice(
            a=self.n_unique_envs, size=n_subenvs, replace=True
        )
        subenvs = self.get_env_instances_by_unique_indices(indices_unique)
        return subenvs

    def set_subenvs(self, subenvs: Iterable[GFlowNetEnv]):
        """
        Sets the sub-environments of the Set and applies the rest of necessary changes
        to the environment.

        - Sets self.subenvs
        - Determines the indices of the unique environments for each subenv.
        - Sets self.state by setting the correct dones, unique indices source states of
          the sub-envs.

        The sub-environments can be thought as the conditioning variables of a specific
        trajectory, since the specific sub-environments are expected to be variable in
        a SetFlex, unlike in the (simpler) Set.

        Parameters
        ----------
        subenvs : iterable
            The list of sub-environments to condition a trajectory.
        """
        self.subenvs = tuple(subenvs)
        n_subenvs = len(self.subenvs)
        # Obtain indices of unique environments and pad with -1's.
        unique_indices = self._compute_unique_indices_of_subenvs(self.subenvs)
        unique_indices += [-1] * (self.max_elements - n_subenvs)
        # Set done of sub-environments to 0 and pad with 1's.
        dones = [0] * n_subenvs + [1] * (self.max_elements - n_subenvs)
        # Set self.state
        self.state = {
            "_active": -1,
            "_toggle": 0,
            "_dones": dones,
            "_envs_unique": unique_indices,
        }
        self.state.update({idx: subenv.source for idx, subenv in enumerate(subenvs)})

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment, including the EOS
        action.

        The maximum length of a trajectory is the maximum of the maximum trajectory
        lengths of the unique environments, times 3 because each action requires
        two Set actions, times :py:const:`self.max_elements`, plus one (EOS).

        Returns
        -------
        int
            The maximum possible length of a trajectory.
        """
        return (
            max([subenv.max_traj_length for subenv in self.envs_unique])
            * 3
            * self.max_elements
            + 1
        )

    def reset(self, env_id: Union[int, str] = None):
        """
        Resets the environment by resetting the sub-environments.

        If self.do_random_subenvs is True, the set of sub-environments is re-sampled.
        """
        if self.do_random_subenvs:
            subenvs = self._sample_random_subenvs()
        elif self.subenvs is not None:
            for subenv in self.subenvs:
                subenv.reset()
            subenvs = self.subenvs
        else:
            subenvs = None

        # If subenv is None, simply call super()'s reset.
        if subenvs is None:
            super().reset(env_id=env_id)
        # Otherwise, reset the environment manually and call set_subenvs. super()'s
        # reset is not called to avoid setting self.state = copy(self.source).
        else:
            self.set_subenvs(subenvs)
            self.n_actions = 0
            self.done = False
            if env_id is None:
                self.id = str(uuid.uuid4())
            else:
                self.id = env_id
        return self

    def get_env_instances_by_unique_indices(self, unique_indices: List):
        """
        Returns a list of env instances corresponding to the requested unique
        environments. The instances have already been reset and their ID set.

        Parameters
        ----------
        unique_indices : list
            Indices of the unique environments

        Returns
        -------
        A list containing instances of the requested environments
        """

        # Allocate counter of how many instances of each unique env have been used
        # to fulfill the request
        envs_counter = {idx: 0 for idx in self.envs_unique_cache.keys()}

        # Go through requested envs, only making new env copies if there aren't already
        # enough in the cache.
        envs = []
        for idx, idx_unique in enumerate(unique_indices):
            # If too few instances of the requested unique env are available, create
            # one more.
            env_instances_used = envs_counter[idx_unique]
            env_instances_available = len(self.envs_unique_cache[idx_unique])
            if env_instances_available <= env_instances_used:
                # Allocate new env instance and add it to the cache
                new_env_instance = self._get_env_unique(idx_unique).copy()
                self.envs_unique_cache[idx_unique].append(new_env_instance)

            # Use one available instance of the requested unique env
            selected_instance = self.envs_unique_cache[idx_unique][env_instances_used]
            selected_instance.reset().set_id(idx)
            envs.append(selected_instance)
            envs_counter[idx_unique] += 1

        return envs

    def set_state(self, state: Dict, done: Optional[bool] = False):
        """
        Sets a state and done.

        It also sets the sub-environments as specified in the unique indices of the
        state by making copies of the unique environments.

        Parameters
        ----------
        state : dict
            A state of the Set environment.
        done : bool
            Whether the trajectory of the environment is done or not.
        """
        # Obtain the sub-environments from the unique indices from the state
        unique_indices = self._get_unique_indices(state)
        subenvs = self.get_env_instances_by_unique_indices(unique_indices)

        # Set sub-environments
        self.set_subenvs(subenvs)

        # If done is True, then the done flags in the set should all be 1
        dones = [bool(el) for el in self._get_dones(state)]
        if done:
            assert all(dones)

        # Call set_state from the parent to set the global state
        super().set_state(state, done)

        # Set state and done of each sub-environment
        for idx, (subenv, done_subenv) in enumerate(zip(self.subenvs, dones)):
            subenv.set_state(self._get_substate(self.state, idx), done_subenv)

        return self

    # TODO: This method is currently implemented with a basic (constant) representation
    # of the states. In order to learn a GFlowNet invariant to permutations of the
    # elements of the Set, the representation needs to be invariant to permutation of
    # the elements.  A simple but potentially effective representation could involve
    # randomly permuting the elements of the set in the policy representation.
    def states2policy(
        self, states: List[Dict]
    ) -> TensorType["batch", "state_policy_dim"]:
        """
        Prepares a batch of states in environment format for the policy model.

        The default policy representation is the concatenation of the following
        elements:
        - One-hot encoding of the active sub-environment
        - Toggle flag
        - Done flag of each sub-environment
        - A vector indicating the number of sub-environments of each unique environment
          present in the state.
        - For each unique environment:
            - A concatenation of the policy-format states of the sub-environments,
              padded up to self.max_elements with the policy representation of the
              source states.

        Parameters
        ----------
        states : list
            A batch of states in environment format.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        n_states = len(states)

        # Obtain torch tensors for the active sub-environments, the toggle flags,
        # the done indicators and the number of subenvs per unique environment
        active_subenvs = torch.zeros((n_states, self.max_elements), dtype=self.float)
        dones = []
        n_subenvs_per_unique_env = []
        for idx_state, state in enumerate(states):
            active_subenv = self._get_active_subenv(state)
            if active_subenv != -1:
                active_subenvs[idx_state, active_subenv] = 1.0
            dones.append(self._get_dones(state))
            n_subenvs = np.zeros(self.n_unique_envs)
            indices, counts = np.unique(
                self._get_unique_indices(state), return_counts=True
            )
            if len(indices) != 0:
                n_subenvs[indices] = counts
            n_subenvs_per_unique_env.append(n_subenvs.tolist())
        dones = torch.tensor(dones, dtype=self.float)
        n_subenvs_per_unique_env = torch.tensor(
            n_subenvs_per_unique_env, dtype=self.float
        )

        # Obtain the torch tensor containing the toggle flags
        toggle_flags = torch.tensor(
            [self._get_toggle_flag(s) for s in states], dtype=self.float
        ).reshape(
            (-1, 1)
        )  # reshape to (n_states, 1)

        # Initialize the policy representation of the states with self.max_elements
        # source states in their policy representation per unique environment.
        substates = torch.tile(
            torch.cat(
                [
                    subenv.state2policy(subenv.source).tile((self.max_elements,))
                    for subenv in self.envs_unique
                ],
                dim=0,
            ),
            (n_states, 1),
        )

        # Obtain the policy representation of the states that are present.
        for idx_state, state in enumerate(states):
            indices_unique = np.array(self._get_unique_indices(state))
            # Obtain the states of each unique environment
            offset = 0
            for idx_unique in range(self.n_unique_envs):
                subenv = self._get_env_unique(idx_unique)
                indices = np.where(indices_unique == idx_unique)[0]
                if len(indices) == 0:
                    offset += subenv.policy_input_dim * self.max_elements
                    continue
                substates_idx = subenv.states2policy(
                    [self._get_substate(state, idx) for idx in indices]
                ).flatten()
                substates[idx_state, offset : offset + substates_idx.shape[0]] = (
                    substates_idx
                )
                offset += subenv.policy_input_dim * self.max_elements

        return torch.cat(
            [active_subenvs, toggle_flags, dones, n_subenvs_per_unique_env, substates],
            dim=1,
        )

    # TODO: this implementation may be useles for flexible sets.
    def states2proxy(
        self, states: List[Dict]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in environment format for a proxy.

        The default proxy format is similar to the environment format, except that the
        states of the sub-enviroments in the dictionary are converted into their
        corresponding proxy formats.

        Parameters
        ----------
        states : list
            A batch of states in environment format.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states_proxy = copy(states)
        for state in states_proxy:
            for idx, substate in enumerate(self._get_substates(state)):
                subenv = self._get_unique_env_of_subenv(idx, state)
                self._set_substate(idx, subenv.state2proxy(substate)[0], state)
        return states_proxy

    def state2readable(self, state: Optional[Dict] = None) -> str:
        """
        Converts a state into human-readable representation.

        It concatenates the readable representations of each sub-environment preceded
        by the index of its unique environment (idx: ...), separated by "; " and all
        preceded by Set meta-data: active sub-environment and toggle flag.  If a
        sub-environment is done, it is indicanted with " | done" after the state.

        Parameters
        ----------
        state : dict
            A state in environment format.

        Returns
        -------
        str
            The state in readable format.
        """

        def _done2str(done: bool):
            """
            Converts a boolean done variable into a string suitable for the readable
            representation.

            Parameters
            ----------
            done : bool
                A boolean variable indicating whether a trajectory is done.

            Returns
            -------
            str
                " | done" if done is True; "" otherwise.
            """
            if done:
                return " | done"
            else:
                return ""

        if state is None:
            state = self.state
        indices_unique = self._get_unique_indices(state)
        dones = self._get_dones(state)
        substates = self._get_substates(state)
        readable = (
            f"Active subenv {self._get_active_subenv(state)}; "
            + f"Toggle flag {self._get_toggle_flag(state)};\n"
            + "".join(
                [
                    f"{idx}: "
                    + self._get_env_unique(idx).state2readable(substate)
                    + _done2str(done)
                    + ";\n"
                    for idx, done, substate in zip(indices_unique, dones, substates)
                ]
            )
        )
        readable = readable[:-2]
        return readable

    def readable2state(self, readable: str) -> List[int]:
        """
        Converts a human-readable representation of a state into the standard format.
        """
        state = copy(self.source)
        readables = readable.split(";")
        self._set_active_subenv(int(readables[0].split(" ")[-1]), state)
        self._set_toggle_flag(int(readables[1].split(" ")[-1]), state)
        readables = [readable.strip() for readable in readables[2:]]
        for idx, (subenv, readable) in enumerate(zip(self.subenvs, readables)):
            idx_unique, readable = readable.split(": ")
            self._set_unique_index(idx, int(idx_unique), state)
            self._set_substate(
                idx, subenv.readable2state(readable.split(" | ")[0]), state
            )
            self._set_subdone(idx, " | done" in readable, state)
        return state


def make_set(
    is_flexible: bool,
    subenvs: Iterable[GFlowNetEnv] = None,
    max_elements: int = None,
    envs_unique: Iterable[GFlowNetEnv] = None,
    do_random_subenvs: bool = False,
    **kwargs,
):
    """
    Factory method to create SetFix or SetFlex classes depending on the input
    is_flexible.

    This method mimics conditional inheritance.

    Parameters
    ----------
    is_flexible : bool
        If True, return a SetFlex environment. If False, return a SetFix environment.
    """
    # If is_flexible is False, subenvs must be defined.
    if not is_flexible and subenvs is None:
        raise ValueError(f"subenvs must be defined to use the SetFix")
    # If is_flexible is True, then max_elements must be defined.
    if is_flexible and max_elements is None:
        raise ValueError(f"max_elements must be defined to use the SetFlex")

    if is_flexible:
        return SetFlex(
            max_elements=max_elements,
            envs_unique=envs_unique,
            subenvs=subenvs,
            do_random_subenvs=do_random_subenvs,
            **kwargs,
        )
    else:
        return SetFix(subenvs=subenvs, **kwargs)
