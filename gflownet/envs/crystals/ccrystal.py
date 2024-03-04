import json
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.crystals.clattice_parameters import CLatticeParameters
from gflownet.envs.crystals.composition import Composition
from gflownet.envs.crystals.spacegroup import SpaceGroup
from gflownet.utils.common import copy, tbool, tfloat, tlong
from gflownet.utils.crystals.constants import TRICLINIC


class Stage(Enum):
    """
    In addition to encoding current stage, contains methods used for padding individual
    component environment's actions (to ensure they have the same length for
    tensorization).
    """

    COMPOSITION = 0
    SPACE_GROUP = 1
    LATTICE_PARAMETERS = 2
    DONE = 3

    def to_pad(self) -> int:
        """
        Maps stage value to a padding. The following mapping is used:

        COMPOSITION = -2
        SPACE_GROUP = -3
        LATTICE_PARAMETERS = -4

        We use negative numbers starting from -2 because they are not used by any of
        the underlying environments, which should lead to every padded action being
        unique.
        """
        return -(self.value + 2)

    @classmethod
    def from_pad(cls, pad_value: int) -> "Stage":
        return Stage(-pad_value - 2)


class CCrystal(GFlowNetEnv):
    """
    A combination of Composition, SpaceGroup and CLatticeParameters into a single
    environment. Works sequentially, by first filling in the Composition, then
    SpaceGroup, and finally CLatticeParameters.
    """

    def __init__(
        self,
        composition_kwargs: Optional[Dict] = None,
        space_group_kwargs: Optional[Dict] = None,
        lattice_parameters_kwargs: Optional[Dict] = None,
        do_composition_to_sg_constraints: bool = True,
        do_sg_to_composition_constraints: bool = True,
        do_sg_to_lp_constraints: bool = True,
        do_sg_before_composition: bool = False,
        **kwargs,
    ):
        self.do_sg_to_composition_constraints = (
            do_sg_to_composition_constraints and do_sg_before_composition
        )
        self.do_composition_to_sg_constraints = (
            do_composition_to_sg_constraints and not do_sg_before_composition
        )
        self.do_sg_to_lp_constraints = do_sg_to_lp_constraints
        self.do_sg_before_composition = do_sg_before_composition

        self.composition_kwargs = dict(
            composition_kwargs or {},
            do_spacegroup_check=self.do_sg_to_composition_constraints,
        )
        self.space_group_kwargs = space_group_kwargs or {}
        self.lattice_parameters_kwargs = lattice_parameters_kwargs or {}

        composition = Composition(**self.composition_kwargs)
        space_group = SpaceGroup(**self.space_group_kwargs)
        # We initialize lattice parameters with triclinic lattice system as it is the
        # most general one, but it will have to be reinitialized using proper lattice
        # system from space group once that is determined.
        lattice_parameters = CLatticeParameters(
            lattice_system=TRICLINIC, **self.lattice_parameters_kwargs
        )
        self.subenvs = OrderedDict(
            {
                Stage.COMPOSITION: composition,
                Stage.SPACE_GROUP: space_group,
                Stage.LATTICE_PARAMETERS: lattice_parameters,
            }
        )

        # 0-th element of state encodes current stage: 0 for composition,
        # 1 for space group, 2 for lattice parameters
        initial_stage = self._get_next_stage(None)
        self.source = [initial_stage.value]
        for subenv in self.subenvs.values():
            self.source.extend(subenv.source)

        # Get action dimensionality by computing the maximum action length among all
        # sub-environments.
        self.max_action_length = max(
            [len(subenv.eos) for subenv in self.subenvs.values()]
        )

        # EOS is EOS of the last stage (lattice parameters)
        self.eos = self._pad_action(
            self.subenvs[Stage.LATTICE_PARAMETERS].eos, Stage.LATTICE_PARAMETERS
        )

        # Mask dimensionality
        self.mask_dim = sum([subenv.mask_dim for subenv in self.subenvs.values()])

        # Base class init
        # Since only the lattice parameters subenv has distribution parameters, only
        # these are pased to the base init.
        super().__init__(
            fixed_distr_params=self.subenvs[
                Stage.LATTICE_PARAMETERS
            ].fixed_distr_params,
            random_distr_params=self.subenvs[
                Stage.LATTICE_PARAMETERS
            ].random_distr_params,
            **kwargs,
        )
        self.continuous = True

    # TODO: remove or redo
    def _set_lattice_parameters(self):
        """
        Sets CLatticeParameters conditioned on the lattice system derived from the
        SpaceGroup.
        """
        if self.subenvs[Stage.SPACE_GROUP].lattice_system == "None":
            raise ValueError(
                "Cannot set lattice parameters without lattice system determined in "
                "the space group."
            )
        self.subenvs[Stage.LATTICE_PARAMETERS] = CLatticeParameters(
            lattice_system=self.subenvs[Stage.SPACE_GROUP].lattice_system,
            **self.lattice_parameters_kwargs,
        )

    def _pad_action(self, action: Tuple[int], stage: Stage) -> Tuple[int]:
        """
        Pads action such that all actions, regardless of the underlying environment,
        have the same length. Required due to the fact that action space has to be
        convertable to a tensor.
        """
        return action + (Stage.to_pad(stage),) * (self.max_action_length - len(action))

    def _pad_action_space(
        self, action_space: List[Tuple[int]], stage: Stage
    ) -> List[Tuple[int]]:
        return [self._pad_action(a, stage) for a in action_space]

    def _depad_action(self, action: Tuple[int], stage: Stage) -> Tuple[int]:
        """
        Reverses padding operation, such that the resulting action can be passed to the
        underlying environment.
        """
        return action[: len(self.subenvs[stage].eos)]

    # TODO: consider removing if unused because too simple
    def _get_actions_of_subenv(
        self, actions: TensorType["n_states", "action_dim"], stage: Stage
    ):
        """
        Returns the columns of a tensor of actions that correspond to the
        sub-environment indicated by stage.

        Args
        actions
        mask : tensor
            A tensor containing a batch of actions. It is assumed that all the rows in
            the this tensor correspond to the same stage.

        stage : Stage
            Identifier of the sub-environment of which the corresponding columns of the
            actions are to be extracted.
        """
        return actions[:, len(self.subenvs[stage].eos)]

    def get_action_space(self) -> List[Tuple[int]]:
        action_space = []
        for stage, subenv in self.subenvs.items():
            action_space.extend(self._pad_action_space(subenv.action_space, stage))

        if len(action_space) != len(set(action_space)):
            raise ValueError(
                "Detected duplicate actions between different components of Crystal "
                "environment."
            )

        return action_space

    def action2representative(self, action: Tuple) -> Tuple:
        """
        Replaces the continuous values of lattice parameters actions by the
        representative action of the environment so that it can be compared against the
        action space.
        """
        if self._get_stage() == Stage.LATTICE_PARAMETERS:
            return self.subenvs[Stage.LATTICE_PARAMETERS].action2representative(
                self._depad_action(action, Stage.LATTICE_PARAMETERS)
            )
        return action

    def get_max_traj_length(self) -> int:
        return sum([subenv.get_max_traj_length() for subenv in self.subenvs.values()])

    def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model.

        The policy output is in this case the concatenation of the policy outputs of
        the three sub-environments.
        """
        return torch.cat(
            [subenv.get_policy_output(params) for subenv in self.subenvs.values()]
        )

    def _get_policy_outputs_of_subenv(
        self, policy_outputs: TensorType["n_states", "policy_output_dim"], stage: Stage
    ):
        """
        Returns the columns of the policy outputs that correspond to the
        sub-environment indicated by stage.

        Args
        ----
        policy_outputs : tensor
            A tensor containing a batch of policy outputs. It is assumed that all the
            rows in the this tensor correspond to the same stage.

        stage : Stage
            Identifier of the sub-environment of which the corresponding columns of the
            policy outputs are to be extracted.
        """
        init_col = 0
        for stg, subenv in self.subenvs.items():
            end_col = init_col + subenv.policy_output_dim
            if stg == stage:
                return policy_outputs[:, init_col:end_col]
            init_col = end_col

    def _get_mask_of_subenv(
        self, mask: Union[List, TensorType["n_states", "mask_dim"]], stage: Stage
    ):
        """
        Returns the columns of a tensor of masks that correspond to the sub-environment
        indicated by stage.

        Args
        ----
        mask : list or tensor
            A mask of a single state as a list or a tensor containing a batch of masks.
            It is assumed that all the rows in the this tensor correspond to the same
            stage.

        stage : Stage
            Identifier of the sub-environment of which the corresponding columns of the
            masks are to be extracted.
        """
        init_col = 0
        for stg, subenv in self.subenvs.items():
            end_col = init_col + subenv.mask_dim
            if stg == stage:
                if isinstance(mask, list):
                    return mask[init_col:end_col]
                else:
                    return mask[:, init_col:end_col]
            init_col = end_col

    def reset(self, env_id: Union[int, str] = None):
        self.subenvs[Stage.COMPOSITION].reset()
        self.subenvs[Stage.SPACE_GROUP].reset()
        self.subenvs[Stage.LATTICE_PARAMETERS] = CLatticeParameters(
            lattice_system=TRICLINIC, **self.lattice_parameters_kwargs
        )

        super().reset(env_id=env_id)
        self._set_stage(self._get_next_stage(None))

        return self

    def _get_stage(self, state: Optional[List] = None) -> Stage:
        """
        Returns the stage of the current environment from self.state[0] or from the
        state passed as an argument.
        """
        if state is None:
            state = self.state
        return Stage(state[0])

    def _set_stage(self, stage: Stage, state: Optional[List] = None):
        """
        Sets the stage of the current environment (self.state) or of the state passed
        as an argument by updating state[0].
        """
        if state is None:
            state = self.state
        state[0] = stage.value

    def _get_policy_states_of_subenv(
        self, state: TensorType["n_states", "state_dim"], stage: Stage
    ):
        """
        Returns the part of the states corresponding to the subenv indicated by stage.

        Args
        ----
        states : tensor
            A tensor containing a batch of states in policy format.

        stage : Stage
            Identifier of the sub-environment of which the corresponding columns of the
            batch of states are to be extracted.
        """
        init_col = 0
        for stg, subenv in self.subenvs.items():
            end_col = init_col + subenv.policy_input_dim
            if stg == stage:
                return states[:, init_col:end_col]
            init_col = end_col

    def _get_state_of_subenv(self, state: List, stage: Optional[Stage] = None):
        """
        Returns the part of the state corresponding to the subenv indicated by stage.

        Args
        ----
        state : list
            A state of the parent Crystal environment.

        stage : Stage
            Identifier of the sub-environment of which the corresponding part of the
            state is to be extracted. If None, it is inferred from the state.
        """
        if stage is None:
            stage = self._get_stage(state)
        init_col = 1
        for stg, subenv in self.subenvs.items():
            end_col = init_col + len(subenv.source)
            if stg == stage:
                return state[init_col:end_col]
            init_col = end_col

    def _get_states_of_subenv(
        self, states: TensorType["n_states", "state_dim"], stage: Stage
    ):
        """
        Returns the part of the batch of states corresponding to the subenv indicated
        by stage.

        Args
        ----
        states : tensor
            A batch of states of the parent Crystal environment.

        stage : Stage
            Identifier of the sub-environment of which the corresponding part of the
            states is to be extracted. If None, it is inferred from the states.
        """
        init_col = 1
        for stg, subenv in self.subenvs.items():
            end_col = init_col + len(subenv.source)
            if stg == stage:
                return states[:, init_col:end_col]
            init_col = end_col

    def _is_source_state(self, state) -> bool:
        """Determines if the provided state is a source state.
        This method returns True if the provided state corresponds to the initial state
        of any of the sub-environments. Returns False otherwise.
        """
        stage = self._get_stage(state)
        return self._get_state_of_subenv(state, stage) == self.subenvs[stage].source

    def _get_previous_stage(self, stage: Stage) -> Stage:
        """Return the stage that preceeds the provided stage.
        There are two possible stage ordering depending on
        self.do_sg_before_composition. Either :
        Composition -> SpaceGroup -> LatticeParameter -> Done
        or
        SpaceGroup -> Composition -> LatticeParameter -> Done
        """
        if self.do_sg_before_composition:
            if stage is Stage.SPACE_GROUP:
                # Space group is the initial stage. No previous stage.
                return Stage.DONE
            elif stage is Stage.COMPOSITION:
                return Stage.SPACE_GROUP
            elif stage is Stage.LATTICE_PARAMETERS:
                return Stage.COMPOSITION
            elif stage is Stage.DONE:
                return Stage.LATTICE_PARAMETERS
            else:
                raise ValueError(f"Unrecognized stage {stage}.")

        else:
            if stage is Stage.COMPOSITION:
                # Space group is the initial stage. No previous stage.
                return Stage.DONE
            elif stage is Stage.SPACE_GROUP:
                return Stage.COMPOSITION
            elif stage is Stage.LATTICE_PARAMETERS:
                return Stage.SPACE_GROUP
            elif stage is Stage.DONE:
                return Stage.LATTICE_PARAMETERS
            else:
                raise ValueError(f"Unrecognized stage {stage}.")

    def _get_next_stage(self, stage: Stage = None) -> Stage:
        """Returns the stage that follows the provided stage.
        If no stage is provided, this function will return the initial stage. There are
        two possible stage ordering depending on self.do_sg_before_composition. Either :
        Composition -> SpaceGroup -> LatticeParameter -> Done
        or
        SpaceGroup -> Composition -> LatticeParameter -> Done
        """
        if self.do_sg_before_composition:
            if stage is None:
                # In the event of a environment reset, return the initial stage
                return Stage.SPACE_GROUP
            elif stage is Stage.SPACE_GROUP:
                return Stage.COMPOSITION
            elif stage is Stage.COMPOSITION:
                return Stage.LATTICE_PARAMETERS
            elif stage is Stage.LATTICE_PARAMETERS:
                return Stage.DONE
            elif stage is Stage.DONE:
                return None
            else:
                raise ValueError(f"Unrecognized stage {stage}.")

        else:
            if stage is None:
                # In the event of a environment reset, return the initial stage
                return Stage.COMPOSITION
            elif stage is Stage.COMPOSITION:
                return Stage.SPACE_GROUP
            elif stage is Stage.SPACE_GROUP:
                return Stage.LATTICE_PARAMETERS
            elif stage is Stage.LATTICE_PARAMETERS:
                return Stage.DONE
            elif stage is Stage.DONE:
                return None
            else:
                raise ValueError(f"Unrecognized stage {stage}.")

    # TODO: set mask of done state if stage is not the current one for correctness.
    def get_mask_invalid_actions_forward(
        self, state: Optional[List[int]] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the forward actions mask of the state.
        """
        state = self._get_state(state)
        stage = self._get_stage(state)
        done = self._get_done(done)

        mask = []
        for subenv_stage, subenv in self.subenvs.items():
            # Get the mask of the current stage
            subenv_mask = subenv.get_mask_invalid_actions_forward(
                self._get_state_of_subenv(state, subenv_stage), done
            )

            # If the subenv is not the current stage, make all actions invalid.
            # TODO : We could save on computation by not calling
            # _get_state_of_subenv() to generate these all-invalid masks
            if subenv_stage != stage:
                subenv_mask = [True] * len(subenv_mask)

            mask.extend(subenv_mask)

        return mask

    # TODO: this piece of code looks awful
    def get_mask_invalid_actions_backward(
        self, state: Optional[List[int]] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the backward actions mask of the state.

        The mask of the parent crystal is, in general, simply the concatenation of the
        masks of the three sub-environments. Only the mask of the state of the current
        sub-environment is computed; for the other sub-environments, the mask of the
        source is used. Note that this assumes that the methods that will use the mask
        will extract the part corresponding to the relevant stage and ignore the rest.

        Nonetheless, in order to enable backward transitions between stages, the EOS
        action of the preceding stage has to be the only valid action when the state of
        a sub-environment is the source. Additionally, sample_batch_actions will have
        to also detect the source states and change the stage.

        Note that the sub-environments are iterated in reversed order so as to save
        unnecessary computations and simplify the code.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        stage = self._get_stage(state)

        mask = []
        do_eos_only = False

        # Iterate stages in reverse order
        subenv_masks = {}
        stg = self._get_previous_stage(Stage.DONE)
        while stg != Stage.DONE:
            subenv = self.subenvs[stg]

            state_subenv = self._get_state_of_subenv(state, stg)
            # Set mask of done state because state of next subenv is source
            if do_eos_only:
                mask_subenv = subenv.get_mask_invalid_actions_backward(
                    state_subenv, done=True
                )
                do_eos_only = False
            # General case
            else:
                # stg is the current stage
                if stg == stage:
                    # state of subenv is the source state
                    prev_stg = self._get_previous_stage(stg)
                    if prev_stg != Stage.DONE and state_subenv == subenv.source:
                        do_eos_only = True
                        mask_subenv = subenv.get_mask_invalid_actions_backward(
                            subenv.source
                        )
                    # General case
                    else:
                        mask_subenv = subenv.get_mask_invalid_actions_backward(
                            state_subenv, done
                        )
                # stg is not current stage, so set mask of source
                else:
                    mask_subenv = subenv.get_mask_invalid_actions_backward(
                        subenv.source
                    )
            subenv_masks[stg] = mask_subenv
            stg = self._get_previous_stage(stg)

        # Combine the individual masks to produce the global mask
        for stg, subenv in self.subenvs.items():
            mask.extend(subenv_masks[stg])
        return mask

    def _update_state(self, stage: Stage):
        """
        Updates the global state based on the states of the sub-environments and the
        stage passed as an argument.
        """
        state = [stage.value]
        for subenv in self.subenvs.values():
            state.extend(subenv.state)
        return state

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes forward step given an action.

        The action is performed by the corresponding sub-environment and then the
        global state is updated accordingly. If the action is the EOS of the
        sub-environment, the stage is advanced and constraints are set on the
        subsequent sub-environment.

        Args
        ----
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
        stage = self._get_stage(self.state)
        # Skip mask check if stage is lattice parameters (continuous actions)
        if stage == Stage.LATTICE_PARAMETERS:
            skip_mask_check = True
        # Replace action by its representative to check against the mask.
        action_to_check = self.action2representative(action)
        do_step, self.state, action_to_check = self._pre_step(
            action_to_check,
            skip_mask_check=(skip_mask_check or self.skip_mask_check),
        )
        if not do_step:
            return self.state, action, False

        # Call step of current subenvironment
        action_subenv = self._depad_action(action, stage)
        _, action_subenv, valid = self.subenvs[stage].step(action_subenv)

        # If action is invalid, exit immediately. Otherwise increment actions and go on
        if not valid:
            return self.state, action, False
        self.n_actions += 1

        # If action is EOS of subenv, advance stage and set constraints or exit
        if action_subenv == self.subenvs[stage].eos:
            stage = self._get_next_stage(stage)

            if stage is Stage.SPACE_GROUP:
                if (
                    not self.do_sg_before_composition
                    and self.do_composition_to_sg_constraints
                ):
                    self.subenvs[Stage.SPACE_GROUP].set_n_atoms_compatibility_dict(
                        self.subenvs[Stage.COMPOSITION].state
                    )

            elif stage is Stage.COMPOSITION:
                if (
                    self.do_sg_before_composition
                    and self.do_sg_to_composition_constraints
                ):
                    space_group = self.subenvs[Stage.SPACE_GROUP].space_group
                    self.subenvs[Stage.COMPOSITION].space_group = space_group

            elif stage is Stage.LATTICE_PARAMETERS:
                if self.do_sg_to_lp_constraints:
                    lattice_system = self.subenvs[Stage.SPACE_GROUP].lattice_system
                    self.subenvs[Stage.LATTICE_PARAMETERS].set_lattice_system(
                        lattice_system
                    )

            elif stage is Stage.DONE:
                self.n_actions += 1
                self.done = True
                return self.state, self.eos, True

            else:
                raise ValueError(f"Unrecognized stage {stage}.")

        self.state = self._update_state(stage)
        return self.state, action, valid

    def step_backwards(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes backward step given an action.

        The action is performed by the corresponding sub-environment and then the
        global state is updated accordingly. If the updated state of the
        sub-environment becomes its source, the stage is decreased.

        Args
        ----
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
        stage = self._get_stage(self.state)
        # Skip mask check if stage is lattice parameters (continuous actions)
        if stage == Stage.LATTICE_PARAMETERS:
            skip_mask_check = True
        # Replace action by its representative to check against the mask.
        action_to_check = self.action2representative(action)
        do_step, self.state, action_to_check = self._pre_step(
            action_to_check,
            backward=True,
            skip_mask_check=(skip_mask_check or self.skip_mask_check),
        )
        if not do_step:
            return self.state, action, False

        # If state of subenv is source of subenv, decrease stage
        if self._get_state_of_subenv(self.state, stage) == self.subenvs[stage].source:
            stage = self._get_previous_stage(stage)
            # If stage is DONE, we've returned to the environment's initial state,
            # set global source and return
            if stage is Stage.DONE:
                self.state = self.source
                return self.state, action, True

        # Call step of current subenvironment
        action_subenv = self._depad_action(action, stage)
        state_next, _, valid = self.subenvs[stage].step_backwards(action_subenv)

        # If action is invalid, exit immediately. Otherwise continue,
        if not valid:
            return self.state, action, False
        self.n_actions += 1

        # If action from done, set done False
        if self.done:
            assert action == self.eos
            self.done = False

        self.state = self._update_state(stage)
        return self.state, action, valid

    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "policy_output_dim"]] = None,
        states_from: List = None,
        is_backward: Optional[bool] = False,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs.

        This method calls the sample_actions_batch() method of the sub-environment
        corresponding to each state in the batch. For composition and space_group it
        will be the method from the base discrete environment; for the lattice
        parameters, it will be the method from the cube environment.

        Note that in order to call sample_actions_batch() of the sub-environments, we
        need to first extract the part of the policy outputs, the masks and the states
        that correspond to the sub-environment.
        """
        states_dict = {stage: [] for stage in Stage}
        """
        A dictionary with keys equal to Stage and the values are the list of states in
        the stage of the key. The states are only the part corresponding to the
        sub-environment.
        """
        stages = []
        for s in states_from:
            stage = self._get_stage(s)
            state_subenv = self._get_state_of_subenv(s, stage)
            # If the actions are backwards and state is source of subenv, decrease
            # stage so that EOS of preceding stage is sampled.
            if (
                is_backward
                and self._get_previous_stage(stage) != Stage.DONE
                and state_subenv == self.subenvs[stage].source
            ):
                stage = self._get_previous_stage(stage)
            states_dict[stage].append(state_subenv)
            stages.append(stage)
        stages_tensor = tlong([stage.value for stage in stages], device=self.device)
        is_subenv_dict = {stage: stages_tensor == stage.value for stage in Stage}

        # Sample actions from each sub-environment
        actions_logprobs_dict = {
            stage: subenv.sample_actions_batch(
                self._get_policy_outputs_of_subenv(
                    policy_outputs[is_subenv_dict[stage]], stage
                ),
                self._get_mask_of_subenv(mask[is_subenv_dict[stage]], stage),
                states_dict[stage],
                is_backward,
                sampling_method,
                temperature_logits,
                max_sampling_attempts,
            )
            for stage, subenv in self.subenvs.items()
            if torch.any(is_subenv_dict[stage])
        }

        # Stitch all actions in the right order, with the right padding
        actions = []
        for stage in stages:
            actions.append(
                self._pad_action(actions_logprobs_dict[stage][0].pop(0), stage)
            )
        return actions, None

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", "actions_dim"],
        mask: TensorType["n_states", "mask_dim"],
        states_from: List,
        is_backward: bool,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.

        Args
        ----
        policy_outputs : tensor
            The output of the GFlowNet policy model.

        mask : tensor
            The mask containing information about invalid actions and special cases.

        actions : tensor
            The actions (global) from each state in the batch for which to compute the
            log probability.

        states_from : tensor
            The states originating the actions, in GFlowNet format.

        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default).
        """
        n_states = policy_outputs.shape[0]
        states_dict = {stage: [] for stage in Stage}
        """
        A dictionary with keys equal to Stage and the values are the list of states in
        the stage of the key. The states are only the part corresponding to the
        sub-environment.
        """
        stages = []
        for s in states_from:
            stage = self._get_stage(s)
            state_subenv = self._get_state_of_subenv(s, stage)
            # If the actions are backwards and state is source of subenv, decrease
            # stage so that EOS of preceding stage is sampled.
            if (
                is_backward
                and self._get_previous_stage(stage) != Stage.DONE
                and state_subenv == self.subenvs[stage].source
            ):
                stage = self._get_previous_stage(stage)
            states_dict[stage].append(state_subenv)
            stages.append(stage)
        stages_tensor = tlong([stage.value for stage in stages], device=self.device)
        is_subenv_dict = {stage: stages_tensor == stage.value for stage in Stage}

        # Compute logprobs from each sub-environment
        logprobs = torch.empty(n_states, dtype=self.float, device=self.device)
        for stage, subenv in self.subenvs.items():
            if not torch.any(is_subenv_dict[stage]):
                continue
            logprobs[is_subenv_dict[stage]] = subenv.get_logprobs(
                self._get_policy_outputs_of_subenv(
                    policy_outputs[is_subenv_dict[stage]], stage
                ),
                actions[is_subenv_dict[stage], : len(subenv.eos)],
                self._get_mask_of_subenv(mask[is_subenv_dict[stage]], stage),
                states_dict[stage],
                is_backward,
            )
        return logprobs

    def states2policy(
        self, states: Union[List[List], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_policy_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: simply
        a concatenation of all crystal components.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tfloat(states, device=self.device, float_type=self.float)
        return torch.cat(
            [
                subenv.states2policy(self._get_states_of_subenv(states, stage))
                for stage, subenv in self.subenvs.items()
            ],
            dim=1,
        )

    def states2proxy(
        self, states: Union[List[List], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: simply a
        concatenation of all crystal components.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tfloat(states, device=self.device, float_type=self.float)
        return torch.cat(
            [
                subenv.states2proxy(self._get_states_of_subenv(states, stage))
                for stage, subenv in self.subenvs.items()
            ],
            dim=1,
        )

    def set_state(self, state: List, done: Optional[bool] = False):
        super().set_state(state, done)

        stage = self._get_stage(state)
        stage_idx = self._get_stage(state).value

        # Determine which subenvs are done based on stage and done
        done_subenvs = {
            Stage.COMPOSITION: False,
            Stage.SPACE_GROUP: False,
            Stage.LATTICE_PARAMETERS: False,
        }
        if stage is Stage.COMPOSITION and self.do_sg_before_composition:
            done_subenvs[Stage.SPACE_GROUP] = True
        elif stage is Stage.SPACE_GROUP and not self.do_sg_before_composition:
            done_subenvs[Stage.COMPOSITION] = True
        elif stage is Stage.LATTICE_PARAMETERS:
            done_subenvs[Stage.COMPOSITION] = True
            done_subenvs[Stage.SPACE_GROUP] = True
        elif stage is Stage.DONE:
            for subenv in done_subenvs:
                done_subenvs[subenv] = True
        done_subenvs[Stage.LATTICE_PARAMETERS] = done

        # Set state and done of each sub-environment
        for (stage, subenv), subenv_done in zip(self.subenvs.items(), done_subenvs):
            stage_done = done_subenvs[stage]
            subenv.set_state(self._get_state_of_subenv(state, stage), stage_done)

        if self.subenvs[Stage.SPACE_GROUP].done:
            """
            We synchronize LatticeParameter's lattice system with the one of SpaceGroup
            (if it was set) or reset it to the default triclinic otherwise. Why this is
            needed: for backward sampling, where we start from an arbitrary terminal
            state and need to synchronize the LatticeParameter's lattice system to what
            that state indicates,
            """
            lattice_system = self.subenvs[Stage.SPACE_GROUP].lattice_system
            if lattice_system != "None" and self.do_sg_to_lp_constraints:
                self.subenvs[Stage.LATTICE_PARAMETERS].set_lattice_system(
                    lattice_system
                )
            else:
                self.subenvs[Stage.LATTICE_PARAMETERS].set_lattice_system(TRICLINIC)

            # Set the stoichiometry constraints in the composition sub-environment
            if self.do_sg_before_composition and self.do_sg_to_composition_constraints:
                space_group = self.subenvs[Stage.SPACE_GROUP].space_group
                self.subenvs[Stage.COMPOSITION].space_group = space_group

        # Set stoichiometry constraints in space group sub-environment
        if (
            self.do_composition_to_sg_constraints
            and self.subenvs[Stage.COMPOSITION].done
        ):
            self.subenvs[Stage.SPACE_GROUP].set_n_atoms_compatibility_dict(
                self.subenvs[Stage.COMPOSITION].state
            )

    def state2readable(self, state: Optional[List[int]] = None) -> str:
        if state is None:
            state = self.state

        readables = [
            subenv.state2readable(self._get_state_of_subenv(state, stage))
            for stage, subenv in self.subenvs.items()
        ]
        return (
            f"{self._get_stage(state)}; "
            f"Composition = {readables[0]}; "
            f"SpaceGroup = {readables[1]}; "
            f"LatticeParameters = {readables[2]}"
        )

    def process_data_set(self, data: List[List]) -> List[List]:
        is_valid_list = []
        for x in data:
            is_valid_list.append(
                all(
                    [
                        subenv.is_valid(self._get_state_of_subenv(x, stage))
                        for stage, subenv in self.subenvs.items()
                    ]
                )
            )
        return [x for x, is_valid in zip(data, is_valid_list) if is_valid]

    # TODO: redo


#     def readable2state(self, readable: str) -> List[int]:
#         splits = readable.split("; ")
#         readables = [x.split(" = ")[1] for x in splits]
#
#         return (
#             [int(readables[0])]
#             + self.composition.readable2state(
#                 json.loads(readables[1].replace("'", '"'))
#             )
#             + self.space_group.readable2state(readables[2])
#             + self.lattice_parameters.readable2state(readables[3])
#         )
