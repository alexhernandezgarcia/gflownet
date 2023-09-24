import json
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

    def next(self) -> "Stage":
        """
        Returns the next Stage in the enumeration or None if at the last stage.
        """
        if self.value + 1 == len(Stage):
            return None
        return Stage(self.value + 1)

    def prev(self) -> "Stage":
        """
        Returns the previous Stage in the enumeration or DONE if from the first stage.
        """
        if self.value - 1 < 0:
            return Stage.DONE
        return Stage(self.value - 1)

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
        do_stoichiometry_sg_check: bool = False,
        **kwargs,
    ):
        self.composition_kwargs = composition_kwargs or {}
        self.space_group_kwargs = space_group_kwargs or {}
        self.lattice_parameters_kwargs = lattice_parameters_kwargs or {}
        self.do_stoichiometry_sg_check = do_stoichiometry_sg_check

        self.composition = Composition(**self.composition_kwargs)
        self.space_group = SpaceGroup(**self.space_group_kwargs)
        # We initialize lattice parameters with triclinic lattice system as it is the
        # most general one, but it will have to be reinitialized using proper lattice
        # system from space group once that is determined.
        self.lattice_parameters = CLatticeParameters(
            lattice_system=TRICLINIC, **self.lattice_parameters_kwargs
        )
        self.subenvs = {
            Stage.COMPOSITION: self.composition,
            Stage.SPACE_GROUP: self.space_group,
            Stage.LATTICE_PARAMETERS: self.lattice_parameters,
        }

        # 0-th element of state encodes current stage: 0 for composition,
        # 1 for space group, 2 for lattice parameters
        self.source = (
            [Stage.COMPOSITION.value]
            + self.composition.source
            + self.space_group.source
            + self.lattice_parameters.source
        )

        # start and end indices of individual substates
        self.composition_state_start = 1
        self.composition_state_end = self.composition_state_start + len(
            self.composition.source
        )
        self.space_group_state_start = self.composition_state_end
        self.space_group_state_end = self.space_group_state_start + len(
            self.space_group.source
        )
        self.lattice_parameters_state_start = self.space_group_state_end
        self.lattice_parameters_state_end = self.lattice_parameters_state_start + len(
            self.lattice_parameters.source
        )

        # start and end indices of individual submasks
        self.composition_mask_start = 0
        self.composition_mask_end = self.composition_mask_start + len(
            self.composition.action_space
        )
        self.space_group_mask_start = self.composition_mask_end
        self.space_group_mask_end = self.space_group_mask_start + len(
            self.space_group.action_space
        )
        self.lattice_parameters_mask_start = self.space_group_mask_end
        self.lattice_parameters_mask_end = self.lattice_parameters_mask_start + len(
            self.lattice_parameters.action_space
        )

        self.composition_action_length = max(
            len(a) for a in self.composition.action_space
        )
        self.space_group_action_length = max(
            len(a) for a in self.space_group.action_space
        )
        self.lattice_parameters_action_length = max(
            len(a) for a in self.lattice_parameters.action_space
        )
        self.max_action_length = max(
            self.composition_action_length,
            self.space_group_action_length,
            self.lattice_parameters_action_length,
        )

        # EOS is EOS of LatticeParameters because it is the last stage
        self.eos = self._pad_action(
            self.lattice_parameters.eos, Stage.LATTICE_PARAMETERS
        )

        # Mask dimensionality
        self.mask_dim = sum([subenv.mask_dim for subenv in self.subenvs.values()])

        # Conversions
        self.state2proxy = self.state2oracle
        self.statebatch2proxy = self.statebatch2oracle
        self.statetorch2proxy = self.statetorch2oracle

        # Base class init
        # Since only the lattice parameters subenv has distribution parameters, only
        # these are pased to the base init.
        super().__init__(
            fixed_distr_params=self.lattice_parameters.fixed_distr_params,
            random_distr_params=self.lattice_parameters.random_distr_params,
            **kwargs,
        )
        self.continuous = True

    def _set_lattice_parameters(self):
        """
        Sets CLatticeParameters conditioned on the lattice system derived from the
        SpaceGroup.
        """
        if self.space_group.lattice_system == "None":
            raise ValueError(
                "Cannot set lattice parameters without lattice system determined in "
                "the space group."
            )

        self.lattice_parameters = CLatticeParameters(
            lattice_system=self.space_group.lattice_system,
            **self.lattice_parameters_kwargs,
        )
        self.subenvs[Stage.LATTICE_PARAMETERS] = self.lattice_parameters

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
        if stage == Stage.COMPOSITION:
            dim = self.composition_action_length
        elif stage == Stage.SPACE_GROUP:
            dim = self.space_group_action_length
        elif stage == Stage.LATTICE_PARAMETERS:
            dim = self.lattice_parameters_action_length
        else:
            raise ValueError(f"Unrecognized stage {stage}.")

        return action[:dim]

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
        composition_action_space = self._pad_action_space(
            self.composition.action_space, Stage.COMPOSITION
        )
        space_group_action_space = self._pad_action_space(
            self.space_group.action_space, Stage.SPACE_GROUP
        )
        lattice_parameters_action_space = self._pad_action_space(
            self.lattice_parameters.action_space, Stage.LATTICE_PARAMETERS
        )

        action_space = (
            composition_action_space
            + space_group_action_space
            + lattice_parameters_action_space
        )

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
            return self.lattice_parameters.action2representative(
                self._depad_action(action, Stage.LATTICE_PARAMETERS)
            )
        return action

    def get_max_traj_length(self) -> int:
        return (
            self.composition.get_max_traj_length()
            + self.space_group.get_max_traj_length()
            + self.lattice_parameters.get_max_traj_length()
        )

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
        self, mask: TensorType["n_states", "mask_dim"], stage: Stage
    ):
        """
        Returns the columns of a tensor of masks that correspond to the sub-environment
        indicated by stage.

        Args
        ----
        mask : tensor
            A tensor containing a batch of masks. It is assumed that all the rows in
            the this tensor correspond to the same stage.

        stage : Stage
            Identifier of the sub-environment of which the corresponding columns of the
            masks are to be extracted.
        """
        init_col = 0
        for stg, subenv in self.subenvs.items():
            end_col = init_col + subenv.mask_dim
            if stg == stage:
                return mask[:, init_col:end_col]
            init_col = end_col

    def reset(self, env_id: Union[int, str] = None):
        self.composition.reset()
        self.space_group.reset()
        self.lattice_parameters = CLatticeParameters(
            lattice_system=TRICLINIC, **self.lattice_parameters_kwargs
        )

        super().reset(env_id=env_id)
        self._set_stage(Stage.COMPOSITION)

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

    def _get_composition_state(self, state: Optional[List[int]] = None) -> List[int]:
        state = self._get_state(state)

        return state[self.composition_state_start : self.composition_state_end]

    def _get_composition_tensor_states(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_oracle_dim"]:
        return states[:, self.composition_state_start : self.composition_state_end]

    def _get_space_group_state(self, state: Optional[List[int]] = None) -> List[int]:
        state = self._get_state(state)

        return state[self.space_group_state_start : self.space_group_state_end]

    def _get_space_group_tensor_states(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_oracle_dim"]:
        return states[:, self.space_group_state_start : self.space_group_state_end]

    def _get_lattice_parameters_state(
        self, state: Optional[List[int]] = None
    ) -> List[int]:
        state = self._get_state(state)

        return state[
            self.lattice_parameters_state_start : self.lattice_parameters_state_end
        ]

    def _get_lattice_parameters_tensor_states(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_oracle_dim"]:
        return states[
            :, self.lattice_parameters_state_start : self.lattice_parameters_state_end
        ]

    def get_mask_invalid_actions_forward(
        self, state: Optional[List[int]] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the forward actions mask of the state.

        The mask of the parent crystal is simply the concatenation of the masks of the
        three sub-environments. This assumes that the methods that will use the mask
        will extract the part corresponding to the relevant stage and ignore the rest.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        mask = []
        for stage, subenv in self.subenvs.items():
            mask.extend(
                subenv.get_mask_invalid_actions_forward(
                    self._get_state_of_subenv(state, stage), done
                )
            )
        return mask

    def get_mask_invalid_actions_backward(
        self, state: Optional[List[int]] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the backward actions mask of the state.

        The mask of the parent crystal is simply the concatenation of the masks of the
        three sub-environments. This assumes that the methods that will use the mask
        will extract the part corresponding to the relevant stage and ignore the rest.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        mask = []
        for stage, subenv in self.subenvs.items():
            mask.extend(
                subenv.get_mask_invalid_actions_backward(
                    self._get_state_of_subenv(state, stage), done
                )
            )
        return mask

    def _update_state(self, stage: Stage):
        """
        Updates the global state based on the states of the sub-environments and the
        stage passed as an argument.
        """
        return (
            [stage.value]
            + self.composition.state
            + self.space_group.state
            + self.lattice_parameters.state
        )

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
            stage = Stage.next(stage)
            if stage == Stage.SPACE_GROUP:
                if self.do_stoichiometry_sg_check:
                    self.space_group.set_n_atoms_compatibility_dict(
                        self.composition.state
                    )
            elif stage == Stage.LATTICE_PARAMETERS:
                self._set_lattice_parameters()
            elif stage == Stage.DONE:
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

        # Call step of current subenvironment
        action_subenv = self._depad_action(action, stage)
        state_next, _, valid = self.subenvs[stage].step_backwards(action_subenv)

        # If action is invalid, exit immediately. Otherwise continue,
        if not valid:
            return self.state, action, False
        self.n_actions += 1

        # If next state is source of subenv, decrease stage.
        if state_next == self.subenvs[stage].source:
            stage = Stage.prev(stage)
            # If stage is DONE, return the global source
            if stage is Stage.DONE:
                return self.source, action, True

        self.state = self._update_state(stage)
        return self.state, action, valid

    def _build_state(self, substate: List, stage: Stage) -> List:
        """
        Converts the state coming from one of the subenvironments into a combined state
        format used by the Crystal environment.
        """
        if stage == Stage.COMPOSITION:
            output = (
                [0]
                + substate
                + self.space_group.source
                + self.lattice_parameters.source
            )
        elif stage == Stage.SPACE_GROUP:
            output = (
                [1] + self.composition.state + substate + self.lattice_parameters.source
            )
        elif stage == Stage.LATTICE_PARAMETERS:
            output = [2] + self.composition.state + self.space_group.state + substate
        else:
            raise ValueError(f"Unrecognized stage {stage}.")

        return output

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
            states_dict[stage].append(self._get_state_of_subenv(s, stage))
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

    def sample_actions_batch(
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
        states_dict = {stage: [] for stage in Stage}
        """
        A dictionary with keys equal to Stage and the values are the list of states in
        the stage of the key. The states are only the part corresponding to the
        sub-environment.
        """
        stages = []
        for s in states_from:
            stage = self._get_stage(s)
            states_dict[stage].append(self._get_state_of_subenv(s, stage))
            stages.append(stage)
        stages_tensor = tlong([stage.value for stage in stages], device=self.device)
        is_subenv_dict = {stage: stages_tensor == stage.value for stage in Stage}

        # Compute logprobs from each sub-environment
        logprobs = torch.empty(
            policy_input_dim.shape[0], dtype=self.float, device=self.device
        )
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

    # TODO: Consider removing altogether
    def get_parents(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        state = self._get_state(state)
        done = self._get_done(done)
        stage = self._get_stage(state)

        if done:
            return [state], [self.eos]

        if stage == Stage.COMPOSITION or (
            stage == Stage.SPACE_GROUP
            and self._get_space_group_state(state) == self.space_group.source
        ):
            composition_done = stage == Stage.SPACE_GROUP
            parents, actions = self.composition.get_parents(
                state=self._get_composition_state(state), done=composition_done
            )
            parents = [self._build_state(p, Stage.COMPOSITION) for p in parents]
            actions = [self._pad_action(a, Stage.COMPOSITION) for a in actions]
        elif stage == Stage.SPACE_GROUP or (
            stage == Stage.LATTICE_PARAMETERS
            and self._get_lattice_parameters_state(state)
            == self.lattice_parameters.source
        ):
            space_group_done = stage == Stage.LATTICE_PARAMETERS
            parents, actions = self.space_group.get_parents(
                state=self._get_space_group_state(state), done=space_group_done
            )
            parents = [self._build_state(p, Stage.SPACE_GROUP) for p in parents]
            actions = [self._pad_action(a, Stage.SPACE_GROUP) for a in actions]
        elif stage == Stage.LATTICE_PARAMETERS:
            """
            get_parents() is not well defined for continuous environment. Here we
            simply return the same state and the representative action.
            """
            parents = [state]
            actions = [self.action2representative(action)]
        else:
            raise ValueError(f"Unrecognized stage {stage}.")

        return parents, actions

    def state2oracle(self, state: Optional[List[int]] = None) -> Tensor:
        """
        Prepares a list of states in "GFlowNet format" for the oracle. Simply
        a concatenation of all crystal components.
        """
        if state is None:
            state = self.state.copy()

        composition_oracle_state = self.composition.state2oracle(
            state=self._get_composition_state(state)
        ).to(self.device)
        space_group_oracle_state = (
            self.space_group.state2oracle(state=self._get_space_group_state(state))
            .unsqueeze(-1)  # StateGroup oracle state is a single number
            .to(self.device)
        )
        lattice_parameters_oracle_state = self.lattice_parameters.state2oracle(
            state=self._get_lattice_parameters_state(state)
        ).to(self.device)

        return torch.cat(
            [
                composition_oracle_state,
                space_group_oracle_state,
                lattice_parameters_oracle_state,
            ]
        )

    def statebatch2oracle(
        self, states: List[List]
    ) -> TensorType["batch", "state_oracle_dim"]:
        return self.statetorch2oracle(
            torch.tensor(states, device=self.device, dtype=torch.long)
        )

    def statetorch2oracle(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_oracle_dim"]:
        composition_oracle_states = self.composition.statetorch2oracle(
            self._get_composition_tensor_states(states)
        ).to(self.device)
        space_group_oracle_states = self.space_group.statetorch2oracle(
            self._get_space_group_tensor_states(states)
        ).to(self.device)
        lattice_parameters_oracle_states = self.lattice_parameters.statetorch2oracle(
            self._get_lattice_parameters_tensor_states(states)
        ).to(self.device)
        return torch.cat(
            [
                composition_oracle_states,
                space_group_oracle_states,
                lattice_parameters_oracle_states,
            ],
            dim=1,
        )

    def set_state(self, state: List, done: Optional[bool] = False):
        super().set_state(state, done)

        stage_idx = self._get_stage(state).value

        # Determine which subenvs are done based on stage and done
        done_subenvs = [True] * stage_idx + [False] * (len(self.subenvs) - stage_idx)
        done_subenvs[-1] = done
        # Set state and done of each sub-environment
        for (stage, subenv), subenv_done in zip(self.subenvs.items(), done_subenvs):
            subenv.set_state(self._get_state_of_subenv(state, stage), subenv_done)

        """
        We synchronize LatticeParameter's lattice system with the one of SpaceGroup
        (if it was set) or reset it to the default triclinic otherwise. Why this is 
        needed: for backward sampling, where we start from an arbitrary terminal state,
        and need to synchronize the LatticeParameter's lattice system to what that
        state indicates,
        """
        lattice_system = self.space_group.lattice_system
        if lattice_system != "None":
            self.lattice_parameters.lattice_system = lattice_system
        else:
            self.lattice_parameters.lattice_system = TRICLINIC

    def state2readable(self, state: Optional[List[int]] = None) -> str:
        if state is None:
            state = self.state

        composition_readable = self.composition.state2readable(
            state=self._get_composition_state(state)
        )
        space_group_readable = self.space_group.state2readable(
            state=self._get_space_group_state(state)
        )
        lattice_parameters_readable = self.lattice_parameters.state2readable(
            state=self._get_lattice_parameters_state(state)
        )

        return (
            f"Stage = {state[0]}; "
            f"Composition = {composition_readable}; "
            f"SpaceGroup = {space_group_readable}; "
            f"LatticeParameters = {lattice_parameters_readable}"
        )

    def readable2state(self, readable: str) -> List[int]:
        splits = readable.split("; ")
        readables = [x.split(" = ")[1] for x in splits]

        return (
            [int(readables[0])]
            + self.composition.readable2state(
                json.loads(readables[1].replace("'", '"'))
            )
            + self.space_group.readable2state(readables[2])
            + self.lattice_parameters.readable2state(readables[3])
        )
