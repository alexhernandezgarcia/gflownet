import json
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.crystals.composition import Composition
from gflownet.envs.crystals.lattice_parameters import LatticeParameters
from gflownet.envs.crystals.spacegroup import SpaceGroup
from gflownet.utils.common import tlong
from gflownet.utils.crystals.constants import TRICLINIC


class Stage(Enum):
    """
    In addition to encoding current stage, contains methods used for padding individual
    component environment's actions (to ensure they have the same length for tensorization).
    """

    COMPOSITION = 0
    SPACE_GROUP = 1
    LATTICE_PARAMETERS = 2

    def to_pad(self) -> int:
        """
        Maps stage value to a padding. The following mapping is used:

        COMPOSITION = -2
        SPACE_GROUP = -3
        LATTICE_PARAMETERS = -4

        We use negative numbers starting from -2 because they are not used by any of the
        underlying environments, which should lead to every padded action being unique.
        """
        return -(self.value + 2)

    @classmethod
    def from_pad(cls, pad_value: int) -> "Stage":
        return Stage(-pad_value - 2)


class Crystal(GFlowNetEnv):
    """
    A combination of Composition, SpaceGroup and LatticeParameters into a single environment.
    Works sequentially, by first filling in the Composition, then SpaceGroup, and finally
    LatticeParameters.
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
        # We initialize lattice parameters here with triclinic lattice system to access
        # all the methods of that environment, but it will have to be reinitialized using
        # proper lattice system from space group once that is determined.
        # Triclinic was used because it doesn't force any initial starting angles.
        self.lattice_parameters = LatticeParameters(
            lattice_system=TRICLINIC, **self.lattice_parameters_kwargs
        )

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

        super().__init__(**kwargs)

    def _set_lattice_parameters(self):
        """
        Sets LatticeParameters conditioned on the lattice system derived from the SpaceGroup.
        """
        if self.space_group.lattice_system == "None":
            raise ValueError(
                "Cannot set lattice parameters without lattice system determined in the space group."
            )

        self.lattice_parameters = LatticeParameters(
            lattice_system=self.space_group.lattice_system,
            **self.lattice_parameters_kwargs,
        )

    def _pad_action(self, action: Tuple[int], stage: Stage) -> Tuple[int]:
        """
        Pads action such that all actions, regardless of the underlying environment, have
        the same length. Required due to the fact that action space has to be convertable to
        a tensor.
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
                "Detected duplicate actions between different components of Crystal environment."
            )

        return action_space

    def get_max_traj_length(self) -> int:
        return (
            self.composition.get_max_traj_length()
            + self.space_group.get_max_traj_length()
            + self.lattice_parameters.get_max_traj_length()
        )

    def reset(self, env_id: Union[int, str] = None):
        self.composition.reset()
        self.space_group.reset()
        self.lattice_parameters = LatticeParameters(
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

    def _get_composition_state(self, state: Optional[List[int]] = None) -> List[int]:
        if state is None:
            state = self.state.copy()

        return state[self.composition_state_start : self.composition_state_end]

    def _get_composition_tensor_states(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_proxy_dim"]:
        return states[:, self.composition_state_start : self.composition_state_end]

    def _get_space_group_state(self, state: Optional[List[int]] = None) -> List[int]:
        if state is None:
            state = self.state.copy()

        return state[self.space_group_state_start : self.space_group_state_end]

    def _get_space_group_tensor_states(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_proxy_dim"]:
        return states[:, self.space_group_state_start : self.space_group_state_end]

    def _get_lattice_parameters_state(
        self, state: Optional[List[int]] = None
    ) -> List[int]:
        if state is None:
            state = self.state.copy()

        return state[
            self.lattice_parameters_state_start : self.lattice_parameters_state_end
        ]

    def _get_lattice_parameters_tensor_states(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_proxy_dim"]:
        return states[
            :, self.lattice_parameters_state_start : self.lattice_parameters_state_end
        ]

    def get_mask_invalid_actions_forward(
        self, state: Optional[List[int]] = None, done: Optional[bool] = None
    ) -> List[bool]:
        state = self._get_state(state)
        done = self._get_done(done)
        stage = self._get_stage(state)

        if done:
            return [True] * self.action_space_dim

        mask = [True] * self.action_space_dim

        if stage == Stage.COMPOSITION:
            composition_mask = self.composition.get_mask_invalid_actions_forward(
                state=self._get_composition_state(state), done=False
            )
            mask[self.composition_mask_start : self.composition_mask_end] = (
                composition_mask
            )
        elif stage == Stage.SPACE_GROUP:
            space_group_state = self._get_space_group_state(state)
            space_group_mask = self.space_group.get_mask_invalid_actions_forward(
                state=space_group_state, done=False
            )
            mask[self.space_group_mask_start : self.space_group_mask_end] = (
                space_group_mask
            )
        elif stage == Stage.LATTICE_PARAMETERS:
            """
            TODO: to be stateless (meaning, operating as a function, not a method with
            current object context) this needs to set lattice system based on the passed
            state only. Right now it uses the current LatticeParameter environment, in
            particular the lattice system that it was set to, and that changes the invalid
            actions mask.

            If for some reason a state will be passed to this method that describes an
            object with different lattice system than what self.lattice_system contains,
            the result will be invalid.
            """
            lattice_parameters_state = self._get_lattice_parameters_state(state)
            lattice_parameters_mask = (
                self.lattice_parameters.get_mask_invalid_actions_forward(
                    state=lattice_parameters_state, done=False
                )
            )
            mask[
                self.lattice_parameters_mask_start : self.lattice_parameters_mask_end
            ] = lattice_parameters_mask
        else:
            raise ValueError(f"Unrecognized stage {stage}.")

        return mask

    def _update_state(self):
        """
        Updates current state based on the states of underlying environments.
        """
        self.state = (
            [self._get_stage(self.state).value]
            + self.composition.state
            + self.space_group.state
            + self.lattice_parameters.state
        )

    def step(self, action: Tuple[int]) -> Tuple[List[int], Tuple[int], bool]:
        # If action not found in action space raise an error
        if action not in self.action_space:
            raise ValueError(
                f"Tried to execute action {action} not present in action space."
            )
        else:
            action_idx = self.action_space.index(action)
        # If action is in invalid mask, exit immediately
        if self.get_mask_invalid_actions_forward()[action_idx]:
            return self.state, action, False
        self.n_actions += 1

        stage = self._get_stage(self.state)
        if stage == Stage.COMPOSITION:
            composition_action = self._depad_action(action, Stage.COMPOSITION)
            _, executed_action, valid = self.composition.step(composition_action)
            if valid and executed_action == self.composition.eos:
                self._set_stage(Stage.SPACE_GROUP)
                if self.do_stoichiometry_sg_check:
                    self.space_group.set_n_atoms_compatibility_dict(
                        self.composition.state
                    )
        elif stage == Stage.SPACE_GROUP:
            stage_group_action = self._depad_action(action, Stage.SPACE_GROUP)
            _, executed_action, valid = self.space_group.step(stage_group_action)
            if valid and executed_action == self.space_group.eos:
                self._set_stage(Stage.LATTICE_PARAMETERS)
                self._set_lattice_parameters()
        elif stage == Stage.LATTICE_PARAMETERS:
            lattice_parameters_action = self._depad_action(
                action, Stage.LATTICE_PARAMETERS
            )
            _, executed_action, valid = self.lattice_parameters.step(
                lattice_parameters_action
            )
            if valid and executed_action == self.lattice_parameters.eos:
                self.done = True
        else:
            raise ValueError(f"Unrecognized stage {stage}.")

        self._update_state()

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
                [1] + self.composition.state + substate + [0] * 6
            )  # hard-code LatticeParameters` source, since it can change with other lattice system
        elif stage == Stage.LATTICE_PARAMETERS:
            output = [2] + self.composition.state + self.space_group.state + substate
        else:
            raise ValueError(f"Unrecognized stage {stage}.")

        return output

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
            TODO: to be stateless (meaning, operating as a function, not a method with
            current object context) this needs to set lattice system based on the passed
            state only. Right now it uses the current LatticeParameter environment, in
            particular the lattice system that it was set to, and that changes the invalid
            actions mask.

            If for some reason a state will be passed to this method that describes an
            object with different lattice system than what self.lattice_system contains,
            the result will be invalid.
            """
            parents, actions = self.lattice_parameters.get_parents(
                state=self._get_lattice_parameters_state(state), done=done
            )
            parents = [self._build_state(p, Stage.LATTICE_PARAMETERS) for p in parents]
            actions = [self._pad_action(a, Stage.LATTICE_PARAMETERS) for a in actions]
        else:
            raise ValueError(f"Unrecognized stage {stage}.")

        return parents, actions

    def states2proxy(
        self, states: Union[List[List], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in "environment format" for the proxy: simply a
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
        states = tlong(states, device=self.device)
        composition_proxy_states = self.composition.states2proxy(
            self._get_composition_tensor_states(states)
        ).to(self.device)
        space_group_proxy_states = self.space_group.states2proxy(
            self._get_space_group_tensor_states(states)
        ).to(self.device)
        lattice_parameters_proxy_states = self.lattice_parameters.states2proxy(
            self._get_lattice_parameters_tensor_states(states)
        ).to(self.device)
        return torch.cat(
            [
                composition_proxy_states,
                space_group_proxy_states,
                lattice_parameters_proxy_states,
            ],
            dim=1,
        )

    def set_state(self, state: List, done: Optional[bool] = False):
        super().set_state(state, done)

        stage = self._get_stage(state)

        composition_done = stage in [Stage.SPACE_GROUP, Stage.LATTICE_PARAMETERS]
        space_group_done = stage == Stage.LATTICE_PARAMETERS
        lattice_parameters_done = done

        self.composition.set_state(self._get_composition_state(state), composition_done)
        self.space_group.set_state(self._get_space_group_state(state), space_group_done)
        self.lattice_parameters.set_state(
            self._get_lattice_parameters_state(state), lattice_parameters_done
        )

        """
        We synchronize LatticeParameter's lattice system with the one of SpaceGroup
        (if it was set) or reset it to the default triclinic otherwise. Why this is 
        needed:
        1) the first case is necessary for backward sampling, where we start from
           an arbitrary terminal state, and need to synchronize the LatticeParameter's
           lattice system to what that state indicates,
        2) the second case is also necessary in backward sampling, but when we 
           transition from Stage.LATTICE_PARAMETERS to Stage.SPACE_GROUP. We then need
           to reset the lattice system to the default triclinic, such that its
           source is back to the original one, and corresponds to the source of the
           general Crystal environment.
        """
        lattice_system = self.space_group.lattice_system
        if lattice_system != "None":
            self.lattice_parameters.lattice_system = lattice_system
        else:
            self.lattice_parameters.lattice_system = TRICLINIC
        self.lattice_parameters._set_source()

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
