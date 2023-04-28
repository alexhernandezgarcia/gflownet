from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.crystals.composition import Composition
from gflownet.envs.crystals.lattice_parameters import LatticeParameters
from gflownet.envs.crystals.spacegroup import SpaceGroup
from gflownet.utils.crystals.constants import TRICLINIC


class Stage(Enum):
    """
    In addition to encoding current stage, values of this enum are used for padding individual
    component environment's actions (to ensure they have the same length for tensorization).
    """

    COMPOSITION = -2
    SPACE_GROUP = -3
    LATTICE_PARAMETERS = -4


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
        **kwargs,
    ):
        self.composition_kwargs = composition_kwargs or {}
        self.space_group_kwargs = space_group_kwargs or {}
        self.lattice_parameters_kwargs = lattice_parameters_kwargs or {}

        self.composition = Composition(**self.composition_kwargs)
        self.space_group = SpaceGroup(**self.space_group_kwargs)
        # We initialize lattice parameters here with triclinic lattice system to access
        # all the methods of that environment, but it will have to be reinitialized using
        # proper lattice system from space group once that is determined.
        # Triclinic was used because it doesn't force any initial starting angles.
        self.lattice_parameters = LatticeParameters(
            lattice_system=TRICLINIC, **self.lattice_parameters_kwargs
        )

        self.source = (
            self.composition.source
            + self.space_group.source
            + self.lattice_parameters.source
        )

        # start and end indices of individual substates
        self.composition_state_start = 0
        self.composition_state_end = len(self.composition.source)
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
        self.composition_mask_end = len(self.composition.action_space)
        self.space_group_mask_start = self.composition_mask_end
        self.space_group_mask_end = self.space_group_mask_start + len(
            self.space_group.action_space
        )
        self.lattice_parameters_mask_start = self.space_group_mask_end
        self.lattice_parameters_mask_end = self.lattice_parameters_mask_start + len(
            self.lattice_parameters.action_space
        )

        self.eos = self.lattice_parameters.eos
        self.stage = Stage.COMPOSITION
        self.max_action_length = max(
            max(len(a) for a in self.composition.action_space),
            max(len(a) for a in self.space_group.action_space),
            max(len(a) for a in self.lattice_parameters.action_space),
        )
        self.done = False

        # Conversions
        self.state2proxy = self.state2oracle
        self.statebatch2proxy = self.statebatch2oracle
        self.statetorch2proxy = self.statetorch2oracle

        super().__init__(**kwargs)

    def _set_lattice_parameters(self):
        """
        Sets LatticeParameters conditioned on the lattice system derived from the SpaceGroup.
        """
        crystal_system = self.space_group.get_crystal_system()

        if crystal_system is None:
            raise ValueError(
                "Cannot set lattice parameters without crystal system determined in the space group."
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
        return action + (stage.value,) * (self.max_action_length - len(action))

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
            dim = max(len(a) for a in self.composition.action_space)
        elif stage == Stage.SPACE_GROUP:
            dim = max(len(a) for a in self.space_group.action_space)
        elif stage == Stage.LATTICE_PARAMETERS:
            dim = max(len(a) for a in self.lattice_parameters.action_space)
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
        self.stage = Stage.COMPOSITION

        super().reset(env_id=env_id)

        return self

    def _get_composition_state(self, state: Optional[List[int]] = None) -> List[int]:
        if state is None:
            state = self.state.copy()

        return state[self.composition_state_start : self.composition_state_end]

    def _get_composition_tensor_states(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_oracle_dim"]:
        return states[:, self.composition_state_start : self.composition_state_end]

    def _get_space_group_state(self, state: Optional[List[int]] = None) -> List[int]:
        if state is None:
            state = self.state.copy()

        return state[self.space_group_state_start : self.space_group_state_end]

    def _get_space_group_tensor_states(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_oracle_dim"]:
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
    ) -> TensorType["batch", "state_oracle_dim"]:
        return states[
            :, self.lattice_parameters_state_start : self.lattice_parameters_state_end
        ]

    def get_mask_invalid_actions_forward(
        self, state: Optional[List[int]] = None, done: Optional[bool] = None
    ) -> List[bool]:
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done

        if done:
            return [True for _ in range(self.action_space_dim)]

        mask = [True for _ in range(self.action_space_dim)]

        if self.stage == Stage.COMPOSITION:
            composition_mask = self.composition.get_mask_invalid_actions_forward(
                state=self._get_composition_state(state)
            )
            mask[
                self.composition_mask_start : self.composition_mask_end
            ] = composition_mask
        elif self.stage == Stage.SPACE_GROUP:
            space_group_mask = self.space_group.get_mask_invalid_actions_forward(
                state=self._get_space_group_state(state)
            )
            mask[
                self.space_group_mask_start : self.space_group_mask_end
            ] = space_group_mask
        elif self.stage == Stage.LATTICE_PARAMETERS:
            lattice_parameters_mask = (
                self.lattice_parameters.get_mask_invalid_actions_forward(
                    state=self._get_lattice_parameters_state(state)
                )
            )
            mask[
                self.lattice_parameters_mask_start : self.lattice_parameters_mask_end
            ] = lattice_parameters_mask
        else:
            raise ValueError(f"Unrecognized stage {self.stage}.")

        return mask

    def _update_state(self):
        """
        Updates current state based on the states of underlying environments.
        """
        self.state = (
            self.composition.state
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

        if self.stage == Stage.COMPOSITION:
            composition_action = self._depad_action(action, Stage.COMPOSITION)
            _, state, valid = self.composition.step(composition_action)
            if valid and state == self.composition.eos:
                self.stage = Stage.SPACE_GROUP
        elif self.stage == Stage.SPACE_GROUP:
            stage_group_action = self._depad_action(action, Stage.SPACE_GROUP)
            _, state, valid = self.space_group.step(stage_group_action)
            if valid and state == self.space_group.eos:
                self.stage = Stage.LATTICE_PARAMETERS
                self._set_lattice_parameters()
        elif self.stage == Stage.LATTICE_PARAMETERS:
            lattice_parameters_action = self._depad_action(
                action, Stage.LATTICE_PARAMETERS
            )
            _, state, valid = self.lattice_parameters.step(lattice_parameters_action)
            if valid and state == self.space_group.eos:
                self.done = True
        else:
            raise ValueError(f"Unrecognized stage {self.stage}.")

        self._update_state()

        return self.state, action, valid

    def get_parents(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]

        if self.stage == Stage.COMPOSITION or (
            self.stage == Stage.SPACE_GROUP
            and self.space_group.state == self.space_group.source
        ):
            parents, actions = self.composition.get_parents(
                state=self._get_composition_state(state)
            )
            parents = [
                p + self.space_group.state + self.lattice_parameters.state
                for p in parents
            ]
            actions = [self._pad_action(a, Stage.COMPOSITION) for a in actions]
        elif self.stage == Stage.SPACE_GROUP or (
            self.stage == Stage.LATTICE_PARAMETERS
            and self.lattice_parameters.state == self.lattice_parameters.source
        ):
            parents, actions = self.space_group.get_parents(
                state=self._get_space_group_state(state)
            )
            parents = [
                self.composition.state + p + self.lattice_parameters.state
                for p in parents
            ]
            actions = [self._pad_action(a, Stage.SPACE_GROUP) for a in actions]
        elif self.stage == Stage.LATTICE_PARAMETERS:
            parents, actions = self.lattice_parameters.get_parents(
                state=self._get_lattice_parameters_state(state)
            )
            parents = [
                self.composition.state + self.space_group.state + p for p in parents
            ]
            actions = [self._pad_action(a, Stage.LATTICE_PARAMETERS) for a in actions]
        else:
            raise ValueError(f"Unrecognized stage {self.stage}.")

        return parents, actions

    def state2oracle(self, state: Optional[List[int]] = None) -> Tensor:
        """
        Prepares a list of states in "GFlowNet format" for the oracle.
        """
        if state is None:
            state = self.state.copy()

        composition_oracle_state = self.composition.state2oracle(
            state=self._get_composition_state(state)
        )
        space_group_oracle_state = self.space_group.state2oracle(
            state=self._get_space_group_state(state)
        ).unsqueeze(
            -1
        )  # StateGroup oracle state is a single number
        lattice_parameters_oracle_state = self.lattice_parameters.state2oracle(
            state=self._get_lattice_parameters_state(state)
        )

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
        )
        space_group_oracle_states = self.space_group.statetorch2oracle(
            self._get_space_group_tensor_states(states)
        )
        lattice_parameters_oracle_states = self.lattice_parameters.statetorch2oracle(
            self._get_lattice_parameters_tensor_states(states)
        )
        return torch.cat(
            [
                composition_oracle_states,
                space_group_oracle_states,
                lattice_parameters_oracle_states,
            ],
            dim=1,
        )

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
            f"Composition = {composition_readable}, "
            f"SpaceGroup = {space_group_readable}, "
            f"LatticeParameters = {lattice_parameters_readable}"
        )