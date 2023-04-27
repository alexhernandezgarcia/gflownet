from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.crystals.composition import Composition
from gflownet.envs.crystals.lattice_parameters import LatticeParameters
from gflownet.envs.crystals.spacegroup import SpaceGroup
from gflownet.utils.crystals.constants import TRICLINIC


class Stage(Enum):
    COMPOSITION = 0
    SPACE_GROUP = 1
    LATTICE_PARAMETERS = 2


class Crystal(GFlowNetEnv):
    """
    A combination of Composition, SpaceGroup and LatticeParameters into a single environment.
    Works sequentially, by first filling in the composition, then SpaceGroup, and finally
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

        if len(self.source) != len(set(self.source)):
            raise ValueError(
                "Detected duplicate actions between different components of Crystal environment."
            )

        # start and end indices of individual substates
        self.composition_start = 0
        self.composition_end = len(self.composition.source)
        self.space_group_start = self.composition_end
        self.space_group_end = self.space_group_start + len(self.space_group.source)
        self.lattice_parameters_start = self.space_group_end
        self.lattice_parameters_end = self.lattice_parameters_start + len(
            self.lattice_parameters.source
        )

        self.eos = self.lattice_parameters.eos
        self.stage = Stage.COMPOSITION
        self.done = False

        super().__init__(**kwargs)

    def _set_lattice_parameters(self):
        crystal_system = self.space_group.get_crystal_system()

        if crystal_system is None:
            raise ValueError(
                "Cannot set lattice parameters without crystal system determined in the space group."
            )

        self.lattice_parameters = LatticeParameters(
            lattice_system=self.space_group.lattice_system,
            **self.lattice_parameters_kwargs,
        )

    def get_action_space(self) -> List:
        return (
            self.composition.action_space
            + self.space_group.action_space
            + self.lattice_parameters.action_space
        )

    def get_max_traj_length(self):
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

    def _get_composition_state(self, state=None):
        if state is None:
            state = self.state.copy()

        return state[self.composition_start : self.composition_end]

    def _get_space_group_state(self, state=None):
        if state is None:
            state = self.state.copy()

        return state[self.space_group_start : self.space_group_end]

    def _get_lattice_parameters_state(self, state=None):
        if state is None:
            state = self.state.copy()

        return state[self.lattice_parameters_start : self.lattice_parameters_end]

    def get_mask_invalid_actions_forward(self, state=None, done=None):
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
            mask[self.composition_start : self.composition_end] = composition_mask
        elif self.stage == Stage.SPACE_GROUP:
            space_group_mask = self.space_group.get_mask_invalid_actions_forward(
                state=self._get_space_group_state(state)
            )
            mask[self.space_group_start : self.space_group_end] = space_group_mask
        elif self.stage == Stage.LATTICE_PARAMETERS:
            lattice_parameters_mask = (
                self.lattice_parameters.get_mask_invalid_actions_forward(
                    state=self._get_lattice_parameters_state(state)
                )
            )
            mask[
                self.lattice_parameters_start : self.lattice_parameters_end
            ] = lattice_parameters_mask
        else:
            raise ValueError(f"Unrecognized stage {self.stage}.")

        return mask

    def _update_state(self):
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
            executed_action, state, valid = self.composition.step(action)
            assert executed_action == action
            if valid and state == self.composition.eos:
                self.stage = Stage.SPACE_GROUP
        elif self.stage == Stage.SPACE_GROUP:
            executed_action, state, valid = self.space_group.step(action)
            assert executed_action == action
            if valid and state == self.space_group.eos:
                self.stage = Stage.LATTICE_PARAMETERS
                self._set_lattice_parameters()
        elif self.stage == Stage.LATTICE_PARAMETERS:
            executed_action, state, valid = self.lattice_parameters.step(action)
            assert executed_action == action
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
        elif self.stage == Stage.LATTICE_PARAMETERS:
            parents, actions = self.lattice_parameters.get_parents(
                state=self._get_lattice_parameters_state(state)
            )
            parents = [
                self.composition.state + self.space_group.state + p for p in parents
            ]
        else:
            raise ValueError(f"Unrecognized stage {self.stage}.")

        return parents, actions

    def state2oracle(self, state: Optional[List[int]] = None) -> Tensor:
        """
        Prepares a list of states in "GFlowNet format" for the oracle.

        Args
        ----
        state : list
            A state.

        Returns
        ----
        oracle_state : Tensor
            Tensor containing lengths and angles converted from the Grid format.
        """
        if state is None:
            state = self.state.copy()

        composition_oracle_state = self.composition.state2oracle(
            state=self._get_composition_state(state)
        )
        space_group_oracle_state = self.space_group.state2oracle(
            state=self._get_space_group_state(state)
        )
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
