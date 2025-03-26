"""
This implementation uses the Stack meta-environment and the continuous Lattice
Parameters environment. Alternative implementations preceded this one but have been
removed for simplicity. Check commit 9f3477d8e46c4624f9162d755663993b83196546 to see
these changes or the history previous to that commit to consult previous
implementations.
"""

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torchtyping import TensorType
from tqdm import tqdm

from gflownet.envs.crystals.composition import Composition
from gflownet.envs.crystals.lattice_parameters import PARAMETER_NAMES, LatticeParameters
from gflownet.envs.crystals.spacegroup import SpaceGroup
from gflownet.envs.stack import Stack
from gflownet.utils.common import copy
from gflownet.utils.crystals.constants import TRICLINIC


class Crystal(Stack):
    """
    A combination of Composition, SpaceGroup and LatticeParameters into a single
    environment. Works sequentially, by first filling in the Composition, then
    SpaceGroup, and finally LatticeParameters.

    Attributes
    ----------
    do_spacegroup : bool
        Whether to include the SpaceGroup as a sub-environment and thus sample the
        space group of the crystal.
    do_lattice_parameters : bool
        Whether to include the LatticeParameters as a sub-environment and thus sample
        the lattice parameters (a, b, c, α, β, γ) of the crystal.
    do_sg_before_composition : bool
        Whether the SpaceGroup sub-environment should precede the composition.
    do_composition_to_sg_constraints : bool
        Whether to apply constraints on the space group sub-environment, based on the
        composition, in case the composition goes first.
    do_sg_to_composition_constraints : bool
        Whether to apply constraints on the composition sub-environment, based on the
        space group, in case the space group goes first.
    do_sg_to_lp_constraints : bool
        Whether to apply constraints on the lattice parameters sub-environment, based
        on the space group.
    composition_kwargs : dict
        An optional dictionary with configuration to be passed to the Composition
        sub-environment.
    space_group_kwargs : dict
        An optional dictionary with configuration to be passed to the SpaceGroup
        sub-environment.
    lattice_parameters_kwargs : dict
        An optional dictionary with configuration to be passed to the LatticeParameters
        sub-environment.
    """

    def __init__(
        self,
        do_spacegroup: bool = True,
        do_lattice_parameters: bool = True,
        do_sg_before_composition: bool = True,
        do_composition_to_sg_constraints: bool = True,
        do_sg_to_composition_constraints: bool = True,
        do_sg_to_lp_constraints: bool = True,
        composition_kwargs: Optional[Dict] = None,
        space_group_kwargs: Optional[Dict] = None,
        lattice_parameters_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        self.do_spacegroup = do_spacegroup
        self.do_lattice_parameters = do_lattice_parameters

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

        # Initialize list of subenvs:
        subenvs = []

        if self.do_spacegroup:
            space_group = SpaceGroup(**self.space_group_kwargs)
            if self.do_sg_before_composition:
                subenvs.append(space_group)
                self.stage_spacegroup = 0
                self.stage_composition = 1
        else:
            space_group = None
        composition = Composition(**self.composition_kwargs)
        subenvs.append(composition)
        if not self.do_sg_before_composition and space_group is not None:
            subenvs.append(space_group)
            self.stage_composition = 0
            self.stage_spacegroup = 1
        if self.do_lattice_parameters:
            # We initialize lattice parameters with triclinic lattice system as it is
            # the most general one, but it will have to be reinitialized using proper
            # lattice system from space group once that is determined.
            lattice_parameters = LatticeParameters(
                lattice_system=TRICLINIC, **self.lattice_parameters_kwargs
            )
            subenvs.append(lattice_parameters)
            self.stage_latticeparameters = 2

        # Initialize base Stack environment
        super().__init__(subenvs=tuple(subenvs), **kwargs)

    @property
    def composition(self) -> Union[Composition]:
        """
        Returns the sub-environment corresponding to the composition.

        Returns
        -------
        Composition or None
        """
        if hasattr(self, "stage_composition"):
            return self.subenvs[self.stage_composition]
        return None

    @property
    def space_group(self) -> SpaceGroup:
        """
        Returns the sub-environment corresponding to the space group.

        Returns
        -------
        SpaceGroup or None
        """
        if hasattr(self, "stage_spacegroup"):
            return self.subenvs[self.stage_spacegroup]
        return None

    @property
    def lattice_parameters(self) -> LatticeParameters:
        """
        Returns the sub-environment corresponding to the lattice parameters.

        Returns
        -------
        LatticeParameters or None
        """
        if hasattr(self, "stage_latticeparameters"):
            return self.subenvs[self.stage_latticeparameters]
        return None

    def _check_has_constraints(self) -> bool:
        """
        Checks whether Crystal implements any constraints across sub-environments.

        It returns True if any of the possible constraints is implemented.

        Returns
        -------
        bool
            True if the Crystal has constraints, False otherwise
        """
        return any(
            [
                self.do_composition_to_sg_constraints,
                self.do_sg_to_composition_constraints,
                self.do_sg_to_lp_constraints,
            ]
        )

    def _apply_constraints(
        self,
        action: Tuple = None,
        state: Union[List, torch.Tensor] = None,
        dones: List[bool] = None,
        is_backward: bool = False,
    ):
        """
        Applies constraints across sub-environments, when applicable.

        This method simply selects the corresponding forward or backward method to
        apply the constraints.

        Forward:
        - composition -> space group (if composition is first)
        - space group -> composition (if space group is first)
        - space group -> lattice parameters
        Backward:
        - lattice parameters -> space group

        This method is used in step() and set_state().

        Parameters
        ----------
        action : tuple
            An action from the Crystal environment.
        state : list
            A state from the Crystal environment.
        dones : list
            List of boolean values indicating the sub-environments that are done.
        is_backward : bool
            Boolean flag to indicate whether the action is in the backward direction.
        """
        if is_backward:
            self._apply_constraints_backward(action)
        else:
            self._apply_constraints_forward(action, state, dones)

    def _apply_constraints_forward(
        self,
        action: Tuple = None,
        state: Union[List, torch.Tensor] = None,
        dones: List[bool] = None,
    ):
        """
        Applies constraints across sub-environments, when applicable, in the forward
        direction.

        - composition -> space group (if composition is first)
        - space group -> composition (if space group is first)
        - space group -> lattice parameters

        Parameters
        ----------
        action : tuple
            An action from the Crystal environment.
        state : list or tensor (optional)
            A state from the Crystal environment.
        dones : list
            A list indicating the sub-environments that are done.
        """
        if not self.has_constraints:
            return
        # Apply constraints composition -> space group
        # Apply constraint only if action is None or if it is the composition EOS
        if (
            self.composition.done
            and self.do_composition_to_sg_constraints
            and not self.do_sg_before_composition
            and (action is None or self._depad_action(action) == self.composition.eos)
        ):
            n_atoms_per_element = self.subenvs[
                self.stage_composition
            ].get_n_atoms_per_element(self.composition.state)
            self.space_group.set_n_atoms_compatibility_dict(n_atoms_per_element)

        # Apply constraints:
        # - space group -> composition
        # - space group -> lattice parameters
        # Apply constraint only if action is None or if it is the space group EOS
        if self.space_group.done and (
            action is None or self._depad_action(action) == self.space_group.eos
        ):
            if self.do_sg_before_composition and self.do_sg_to_composition_constraints:
                space_group = self.space_group.space_group
                self.composition.set_space_group(space_group)
            if self.do_sg_to_lp_constraints:
                lattice_system = self.space_group.lattice_system
                self.lattice_parameters.set_lattice_system(lattice_system)

    def _apply_constraints_backward(self, action: Tuple = None):
        """
        Applies constraints across sub-environments, when applicable, in the backward
        direction.

        Parameters
        ----------
        action : tuple
            An action from the Crystal environment.
        """
        if not self.has_constraints:
            return

        # Revert the constraint space group -> lattice parameters
        # Lattice system of LP subenv is set back to TRICLINIC
        # Apply after (backward) EOS of SpaceGroup subenv
        if (
            self.do_spacegroup
            and not self.space_group.done
            and (action is None or self._depad_action(action) == self.space_group.eos)
        ):
            if self.do_sg_to_lp_constraints:
                self.lattice_parameters.set_lattice_system(TRICLINIC)

    def states2proxy(
        self, states: List[List]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: simply a
        concatenation of the proxy-format states of the sub-environments.

        This method is overriden so as to account for the space group before
        composition case, since the proxy expects composition first regardless.

        Args
        ----
        states : list
            A batch of states in environment format.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        if not self.do_sg_before_composition:
            return super().states2proxy(states)
        stages_composition_first = [
            self.stage_composition,
            self.stage_spacegroup,
            self.stage_latticeparameters,
        ]
        return torch.cat(
            [
                self.subenvs[stage].states2proxy([state[stage + 1] for state in states])
                for stage in stages_composition_first
            ],
            dim=1,
        )

    def process_data_set(self, df: pd.DataFrame, progress=False) -> List[List]:
        """
        Converts a data set passed as a pandas DataFrame into a list of states in
        environment format.

        The DataFrame is expected to have the following columns:
            - Formulae: non-reduced formulae of the composition
            - Space Group: international number of the space group
            - a, b, c, alpha, beta, gamma: lattice parameters

        Parameters
        ----------
        df : DataFrame
            A pandas DataFrame containing the necessary columns to represent a crystal
            as described above.
        progress : bool
            Whether to display a progress bar.

        Returns
        -------
        list
            A list of states in environment format.
        """
        data_valid = []
        for row in tqdm(df.iterrows(), total=len(df), disable=not progress):
            # Index 0 is the row index; index 1 is the remaining columns
            row = row[1]
            state = {}
            state[self.stage_composition] = self.subenvs[
                self.stage_composition
            ].readable2state(row["Formulae"])
            state[self.stage_spacegroup] = self.subenvs[
                self.stage_spacegroup
            ]._set_constrained_properties([0, 0, row["Space Group"]])
            state[self.stage_latticeparameters] = self.subenvs[
                self.stage_latticeparameters
            ].parameters2state(tuple(row[list(PARAMETER_NAMES)]))
            is_valid_subenvs = [
                subenv.is_valid(state[stage]) for stage, subenv in self.subenvs.items()
            ]
            if all(is_valid_subenvs):
                # TODO: Consider making stack state a dict which would avoid having to
                # do this, among other advantages
                state_stack = [2] + [state[stage] for stage in self.subenvs]
                data_valid.append(state_stack)
        return data_valid

    def _print_state(self, state: Optional[List] = None):
        """
        Prints a state in more human-readable format, for debugging purposes.
        """
        if state is None:
            state = self.state
        for stage, subenv in self.subenvs.items():
            print(f"Stage {stage}")
            print(self._get_substate(state, stage))
