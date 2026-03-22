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

from gflownet.envs.composite.stack import Stack
from gflownet.envs.crystals.composition import Composition
from gflownet.envs.crystals.lattice_parameters import (
    PARAMETER_NAMES,
    LatticeParameters,
    LatticeParametersSGCCG,
)
from gflownet.envs.crystals.spacegroup import SpaceGroup
from gflownet.utils.common import copy
from gflownet.utils.crystals.constants import LATTICE_SYSTEMS, TRICLINIC


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
    do_projected_lattice_parameters : bool
        If True, the LatticeParametersSGCCG environment is used instead of
        LatticeParameters. The latter operates in the natural space of the lattice
        parameters, while the former operates in a projection which ensures the
        validity of the angles. By default, LatticeParameters is used because
        LatticeParametersSGCCG does not currently allow to set constraints of min and
        max lengths and angles and it slows down the run time. Note that the default
        natural LatticeParameters can generate angles with potentially invalid volumes.
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
        do_projected_lattice_parameters: bool = False,
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
        self.do_projected_lattice_parameters = do_projected_lattice_parameters

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
                self.idx_spacegroup = 0
                self.idx_composition = 1
        else:
            space_group = None
        composition = Composition(**self.composition_kwargs)
        subenvs.append(composition)
        if not self.do_sg_before_composition and space_group is not None:
            subenvs.append(space_group)
            self.idx_composition = 0
            self.idx_spacegroup = 1
        if self.do_lattice_parameters:
            # We initialize lattice parameters with triclinic lattice system as it is
            # the most general one, but it will have to be reinitialized using proper
            # lattice system from space group once that is determined.
            if self.do_projected_lattice_parameters:
                lattice_parameters = LatticeParametersSGCCG(
                    lattice_system=TRICLINIC, **self.lattice_parameters_kwargs
                )
            else:
                lattice_parameters = LatticeParameters(
                    lattice_system=TRICLINIC, **self.lattice_parameters_kwargs
                )
            subenvs.append(lattice_parameters)
            self.idx_latticeparameters = 2

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
        if hasattr(self, "idx_composition"):
            return self.subenvs[self.idx_composition]
        return None

    @property
    def space_group(self) -> SpaceGroup:
        """
        Returns the sub-environment corresponding to the space group.

        Returns
        -------
        SpaceGroup or None
        """
        if hasattr(self, "idx_spacegroup"):
            return self.subenvs[self.idx_spacegroup]
        return None

    @property
    def lattice_parameters(self) -> Union[LatticeParameters, LatticeParametersSGCCG]:
        """
        Returns the sub-environment corresponding to the lattice parameters.

        Returns
        -------
        LatticeParameters or None
        """
        if hasattr(self, "idx_latticeparameters"):
            return self.subenvs[self.idx_latticeparameters]
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

    def _apply_constraints_forward(
        self,
        action: Optional[Tuple] = None,
        state: Optional[Dict] = None,
    ) -> bool:
        """
        Applies constraints across sub-environments, when applicable, in the forward
        direction.

        - composition -> space group (if composition is first)
        - space group -> composition (if space group is first)
        - space group -> lattice parameters

        Parameters
        ----------
        action : tuple (optional)
            An action from the Crystal environment or None.
        state : dict (optional)
            A state from the Crystal environment or None.

        Returns
        -------
        bool
            True if any constraint was applied; False otherwise.
        """
        applied_constraints = False
        # Apply constraints composition -> space group
        # Apply constraint only if action is None or if it is the composition EOS
        if (
            self.do_composition_to_sg_constraints
            and not self.do_sg_before_composition
            and self._do_constraints_for_subenv(
                state, self.idx_composition, action, is_backward=False
            )
        ):
            applied_constraints = True
            state = self._get_state(state)
            composition_substate = self._get_substate(state, self.idx_composition)
            n_atoms_per_element = self.composition.get_n_atoms_per_element(
                composition_substate
            )
            self.space_group.set_n_atoms_compatibility_dict(n_atoms_per_element)

        # Apply constraints:
        # - space group -> composition
        # - space group -> lattice parameters
        # Apply constraint only if action is None or if it is the space group EOS
        if self._do_constraints_for_subenv(
            state, self.idx_spacegroup, action, is_backward=False
        ):
            applied_constraints = True
            state = self._get_state(state)
            spacegroup_substate = self._get_substate(state, self.idx_spacegroup)
            if self.do_sg_before_composition and self.do_sg_to_composition_constraints:
                space_group = self.space_group.get_space_group(spacegroup_substate)
                self.composition.set_space_group(space_group)
            if self.do_sg_to_lp_constraints:
                lattice_system = self.space_group.get_lattice_system(
                    spacegroup_substate
                )
                self.lattice_parameters.set_lattice_system(lattice_system)
                self._set_substate(
                    self.idx_latticeparameters, self.lattice_parameters.state, state
                )

        return applied_constraints

    def _apply_constraints_backward(
        self, action: Optional[Tuple] = None, state: Optional[Dict] = None
    ) -> bool:
        """
        Applies constraints across sub-environments, when applicable, in the backward
        direction.

        Parameters
        ----------
        action : tuple (optional)
            An action from the Crystal environment or None.
        state : dict (optional)
            A state from the Crystal environment or None.

        Returns
        -------
        bool
            True if any constraint was applied; False otherwise.
        """
        applied_constraints = False
        # Revert constraints:
        # - space group -> lattice parameters: lattice system of LatticeParameters is
        # set back to TRICLINIC
        # - space group -> composition: space group of Composition is set back to None
        # Apply constraint only if action is None or if it is the space group EOS
        if (
            self.do_spacegroup
            and self.do_sg_to_lp_constraints
            and self._do_constraints_for_subenv(
                state, self.idx_spacegroup, action, is_backward=True
            )
        ):
            applied_constraints = True
            self.lattice_parameters.set_lattice_system(TRICLINIC)
            self._set_substate(
                self.idx_latticeparameters, self.lattice_parameters.state, state
            )
            self.composition.set_space_group(None)

        # Revert constraints composition -> space group: The number of atoms is set
        # back to None
        # Apply constraint only if action is None or if it is the composition EOS
        if (
            self.do_composition_to_sg_constraints
            and not self.do_sg_before_composition
            and self._do_constraints_for_subenv(
                state, self.idx_composition, action, is_backward=True
            )
        ):
            applied_constraints = True
            self.space_group.set_n_atoms_compatibility_dict(None)

        return applied_constraints

    def states2proxy(
        self, states: List[Dict]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in environment format for the proxies.

        The output is the concatenation of the proxy-format states of the
        sub-environments.

        This method is overriden to improve the efficiency, to create a tensor as an
        output and to account for the space group before composition case, since the
        proxy expects composition first regardless.

        Parameters
        ----------
        states : list
            A batch of states in environment format.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        indices_subenvs_proxy = [
            self.idx_composition,
            self.idx_spacegroup,
            self.idx_latticeparameters,
        ]
        return torch.cat(
            [
                self.subenvs[idx].states2proxy([state[idx] for state in states])
                for idx in indices_subenvs_proxy
                if idx is not None
            ],
            dim=1,
        )

    def process_data_set(
        self, data: Union[pd.DataFrame, List], progress=False
    ) -> List[List]:
        """
        Processes a data set passed as a pandas DataFrame or as a list of states by
        filtering out the states that are not valid according to the environment
        configuration.

        If the input is a DataFrame, the rows are converted into environment states.

        Parameters
        ----------
        data : DataFrame or list
            One of the following:
                - A pandas DataFrame containing the necessary columns to represent a
                  crystal as described above.
                - A list of states in environment format.
        progress : bool
            Whether to display a progress bar.

        Returns
        -------
        list
            A list of states in environment format.
        """
        if isinstance(data, pd.DataFrame):
            return self._process_dataframe(data, progress)
        elif isinstance(data, list) and isinstance(data[0], list):
            return self._process_states_list(data, progress)
        else:
            raise ValueError("Unknown data type")

    def _process_states_list(self, data: List, progress=False) -> List[List]:
        """
        Processes a data set passed a list of states in environment format by filtering
        out the states that are not valid according to the environment configuration.

        Parameters
        ----------
        data : list
            A list of states in environment format.
        progress : bool
            Whether to display a progress bar.

        Returns
        -------
        list
            A list of states in environment format.
        """
        data_valid = []
        for state in tqdm(data, total=len(data), disable=not progress):
            # Index 0 is the row index; index 1 is the remaining columns
            is_valid_subenvs = [
                subenv.is_valid(self._get_substate(state, idx))
                for idx, subenv in enumerate(self.subenvs)
            ]
            if all(is_valid_subenvs):
                data_valid.append(state)
        return data_valid

    def _process_dataframe(self, df: pd.DataFrame, progress=False) -> List[List]:
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
            # Composition
            state[self.idx_composition] = self.subenvs[
                self.idx_composition
            ].readable2state(row["Formulae"])
            # Space group
            state[self.idx_spacegroup] = self.subenvs[
                self.idx_spacegroup
            ]._set_constrained_properties([0, 0, row["Space Group"]])
            # Lattice parameters
            lattice_system = self.space_group.get_lattice_system(
                state[self.idx_spacegroup]
            )
            if lattice_system not in LATTICE_SYSTEMS:
                lattice_system = TRICLINIC
            state_lp = copy(self.lattice_parameters.source)
            state_lp = self.lattice_parameters._set_active_subenv(
                self.lattice_parameters.idx_cube, state_lp
            )
            state_lp = self.lattice_parameters.set_lattice_system(
                lattice_system, state_lp
            )
            state_cube = self.lattice_parameters.revert_lattice_constraints(
                tuple(row[list(PARAMETER_NAMES)]), lattice_system
            )
            state[self.idx_latticeparameters] = self.lattice_parameters._set_substate(
                self.lattice_parameters.idx_cube, state_cube, state_lp
            )
            # Check validity
            is_valid_subenvs = [
                subenv.is_valid(self._get_substate(state, idx))
                for idx, subenv in enumerate(self.subenvs)
            ]
            if all(is_valid_subenvs):
                # Add meta-data to state
                state.update(
                    {
                        "_active": self.max_elements - 1,
                        "_envs_unique": self.unique_indices,
                    }
                )
                data_valid.append(state)
        return data_valid

    def _print_state(self, state: Optional[List] = None):
        """
        Prints a state in more human-readable format, for debugging purposes.
        """
        state = self._get_state(state)
        for idx, subenv in enumerate(self.subenvs):
            print(f"Stage {idx}")
            print(self._get_substate(state, idx))
