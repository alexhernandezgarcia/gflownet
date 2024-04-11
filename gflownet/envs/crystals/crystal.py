"""
This implementation uses the Stack meta-environment and the continuous Lattice
Parameters environment. Alternative implementations preceded this one but have been
removed for simplicity. Check commit 9f3477d8e46c4624f9162d755663993b83196546 to see
these changes or the history previous to that commit to consult previous
implementations.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torchtyping import TensorType
from tqdm import tqdm

from gflownet.envs.crystals.composition import Composition
from gflownet.envs.crystals.lattice_parameters import PARAMETER_NAMES, LatticeParameters
from gflownet.envs.crystals.spacegroup import SpaceGroup
from gflownet.envs.stack import Stack
from gflownet.utils.crystals.constants import TRICLINIC


class Crystal(Stack):
    """
    A combination of Composition, SpaceGroup and LatticeParameters into a single
    environment. Works sequentially, by first filling in the Composition, then
    SpaceGroup, and finally LatticeParameters.
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
        lattice_parameters = LatticeParameters(
            lattice_system=TRICLINIC, **self.lattice_parameters_kwargs
        )

        # Initialize tuple of subenvs and stage indices
        if self.do_sg_before_composition:
            subenvs = (space_group, composition, lattice_parameters)
            self.stage_spacegroup = 0
            self.stage_composition = 1
        else:
            subenvs = (composition, space_group, lattice_parameters)
            self.stage_composition = 0
            self.stage_spacegroup = 1
        self.stage_latticeparameters = 2

        # Initialize base Stack environment
        super().__init__(subenvs=subenvs, **kwargs)

    def _apply_constraints(self, action: Tuple = None):
        """
        Applies constraints across sub-environments, when applicable.

        - composition -> space group (if composition is first)
        - space group -> composition (if space group is first)
        - space group -> lattice parameters

        This method is used in step() and set_state().
        """
        # Apply constraints composition -> space group
        # Apply constraint only if action is None or if it is the composition EOS
        if (
            self.subenvs[self.stage_composition].done
            and self.do_composition_to_sg_constraints
            and not self.do_sg_before_composition
            and (
                action is None
                or self._depad_action(action)
                == self.subenvs[self.stage_composition].eos
            )
        ):
            self.subenvs[self.stage_spacegroup].set_n_atoms_compatibility_dict(
                self.subenvs[self.stage_composition].state
            )

        # Apply constraints:
        # - space group -> composition
        # - space group -> lattice parameters
        # Apply constraint only if action is None or if it is the space group EOS
        if self.subenvs[self.stage_spacegroup].done and (
            action is None
            or self._depad_action(action) == self.subenvs[self.stage_spacegroup].eos
        ):
            if self.do_sg_before_composition and self.do_sg_to_composition_constraints:
                space_group = self.subenvs[self.stage_spacegroup].space_group
                self.subenvs[self.stage_composition].space_group = space_group
            if self.do_sg_to_lp_constraints:
                lattice_system = self.subenvs[self.stage_spacegroup].lattice_system
                self.subenvs[self.stage_latticeparameters].set_lattice_system(
                    lattice_system
                )

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
