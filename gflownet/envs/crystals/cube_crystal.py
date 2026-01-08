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
from gflownet.envs.crystals.lattice_parameters import (
    PARAMETER_NAMES,
    LatticeParameters,
    LatticeParametersSGCCG,
)
from gflownet.envs.crystals.spacegroup import SpaceGroup
from gflownet.envs.stack import Stack
from gflownet.utils.common import copy
from gflownet.utils.crystals.constants import TRICLINIC
from gflownet.envs.cube import ContinuousCube

class CubeCrystal(Stack):
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
        do_sg_to_lp_constraints: bool = True,
        space_group_kwargs: Optional[Dict] = None,
        lattice_parameters_kwargs: Optional[Dict] = None,
        constraints_dict: dict = {},
        **kwargs,
    ):
        self.do_sg_to_lp_constraints = do_sg_to_lp_constraints

        self.space_group_kwargs = space_group_kwargs or {}
        self.lattice_parameters_kwargs = lattice_parameters_kwargs or {}
        self.constraints_dict = constraints_dict
        # Initialize list of subenvs:
        subenvs = []

        space_group = SpaceGroup(**self.space_group_kwargs)
        subenvs.append(space_group)
        self.stage_spacegroup = 0
        
        # We initialize lattice parameters with triclinic lattice system as it is
        # the most general one, but it will have to be reinitialized using proper
        # lattice system from space group once that is determined.
        self.lattice_params = ContinuousCube(n_dim=6)
        subenvs.append(self.lattice_params)
        self.stage_latticeparameters = 1

        # Initialize base Stack environment
        super().__init__(subenvs=tuple(subenvs), **kwargs)

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
    def lattice_parameters(self) -> Union[LatticeParameters, LatticeParametersSGCCG]:
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
        return self.do_sg_to_lp_constraints

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
        
        if self.space_group.done and (
            action is None or self._depad_action(action) == self.space_group.eos
        ):  
            ignored_dims = self.constraints_dict[self.space_group.state[2]]
            self.lattice_params.ignored_dims = [d for d in ignored_dims]

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

        if (not self.space_group.done
            and (action is None or self._depad_action(action) == self.space_group.eos)
        ):
            self.lattice_params.ignored_dims = [False] * 6

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
        return super().states2proxy(states)

    # def process_data_set(
    #     self, data: Union[pd.DataFrame, List], progress=False
    # ) -> List[List]:
    #     """
    #     Processes a data set passed as a pandas DataFrame or as a list of states by
    #     filtering out the states that are not valid according to the environment
    #     configuration.

    #     If the input is a DataFrame, the rows are converted into environment states.

    #     Parameters
    #     ----------
    #     data : DataFrame or list
    #         One of the following:
    #             - A pandas DataFrame containing the necessary columns to represent a
    #               crystal as described above.
    #             - A list of states in environment format.
    #     progress : bool
    #         Whether to display a progress bar.

    #     Returns
    #     -------
    #     list
    #         A list of states in environment format.
    #     """
    #     if isinstance(data, pd.DataFrame):
    #         return self._process_dataframe(data, progress)
    #     elif isinstance(data, list) and isinstance(data[0], list):
    #         return self._process_states_list(data, progress)
    #     else:
    #         raise ValueError("Unknown data type")

    # def _process_states_list(self, data: List, progress=False) -> List[List]:
    #     """
    #     Processes a data set passed a list of states in environment format by filtering
    #     out the states that are not valid according to the environment configuration.

    #     Parameters
    #     ----------
    #     data : list
    #         A list of states in environment format.
    #     progress : bool
    #         Whether to display a progress bar.

    #     Returns
    #     -------
    #     list
    #         A list of states in environment format.
    #     """
    #     data_valid = []
    #     for state in tqdm(data, total=len(data), disable=not progress):
    #         # Index 0 is the row index; index 1 is the remaining columns
    #         is_valid_subenvs = [
    #             subenv.is_valid(state[stage + 1])
    #             for stage, subenv in self.subenvs.items()
    #         ]
    #         if all(is_valid_subenvs):
    #             data_valid.append(state)
    #     return data_valid

    # def _process_dataframe(self, df: pd.DataFrame, progress=False) -> List[List]:
    #     """
    #     Converts a data set passed as a pandas DataFrame into a list of states in
    #     environment format.

    #     The DataFrame is expected to have the following columns:
    #         - Formulae: non-reduced formulae of the composition
    #         - Space Group: international number of the space group
    #         - a, b, c, alpha, beta, gamma: lattice parameters

    #     Parameters
    #     ----------
    #     df : DataFrame
    #         A pandas DataFrame containing the necessary columns to represent a crystal
    #         as described above.
    #     progress : bool
    #         Whether to display a progress bar.

    #     Returns
    #     -------
    #     list
    #         A list of states in environment format.
    #     """
    #     data_valid = []
    #     for row in tqdm(df.iterrows(), total=len(df), disable=not progress):
    #         # Index 0 is the row index; index 1 is the remaining columns
    #         row = row[1]
    #         state = {}
    #         state[self.stage_composition] = self.subenvs[
    #             self.stage_composition
    #         ].readable2state(row["Formulae"])
    #         state[self.stage_spacegroup] = self.subenvs[
    #             self.stage_spacegroup
    #         ]._set_constrained_properties([0, 0, row["Space Group"]])
    #         state[self.stage_latticeparameters] = self.subenvs[
    #             self.stage_latticeparameters
    #         ].parameters2state(tuple(row[list(PARAMETER_NAMES)]))
    #         is_valid_subenvs = [
    #             subenv.is_valid(state[stage]) for stage, subenv in self.subenvs.items()
    #         ]
    #         if all(is_valid_subenvs):
    #             # TODO: Consider making stack state a dict which would avoid having to
    #             # do this, among other advantages
    #             state_stack = [2] + [state[stage] for stage in self.subenvs]
    #             data_valid.append(state_stack)
    #     return data_valid

    def _print_state(self, state: Optional[List] = None):
        """
        Prints a state in more human-readable format, for debugging purposes.
        """
        if state is None:
            state = self.state
        for stage, subenv in self.subenvs.items():
            print(f"Stage {stage}")
            print(self._get_substate(state, stage))
