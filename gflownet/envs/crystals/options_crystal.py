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
from gflownet.envs.options import Options

from gflownet.utils.crystals.constants import (
    CUBIC,
    HEXAGONAL,
    MONOCLINIC,
    ORTHORHOMBIC,
    RHOMBOHEDRAL,
    TETRAGONAL,
    TRICLINIC,
)

class OptionsCrystal(Stack):
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
        do_sg_to_lp_constraints: bool = True,
        space_group_kwargs: Optional[Dict] = None,
        lattice_parameters_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        self.do_spacegroup = do_spacegroup
        self.do_lattice_parameters = do_lattice_parameters
        self.do_projected_lattice_parameters = do_projected_lattice_parameters
        self.do_sg_to_lp_constraints = do_sg_to_lp_constraints

        self.lattice_parameters_kwargs = lattice_parameters_kwargs or {}

        # Initialize list of subenvs:
        subenvs = []

        space_group = Options(options=["Option-136","Option-221"],n_options=2)
        subenvs.append(space_group)
        self.stage_spacegroup = 0
        
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
            
        # Apply constraints:
        # - space group -> lattice parameters
        # Apply constraint only if action is None or if it is the space group EOS
        if self.space_group.done and (
            action is None or self._depad_action(action) == self.space_group.eos
            and self.do_sg_to_lp_constraints 
        ):
            if self.space_group.state[0] == 1:
                self.lattice_parameters.set_lattice_system(TETRAGONAL)
            elif self.space_group.state[0] == 2:
                self.lattice_parameters.set_lattice_system(CUBIC)
            else:
                raise NotImplementedError(f"Unknown option {self.space_group.state}")

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
            and self.do_sg_to_lp_constraints
        ):
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
        stages = [
            self.stage_spacegroup,
            self.stage_latticeparameters,
        ]
        return torch.cat(
            [
                self.subenvs[stage].states2proxy([state[stage + 1] for state in states])
                for stage in stages
            ],
            dim=1,
        )

    def _print_state(self, state: Optional[List] = None):
        """
        Prints a state in more human-readable format, for debugging purposes.
        """
        if state is None:
            state = self.state
        for stage, subenv in self.subenvs.items():
            print(f"Stage {stage}")
            print(self._get_substate(state, stage))

    def states2policy(self, states):
        return super().states2policy(states)