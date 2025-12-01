

# I think this will be a Stack (like Crystal) of a Grid with dimension 1 and length 2 and a Cube
"""
This implementation uses the Stack meta-environment and the continuous Lattice
Parameters environment. Alternative implementations preceded this one but have been
removed for simplicity. Check commit 9f3477d8e46c4624f9162d755663993b83196546 to see
these changes or the history previous to that commit to consult previous
implementations.
"""
import copy
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torchtyping import TensorType
from tqdm import tqdm

from gflownet.envs.grid import Grid
from gflownet.envs.options import Options
from gflownet.envs.cube import ContinuousCube
from gflownet.envs.stack import Stack
from gflownet.utils.common import copy


class FakeCrystal(Stack):
    """
    A combination of Composition, SpaceGroup and LatticeParameters into a single
    environment. Works sequentially, by first filling in the Composition, then
    SpaceGroup, and finally LatticeParameters.

    Attributes
    ----------

    """

    def __init__(
        self,
        do_space_group: bool = True,
        do_lattice_parameters: bool = True,
        use_constraints: bool = False,
        constraints_dict: dict = {},
        cube_dim: int = 2,
        **kwargs,
    ):
        self.do_space_group = do_space_group
        self.do_lattice_parameters = do_lattice_parameters
        self.use_constraints = use_constraints
        self.constraints_dict = constraints_dict
        # self.composition_kwargs = dict( # setup the kwargs later
        #     composition_kwargs or {},
        #     do_spacegroup_check=self.do_sg_to_composition_constraints,
        # )
        # self.space_group_kwargs = space_group_kwargs or {}
        # self.lattice_parameters_kwargs = lattice_parameters_kwargs or {}

        # Initialize list of subenvs:
        subenvs = []

        if not self.do_space_group and not self.do_lattice_parameters:
            raise NotImplementedError(f"One of {self.do_space_group=} and {self.do_lattice_parameters=} needs to be True")

        if self.do_space_group:
            self.space_group = Options(options=["Option-136","Option-221"],n_options=2)# Grid(n_dim=1,length=2)
            subenvs.append(self.space_group)

        if self.do_lattice_parameters:
            self.lattice_params = ContinuousCube(ndim=cube_dim)
            subenvs.append(self.lattice_params)
        
        # Initialize base Stack environment
        super().__init__(subenvs=tuple(subenvs), **kwargs)

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
        
        if self.use_constraints and \
            self.do_space_group and self.do_lattice_parameters \
            and self.space_group.done and (
            action is None or self._depad_action(action) == self.space_group.eos
        ):  
            ignored_dims = self.constraints_dict[self.space_group.state[0]]
            self.lattice_params.ignored_dims = [d for d in ignored_dims] # this is fast but assigning directly is slow

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

        if (self.use_constraints and \
            self.do_space_group and self.do_lattice_parameters 
            and not self.space_group.done
            and (action is None or self._depad_action(action) == self.space_group.eos)
        ):
            self.lattice_params.ignored_dims = [False, False]

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

    def _print_state(self, state: Optional[List] = None):
        """
        Prints a state in more human-readable format, for debugging purposes.
        """
        if state is None:
            state = self.state
        for stage, subenv in self.subenvs.items():
            print(f"Stage {stage}")
            print(self._get_substate(state, stage))


    def _check_has_constraints(self) -> bool:
        """
        Checks whether Crystal implements any constraints across sub-environments.

        It returns True if any of the possible constraints is implemented.

        Returns
        -------
        bool
            True if the Crystal has constraints, False otherwise
        """
        return True
