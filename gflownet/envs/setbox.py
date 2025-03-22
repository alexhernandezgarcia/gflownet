"""
This environment is a conditional set of cubes and grids. The set is flexible in that
in can consist of a variable number of cubes and grids. The number of grids and cubes
for each trajectory can be regarded as the conditions of the set and in this environment
these conditions are sampled by an auxiliary 2D grid environment, where each dimension
will indicate the number of cubes and grids, respectively. Therefore, the overall
environment is a Stack of a Grid and a Set of Cubes and Grids.

This environment was originally designed for debugging the conditional mode of the Set
environment.
"""

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torchtyping import TensorType
from tqdm import tqdm

from gflownet.envs.cube import ContinuousCube
from gflownet.envs.grid import Grid
from gflownet.envs.set import SetFlex
from gflownet.envs.stack import Stack
from gflownet.utils.common import copy, tfloat

# Constants to identify the indices of the Cube and Grid unique environments
IDX_CUBE = 0
IDX_GRID = 1


class SetBox(Stack):
    """
    A Stack of a Grid and a Set of Cubes and Grids. The first grid determines the
    conditions (constraints) of the Set.
    """

    def __init__(
        self,
        max_elements_per_subenv: int = 3,
        n_dim: int = 2,
        cube_kwargs: Optional[Dict] = None,
        grid_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initializes the SetBox environment.

        Parameters
        ----------
        max_elements_per_subenv : int
            The maximum number of elements of each kind in the Set. The total maximum
            number of elements in the set will thus be 2 * max_elements_per_subenv.
        n_dim : int
            The dimensionality of the Cubes and Grids in the Set.
        """
        self.max_elements_per_subenv = max_elements_per_subenv
        self.n_dim = n_dim
        self.cube_kwargs = cube_kwargs or {}
        self.grid_kwargs = grid_kwargs or {}
        # Define sub-environments of the Stack
        self.stage_conditioning_grid = 0
        self.stage_set = 1
        subenvs = [
            Grid(n_dim=2, length=self.max_elements_per_subenv + 1),
            SetFlex(
                max_elements=self.max_elements_per_subenv * 2,
                envs_unique=(
                    ContinuousCube(n_dim=n_dim, **self.cube_kwargs),
                    Grid(n_dim=n_dim, **self.grid_kwargs),
                ),
            ),
        ]
        # Initialize base Stack environment
        super().__init__(subenvs=tuple(subenvs), **kwargs)

    @property
    def conditioning_grid(self) -> Grid:
        """
        Returns the sub-environment corresponding to the Grid that is the first
        sub-environment in the Stack, which is used to sample the conditions of the Set.

        Returns
        -------
        Grid
        """
        return self.subenvs[self.stage_conditioning_grid]

    @property
    def set(self) -> SetFlex:
        """
        Returns the sub-environment corresponding to the set of cubes and grids.

        Returns
        -------
        SetFlex
        """
        return self.subenvs[self.stage_set]

    @property
    def cube(self) -> ContinuousCube:
        """
        Returns the ContinuousCube environment that is used as unique environment to
        define the Cubes in the Set.

        The Cube is the unique environment in the first (0) dimension of subenvs_unique
        in the Set.

        Returns
        -------
        ContinuousCube
        """
        return self.set.envs_unique[IDX_CUBE]

    @property
    def grid(self) -> Grid:
        """
        Returns the Grid environment that is used as unique environment to define the
        Grids in the Set.

        The Grid is the unique environment in the second (1) dimension of
        subenvs_unique in the Set.

        Returns
        -------
        Grid
        """
        return self.set.envs_unique[IDX_GRID]

    def _check_has_constraints(self) -> bool:
        """
        Checks whether the Stack has constraints across sub-environments.

        It returns True, because the SetBox always has constraints from the
        conditioning grid to the Set.

        Returns
        -------
        bool
            True
        """
        return True

    def _apply_constraints(
        self,
        action: Tuple = None,
        state: Union[List, torch.Tensor] = None,
        dones: List[bool] = None,
        is_backward: bool = False,
    ):
        """
        Applies constraints across sub-environments, when applicable.

        This method sets up the Set of cubes and grids, once the first conditioning
        environment (grid) of the Stack is done.

        The first dimension (0) of the conditioning grid indicates the number of cubes
        in the Set and the second dimension indicates the number of grids. If both are
        zero, then the number of grids and cubes will be set at random.

        This method is used in step() and set_state().

        Parameters
        ----------
        action : tuple
            An action from the SetBox environment.
        state : list or tensor (optional)
            A state from the SetBox environment.
        is_backward : bool
            Boolean flag to indicate whether the action is in the backward direction.
        dones : list
            A list indicating the sub-environments that are done.
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

        Parameters
        ----------
        action : tuple
            An action from the SetBox environment.
        state : list or tensor (optional)
            A state from the SetBox environment.
        dones : list
            A list indicating the sub-environments that are done.
        """
        if self.conditioning_grid.done and (
            action is None or self._depad_action(action) == self.conditioning_grid.eos
        ):
            n_cubes = self.conditioning_grid.state[IDX_CUBE]
            n_grids = self.conditioning_grid.state[IDX_GRID]
            if n_cubes == 0 and n_grids == 0:
                subenvs = self.set._sample_random_subenvs()
            else:
                cubes = [IDX_CUBE] * n_cubes
                grids = [IDX_GRID] * n_grids
                subenvs = self.set.get_env_instances_by_unique_indices(cubes + grids)
            self.set.set_subenvs(subenvs=subenvs)
            # If a state is passed as argument, set the state and done of set
            # sub-environment
            if state is not None:
                self.set.set_state(
                    self._get_substate(state, self.stage_set),
                    done=bool(dones[self.stage_set]),
                )
            # Update global Stack state with state of Set
            self._set_substate(self.stage_set, self.set.state)

    def _apply_constraints_backward(self, action: Tuple = None):
        """
        Applies constraints across sub-environments, when applicable, in the backward
        direction.

        Parameters
        ----------
        action : tuple
            An action from the SetBox environment.
        """
        if action is None or self._depad_action(action) == self.conditioning_grid.eos:
            # Reset source of Set and update global Stack state
            self.set.state = copy(self.set.source)
            self._set_substate(self.stage_set, self.set.state)

    def states2proxy(
        self, states: List[List]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy.

        The proxy representation is the average of the proxy representation across all
        the cubes and grids in the set.

        Parameters
        ----------
        states : list
            A batch of states in environment format.

        Returns
        -------
        A tensor containing all the states in the batch in the proxy representation.
        """
        # Keep only the part of the state corresponding to the Set
        states_proxy_set = self.set.states2proxy(
            [self._get_substate(state, self.stage_set) for state in states]
        )
        states_proxy = []
        for state in states_proxy_set:
            states_box = tfloat(
                self.set._get_substates(state),
                float_type=self.float,
                device=self.device,
            )
            states_proxy.append(torch.mean(states_box, dim=0))
        return tfloat(states_proxy, float_type=self.float, device=self.device)
