"""
A conditional Cube environment.

The environment is a Cube with variable dimensionality determined by a conditioning
integer variable.
"""

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torchtyping import TensorType
from tqdm import tqdm

from gflownet.envs.conditional import BaseConditional
from gflownet.envs.conditions.base import BaseCondition
from gflownet.envs.cube import ContinuousCube
from gflownet.utils.common import copy, tfloat


class CubeCondDim(BaseConditional):
    """
    A cube environment whose dimensionality is conditioned by an integer variable.

    The environment must be configured by setting the train dimensions as well as the
    maximum test dimension.
    """

    def __init__(
        self,
        dimensions_tr: Union[int, List] = 2,
        max_dimension: int = 3,
        cube_kwargs: Optional[Dict] = {},
        **kwargs,
    ):
        """
        Initializes the BaseConditional environment.

        Parameters
        ----------
        dimensions_tr : int or list
            The number of dimensions that will be considered during training as
            conditioning variables. If the parameter is an integer, all dimensions from
            1 to ``dimensions_tr`` are considered.
        max_dimension : int
            The maximum dimensionality of the cube. The maximum dimensionality may be
            higher than the maximum dimensionality used during training.
        cube_kwargs : dict
            A dicitonary of parameters to initialize the ContinuousCube environment.
        """
        if isinstance(dimensions_tr, int):
            dimensions_tr = list(range(1, dimensions_tr + 1))
        assert max_dimension >= max(dimensions_tr)
        dimension = BaseCondition(conditions_dataset=[[d] for d in range(max_dimension)])
        cube = ContinuousCube(n_dim=max_dimension, **cube_kwargs)
        # Initialize base conditioanl environment
        super().__init__(condition_env=dimension, base_env=cube, **kwargs)

    @property
    def cube(self) -> ContinuousCube:
        """
        Returns the sub-environment corresponding to the cube (base) environment.

        Returns
        -------
        ContinuousCube
        """
        return self.base_env

    def _check_has_constraints(self) -> bool:
        """
        Checks whether the Stack has constraints across sub-environments.

        It returns True, because the condition sets the ignored dimensions of the cube.

        Returns
        -------
        bool
            True
        """
        return True

    def _apply_constraints_forward(
        self,
        action: Tuple = None,
        state: Union[List, torch.Tensor] = None,
        dones: List[bool] = None,
    ):
        """
        Applies constraints across sub-environments, when applicable, in the forward
        direction.

        When the condition environment is done, its variable sets the number of
        dimensions of the cube that are not ignored.

        Parameters
        ----------
        action : tuple
            An action from the SetBox environment.
        state : list or tensor (optional)
            A state from the SetBox environment.
        dones : list
            A list indicating the sub-environments that are done.
        """
        if self.do_condition_constraints(action, is_backward=False):
            n_dims = self.condition_env.state[0]
            self.cube.ignored_dims = [False] * n_dims + [True] * (
                self.cube.n_dim - n_dims
            )

    def _apply_constraints_backward(self, action: Tuple = None):
        """
        Applies constraints across sub-environments, when applicable, in the backward
        direction.

        Parameters
        ----------
        action : tuple
            An action from the SetBox environment.
        """
        if self.do_condition_constraints(action, is_backward=True):
            self.cube.ignored_dims = [False] * self.cube.n_dim
