"""
This is a meta-environment designed to serve as the base environment for
conditional environments.

The Conditional environment is implemented as a Stack of a Dummy environment which
serves as the condition and another environment which serves as actual (conditioned)
GFlowNet environment.
"""

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torchtyping import TensorType
from tqdm import tqdm

from gflownet.envs.stack import Stack
from gflownet.utils.common import copy, tfloat
from gflownet.envs.conditions.base import BaseCondition
from gflownet.envs.base import GFlowNetEnv


class BaseConditional(Stack):
    """
    A Stack of a Constant (condition) and any other environment.

    Attributes
    ----------
    stage_condition : int
        The stage corresponding to the condition environment: 0
    stage_base : int
        The stage corresponding to the base environment: 1
    """

    def __init__(
        self,
        condition_env: BaseCondition = None,
        base_env: GFlowNetEnv = None,
        **kwargs,
    ):
        """
        Initializes the BaseConditional environment.

        Parameters
        ----------
        condition : BaseCondition
            The condition environment.
        base : GFlowNetEnv
            The base environment to be conditioned.
        """
        self.stage_condition = 0
        self.stage_base = 1
        # If a condition is passed in kwargs, set it as the condition of the condition
        # environment
        if "condition" in kwargs:
            condition_env.set_condition(kwargs["condition"])
        # Initialize base Stack environment
        super().__init__(subenvs=tuple([condition_env, base_env]), **kwargs)

    def _check_has_constraints(self) -> bool:
        """
        Checks whether the Stack has constraints across sub-environments.

        By default, there is no constraints. However, this method may be overriden if,
        for example, the condition sets any constraints on the base environment.

        Returns
        -------
        bool
            False
        """
        return False

    @property
    def condition_env(self) -> BaseCondition:
        """
        Returns the sub-environment corresponding to the condition.

        Returns
        -------
        BaseCondition
        """
        return self.subenvs[self.stage_condition]

    @property
    def base_env(self) -> GFlowNetEnv:
        """
        Returns the sub-environment corresponding to the base environment.

        Returns
        -------
        GFlowNetEnv
        """
        return self.subenvs[self.stage_base]

    def do_condition_constraints(self, action: Tuple, is_backward: bool = False):
        """
        Returns True if the constraints related to the condition should be applied.

        If the condition imposes any constraints on the base environment, this method
        can be called from:
            - :py:meth:`~gflownet.envs.stack.Stack._apply_constraints_forward`
            - :py:meth:`~gflownet.envs.stack.Stack._apply_constraints_backward`

        This method is a simple alias of
        :py:meth:`gflownet.envs.stack.Stack._do_constraints_for_stage`, where the stage
        corresponding to the condition is passed as the ``stage`` parameter.

        Parameters
        ----------
        action : tuple
            The action of the transition.
        is_backward : bool
            Boolean flag to indicate whether the potential constraint is in the
            backward direction (True) or in the forward direction (False).
        """
        return self._do_constraints_for_stage(self.stage_condition, action, is_backward)
