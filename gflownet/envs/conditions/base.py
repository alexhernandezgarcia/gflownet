"""
Base class for conditions, to train conditional GFlowNets.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import numpy.typing as npt
from torchtyping import TensorType

from gflownet.envs.dummy import Dummy


class BaseCondition(Dummy):
    """
    Base condition environment.

    The base condition is a Dummy environment, which allows for arbitrary states. The
    state of this environment is expected to be the conditioning variable to train
    conditional GFlowNets. Conditional environments are created via
    :py:class:`gflownet.envs.conditional.BaseConditional`, which is a stack of a
    condition environment plus a regular environment.

    states2policy, states2proxy, states2readable and readable2state are not implemented
    in this meta-class. Depending on the conditioning variable, the default methods may
    work or not.

    Attributes
    ----------
    conditions_dataset : str
        A data set of conditions. If a path to a data set of conditions is not
        specified, then this attribute is set to None. This will be the case for
        continuous conditions, for example.
    """

    def __init__(
        self,
        condition: Union[List, Dict, TensorType, npt.NDArray] = None,
        conditions_dataset: Union[List, Dict, TensorType, npt.NDArray] = None,
        conditions_path: str = None,
        **kwargs,
    ):
        """
        Initializes the BaseCondition environment.

        Parameters
        ----------
        condition : list, dict, tensor, array (optional)
            A variable to be used as condition. It must be a valid environment state.
        conditions_dataset : list, dict, tensor, array (optional)
            A set of variables to be as conditions. Each variable must be a valid
            environment state.
        conditions_path : str (optional)
            A path to a data set of conditions. Ignored if ``conditions_dataset`` is
            not None.
        """
        if conditions_dataset is not None:
            self.conditions_dataset = conditions_dataset
        elif conditions_path is not None:
            self.conditions_dataset = load_conditions(conditions_path)
            self.n_conditions = len(self.conditions_dataset)
        else:
            self.conditions_dataset = None
        # Set the condition as the state. If no condition has been passed as parameter,
        # sample a random condition
        if condition is not None:
            state = condition
        else:
            state = self._sample_condition()
        super().__init__(state, **kwargs)

    @property
    def n_conditions(self) -> int:
        """
        Returns the number of conditions in the data set.

        If there is no data set (``self.conditions_dataset is None``, None is returned.

        Returns
        -------
        The number of conditions in the data set.
        """
        if self.conditions_dataset is None:
            return None
        else:
            if not hasattr(self, "_n_conditions"):
                self._n_conditions = len(self.conditions_dataset)
            return self._n_conditions

    def _sample_condition(self) -> Union[List, Dict, TensorType, npt.NDArray]:
        """
        Randomly samples a condition.

        By default, this method samples uniformly one of the conditions in
        ``self.conditions_dataset``.

        This method should be overwritten if the procedure to sample a condition
        is different than sampling from a data set. For example, if the conditioning
        variables are continuous.

        Returns
        -------
        list, dict, tensor, array
            A conditioning variable ready to be set as a state of the environment.

        Raises
        ------
        RuntimeError
            If ``self.n_conditions`` is None.
        """
        if self.n_conditions is None:
            raise RuntimeError(
                "Cannot sample a random conditions if self.n_conditions is None"
            )
        else:
            return self.conditions_dataset[np.random.choice(self.n_conditions)]

    def sample_condition(self):
        """
        Samples a conditions and sets it as the state.
        """
        self.set_state(self._sample_condition())

    def is_valid_condition(self, condition: Union[List, Dict, TensorType, npt.NDArray]):
        """
        Checks if the condition passed as parameter is a valid condition.

        By default, this method checks whether the condition is part of the data set.

        This method should be overwritten if the procedure to sample a condition
        is different than sampling from a data set. For example, if the conditioning
        variables are continuous.

        Parameters
        ----------
        condition : list, dict, tensor, array
            A valid conditioning variable to be set as a state of the environment.

        Raises
        ------
        ValueError
            If the condition is not in the data set and the conditions are not
            continuous.
        """
        return condition in self.conditions_dataset

    def set_condition(self, condition: Union[List, Dict, TensorType, npt.NDArray]):
        """
        Sets as state the condition passed as parameter.

        Parameters
        ----------
        condition : list, dict, tensor, array
            A valid conditioning variable to be set as a state of the environment.

        Raises
        ------
        ValueError
            If the condition is not in the data set and the conditions are not
            continuous.
        """
        if not self.is_valid_condition(condition):
            raise ValueError(
                f"Attempted to set invalid condition: {condition}"
            )
        self.set_state(condition)
