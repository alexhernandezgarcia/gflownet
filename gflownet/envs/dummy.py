"""
Base class for dummy environments: identical to the Constant environment, except that
the state might be changed without breaking the expected functioning of the
environment. To account for this, the source state is updated if the state changes, and
its value is not considered to check whether a state is the source. This means that the
state might be different to the state passed at initialization, which may be useful in
certain scenarios, such as conditional environments.
"""

from typing import Any, List, Optional, Tuple

import numpy.typing as npt
from torchtyping import TensorType

from gflownet.envs.constant import Constant
from gflownet.utils.common import copy


class Dummy(Constant):
    """
    Dummy environment.

    The state might be None, in which the state will be set to [0].

    states2policy, states2proxy, states2readable and readable2state are not implemented
    in this meta-class. Depending on the state, the default methods may work or not.

    Attributes
    ----------
    state : list, dict, tensor, array
        The state which will be set as constant, that is as source and final state.
    """

    def __init__(
        self,
        state: Any = None,
        **kwargs,
    ):
        if state is None:
            state = [0]
        super().__init__(state, **kwargs)

    def is_source(self, state: Any = None) -> bool:
        """
        Returns True if the environment's state or the state passed as parameter (if
        not None) is the source state of the environment.

        This method is overriden to completely ignore the state. If the environment is
        done, then the environment is not at the source, and vice versa.

        Parameters
        ----------
        state : list or tensor or None
            Ignored.

        Returns
        -------
        bool
            Whether the environment is done (False) or not (True).
        """
        return not self.done

    def set_state(self, state: List, done: Optional[bool] = False):
        """
        Sets the state and done of an environment.

        This method is overriden to update the source state whenever the state is set.
        """
        self.source = copy(state)
        return super().set_state(state, done)
