"""
Base class for constant environments: simple, helper environment where the source state
is the final state and the only valid action is EOS.
"""

from typing import Any, List, Optional, Tuple

import numpy.typing as npt
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv


class Constant(GFlowNetEnv):
    """
    Constant environment.

    states2policy, states2proxy, states2readable and readable2state are not implemented
    in this meta-class. Depending on the state, the default methods may work or not.

    Attributes
    ----------
    state : list, dict, tensor, array
        The state which will be set as constant, that is as source and final state.
    """

    def __init__(
        self,
        state: Any,
        **kwargs,
    ):
        self.source = state
        self.eos = (0,)
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs list with all possible actions, including eos.

        The action space consists of the EOS action only.
        """
        return [self.eos]

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[Any] = None,
        done: Optional[bool] = None,
    ) -> List[bool]:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.

        The only possible action (EOS) is only valid if the environmet is not done.

        Parameters
        ----------
        state : list, dict, tensor, array
            Ignored.

        done : bool
            Whether the trajectory is done. If None, self.done is used.

        Returns
        -------
        A list of boolean values.
        """
        done = self._get_done(done)
        if done:
            return [True]
        else:
            return [False]

    def get_parents(
        self,
        state: Optional[Any] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        If the environment is done, the only parent is the state, with EOS action.
        Otherwise, the source state has no parents, like all environments.

        Parameters
        ----------
        state : list, dict, tensor, array
            Input state. If None, self.state is used.

        done : bool
            Whether the trajectory is done. If None, self.done is used.

        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format. This environment has a single parent per
            state.

        actions : list
            List of actions that lead to state for each parent in parents. This
            environment has a single parent per state.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [state], [self.eos]
        else:
            return [], []

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> [List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Parameters
        ----------
        action : tuple
            Action to be executed.

        skip_mask_check : bool
            Ignored.

        Returns
        -------
        self.state : list, dict, tensor, array
            The state after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        if self.done:
            return self.state, action, False
        else:
            assert action == self.eos
            self.done = True
            self.n_actions += 1
            return self.state, action, True

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment, including the EOS
        action.
        """
        return 1
