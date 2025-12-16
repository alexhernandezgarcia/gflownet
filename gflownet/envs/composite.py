"""
Base class for composite environments.

Composite environments are environments which consist of multiple environments.
"""

import uuid
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tfloat, tlong


class CompositeBase(GFlowNetEnv):
    """
    Base class for composite environments.

    The states of composite environments are dictionaries. Keys with integers, starting
    from 0, are reserved to contain the states at the position indicated by the key.
    Additionally, the dictionary includes keys with meta-data about the state of the
    composite environment. The following keys and values are expected to be included in
    all composite environments:
        - ``_active``: The index of the currently active sub-environment, or -1 if none
          is active.
        - ``_dones``: A list of flags indicating whether the sub-environments are done
          (1) or not done (0). For example, ``[0, 1, 0]`` indicates that the
          sub-environments at indices 0 and 2 are not done, and the sub-environment at
          index 1 is done.
        - ``_envs_unique``: A list of indices identifying the unique environment
          corresponding to each sub-environment. For example, ``[1, 1, 0]`` indicates
          that the sub-environments at indices 0 and 1 are of the same type, in
          particular the type of environment at index 1 in ``self.envs_unique``; the
          sub-environment at index 2 is of the type at index 0 in ``self.envs_unique``.
          If the environment type at a position is unknown, it contains -1.
    Composite environments may include additional meta-data as needed.

    Attributes
    ----------
    max_elements : int
        The maximum number of elements that can be included in the composite
        environment. Note that this number does not refer to the number of unique
        environments, but the number of elements (instances of a sub-environment) that
        can form a composite environment.
    subenvs : iterable
        The collection of sub-environments included in the composite environment.
    envs_unique : iterable
        The collection of unique environments that can make part of the composite
        environment. Uniqueness is defined in terms of the environment type and action
        space.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initializes the CompositeBase environment.
        """
        # Base class init
        super().__init__(**kwargs)

    def _get_substate(self, state: Dict, idx_subenv: Optional[int] = None):
        """
        Returns the part of the state corresponding to the sub-environment indicated at
        ``idx_subenv``.

        Parameters
        ----------
        state : dict
            A state of the composite environment.
        idx_subenv : int
            Index of the sub-environment of which the corresponding part of the
            state is to be extracted. If None, the state of the active subenv is used.

        Returns
        -------
        The state of a sub-environment.
        """
        if idx_subenv is None:
            idx_subenv = self._get_active_subenv(state)
        if idx_subenv not in range(self.max_elements):
            raise ValueError(
                f"Index {idx_subenv} is not a valid sub-environment index."
            )
        return state[idx_subenv]

    def _get_substates(self, state: Dict) -> List:
        """
        Returns a list with all the states of the sub-environments.

        Parameters
        ----------
        state : list
            A state of the parent Set environment.

        Returns
        -------
        list
            All sub-states in the state.
        """
        substates = []
        for idx in range(self.max_elements):
            if idx in state:
                substates.append(state[idx])
            else:
                break
        return substates

    def _set_substate(
        self,
        idx_subenv: int,
        state_subenv: Union[List, TensorType, dict],
        state: Optional[Dict] = None,
    ) -> Dict:
        """
        Updates the global composite state by setting as substate of subenv
        ``idx_subenv`` the current state of the sub-environment.

        This method modifies ``self.state`` if ``state`` is None.

        Parameters
        ----------
        idx_subenv : int
            Index of the sub-environment of which to set the state.
        state_subenv : list or tensor or dict
            The state of a sub-environment.
        state : dict
            A state of the global composite environment.

        Returns
        -------
        The updated composite state.
        """
        assert idx_subenv in range(self.max_elements)
        if state is None:
            state = self.state
        state[idx_subenv] = state_subenv
        return state

    def _get_active_subenv(self, state: Optional[Dict] = None) -> int:
        """
        Returns the index of the currently active sub-environment.

        If no state is passed, ``self.state`` is used.

        The active sub-environment is indicated in ``state["_active"]``.

        Parameters
        ----------
        state : dict
            A state of the composite environment.
        """
        if state is None:
            state = self.state
        return state["_active"]

    def _set_active_subenv(self, idx_subenv: int, state: Optional[Dict] = None) -> Dict:
        """
        Sets the index of the active sub-environment.

        If no state is passed, ``self.state`` is used.

        The active sub-environment is set in ``state["_active"]``.

        Parameters
        ----------
        idx_subenv : int
            Index of the sub-environment to set as active, or -1.
        state : dict
            A state of the composite environment.

        Returns
        -------
        The updated composite state.
        """
        assert idx_subenv in range(self.max_elements) or idx_subenv == -1
        if state is None:
            state = self.state
        state["_active"] = idx_subenv
        return state

    def _get_dones(self, state: Optional[Dict] = None) -> List[int]:
        """
        Returns the part of the state containing the list of done flags.

        The list of done flags indicate which sub-environments are done or not.

        The list of done flags is indicated in ``state["_dones"]``.

        Parameters
        ----------
        state : dict
            A state of the composite environment.

        Returns
        -------
        The list of dones as integer flags (0 or 1).
        """
        if state is None:
            state = self.state
        return state["_dones"]

    def _set_subdone(
        self, idx_subenv: int, done: bool, state: Optional[Dict] = None
    ) -> Dict:
        """
        Updates the done flag corresponding to the sub-environment at ``idx_subenv``.

        Parameters
        ----------
        idx_subenv : int
            Index of the sub-environment of which to set the done flag.
        done : bool
            The value of done to be set in the state.
        state : dict
            A state of the composite environment.

        Returns
        -------
        The updated composite state.
        """
        assert idx_subenv in range(self.max_elements)
        if state is None:
            state = self.state
        state["_dones"][idx_subenv] = int(done)
        return state

    def _set_unique_index(
        self, idx_subenv: int, idx_unique: int, state: Optional[Dict] = None
    ) -> Dict:
        """
        Updates the index of the sub-environment indicated by idx_subenv in the list of
        unique indices.

        Parameters
        ----------
        idx_subenv : int
            Index of the sub-environment of which to set the unique index.
        idx_unique : int
            The unique index to be set.
        state : dict
            A state of the composite environment.

        Returns
        -------
        The updated composite state.
        """
        assert idx_subenv in range(self.max_elements)
        assert idx_unique in range(self.n_unique_envs)
        if state is None:
            state = self.state
        state["_envs_unique"][idx_subenv] = idx_unique
        return state

    def _get_unique_environments(
        self, subenvs: Iterable[GFlowNetEnv]
    ) -> Tuple[List[GFlowNetEnv], Tuple, List]:
        """
        Determines the set of unique environments in the iterable subenvs passed as an
        argument.

        Uniqueness is determined by both the type of environment and the action space:
        two environments are only considered equal if both the type and the action
        space are the same.

        Parameters
        ----------
        subenvs : iterable
            Iterable of sub-environments.

        Returns
        -------
        envs_unique : list
            The list of unique environments.
        envs_unique_keys : tuple
            A tuple containing the keys that identify each unique environment, namely
            tuples with the type of environment and the action space.
        unique_indices : list
            A list containing the index of the unique environment corresponding to each
            sub-environment in the input iterable.
        """
        envs_unique = []
        envs_unique_keys = []
        unique_indices = []
        for idx, env in enumerate(subenvs):
            env_key = (type(env), tuple(env.action_space))
            if env_key not in envs_unique_keys:
                envs_unique_keys.append(env_key)
                envs_unique.append(env)
            unique_indices.append(envs_unique_keys.index(env_key))
        return envs_unique, tuple(envs_unique_keys), unique_indices

    @property
    def n_unique_envs(self) -> int:
        """
        Returns the number of unique environments.
        """
        if hasattr(self, "_n_unique_envs"):
            return self._n_unique_envs
        if not hasattr(self, "envs_unique"):
            envs_unique, _, _ = self._get_unique_environments(self.subenvs)
            self._n_unique_envs = len(envs_unique)
        return self._n_unique_envs

    def _get_env_unique(self, idx_unique: int) -> GFlowNetEnv:
        """
        Returns the unique environment with index idx_unique``.

        This method requires the definition of the attribute ``envs_unique``
        containing the set of unique enviroments.

        Parameters
        ----------
        idx_unique : int
            The index of the unique environment to be retrieved.

        Returns
        -------
        GFlowNetEnv
            The unique environment with index idx_unique.
        """
        return self.envs_unique[idx_unique]

    def _get_unique_idx_of_subenv(
        self, idx_subenv: int, state: Optional[List] = None
    ) -> int:
        """
        Returns the index of the unique environment corresponding to the subenv at
        index ``idx_subenv``.

        The index refers to the state passed as an input. If the state is None,
        ``self.state`` is used.

        This method requires the definition of the attributes ``envs_unique``,
        containing the set of unique enviroments, and ``max_elements``, containing the
        maximum number of sub-environments allowed in the composite environment.

        Parameters
        ----------
        idx_subenv : int
            Index of a sub-environment (from 0 to ``self.max_elements``). Note that
            this is the index of a subenv, not of the unique environments.
        state : list
            A state of the parent composite environment.
        """
        assert idx_subenv in range(self.max_elements)
        if state is None:
            state = self.state
        return self._get_unique_indices(state)[idx_subenv]

    def _get_unique_env_of_subenv(
        self, idx_subenv: int, state: Optional[List] = None
    ) -> GFlowNetEnv:
        """
        Returns the unique environment corresponding to the sub-environment at index
        ``idx_subenv``.

        Parameters
        ----------
        idx_subenv : int
            Index of a sub-environment (from 0 to ``self.max_elements``). Note that
            this is the index of a subenv, not of the unique environments.
        state : list
            A state of the parent composite environment.
        """
        return self._get_env_unique(self._get_unique_idx_of_subenv(idx_subenv, state))

    def _get_unique_indices(
        self, state: Optional[List] = None, exclude_nonpresent: bool = True
    ) -> int:
        """
        Returns the part of the state containing the unique indices.

        The unique indices identify the type of environment of each element in state of
        the composite environment.

        Parameters
        ----------
        state : list
            A state of the parent composite environment.
        exclude_nonpresent : bool
            If True, return only the indices of sub-environments that are present in
            the state, that is exclude indices with -1.
        """
        if state is None:
            state = self.state
        unique_indices = state["_envs_unique"]
        if exclude_nonpresent:
            return [idx for idx in unique_indices if idx != -1]
        return unique_indices

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs a list with all possible actions, including EOS.

        By default, the action space of a Composite environment consists of:
            - The concatenation of the actions of all unique environments.
            - The EOS action.

        Certain composite environments may make use of additional actions, for example
        to toggle specific sub-environments.

        In order to make all actions the same length (required to construct batches of
        actions as a tensor), the actions are zero-padded from the back.

        In order to make all actions unique, the unique environment index is added as
        the first element of the action.

        Note that the actions of unique environments are only added once to the action
        space, regardless of how many elements of the unique environment
        (sub-environments) there are in the composite environment. In other words,
        identical environments that are part of the composite environment share the
        actions and a given action will have an effect on the sub-environment that is
        next or active.

        See:
        - :py:meth:`~gflownet.envs.composite.CompositeBase._pad_action`
        - :py:meth:`~gflownet.envs.composite.CompositeBase._depad_action`
        """
        # EOS action
        action_space = [self.eos]
        # Action space of each unique environment
        for idx in range(self.n_unique_envs):
            action_space.extend(
                [
                    self._pad_action(action, idx)
                    for action in self._get_env_unique(idx).action_space
                ]
            )
        return action_space

    def _pad_action(self, action: Tuple, idx_unique: int) -> Tuple:
        """
        Pads an action by adding the unique index (or -1) as the first element and zeros
        as padding.

        See:
        - :py:meth:`~gflownet.envs.composite.CompositeBase.get_action_space`

        Parameters
        ----------
        action : tuple
            The action to be padded.
        idx_unique : int
            The index of the unique environment or -1 for meta-actions (EOS and other
            actions of the composite environment)

        Returns
        -------
        tuple
            The padded and pre-fixed action.
        """
        return (idx_unique,) + action + (0,) * (self.action_dim - len(action) - 1)

    def _depad_action(self, action: Tuple, idx_unique: int = None) -> Tuple:
        """
        Reverses the padding operation, such that the resulting action can be passed to
        the underlying environment.

        See:
        - :py:meth:`~gflownet.envs.composite.CompositeBase._pad_action`

        Parameters
        ----------
        action : tuple
            The action to be depadded.
        idx_unique : int
            The index of the unique environment or -1 for meta-actions (EOS and other
            actions of the composite environment)

        Returns
        -------
        tuple
            The depadded action, as it appears in the action space of the
            sub-environment it belongs to. If idx_unique is -1 (meta-action), then the
            returned action is a single-element tuple with the sub-environment index.
        """
        if idx_unique is None:
            idx_unique = action[0]
        else:
            assert idx_unique == action[0]
        if idx_unique != -1:
            return action[1 : 1 + len(self._get_env_unique(idx_unique).eos)]
        return (action[1],)
