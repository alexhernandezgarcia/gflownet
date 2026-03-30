"""
Base class for composite environments.

Composite environments are environments which consist of multiple environments.
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv


class CompositeBase(GFlowNetEnv):
    """
    Base class for composite environments.

    The states of composite environments are dictionaries. Keys with integers, starting
    from 0, are reserved to contain the states at the position indicated by the key.
    Additionally, the dictionary may includes keys with meta-data about the state of
    the composite environment. The following keys and values are supported by default
    by the base composite environment:
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

    Not all supported keys need to be included in the states of sub-classes and new
    keys may be included as needed.

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
        # Constraints
        self._has_constraints = self._check_has_constraints()
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
        if idx_subenv >= self.max_elements:
            raise ValueError(
                f"Index {idx_subenv} is not a valid sub-environment index."
            )
        return state[idx_subenv]

    def _get_substates(self, state: Dict) -> List:
        """
        Returns a list with all the states of the sub-environments.

        Parameters
        ----------
        state : dict
            A state of the global composite environment.

        Returns
        -------
        list
            All sub-states in the state.
        """
        substates = []
        for idx in range(self.max_elements):
            if idx in state:
                substates.append(self._get_substate(state, idx))
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
        assert idx_subenv < self.max_elements
        state = self._get_state(state)
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
        state = self._get_state(state)
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
        assert idx_subenv < self.max_elements or idx_subenv == -1
        state = self._get_state(state)
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
        state = self._get_state(state)
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
        assert idx_subenv < self.max_elements
        state = self._get_state(state)
        state["_dones"][idx_subenv] = int(done)
        return state

    def _get_subdone(self, idx_subenv: int, state: Optional[Dict] = None) -> bool:
        """
        Returns whether if the sub-environment at ``idx_subenv`` is done.

        Parameters
        ----------
        idx_subenv : int
            Index of the sub-environment to query.
        state : dict
            A state of the composite environment.

        Returns
        -------
        True if the sub-environment at ``idx_subenv`` is done; False otherwise.
        """
        assert idx_subenv < self.max_elements
        return bool(self._get_dones(state)[idx_subenv])

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
        assert idx_subenv < self.max_elements
        assert idx_unique < self.n_unique_envs
        state = self._get_state(state)
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

        Returns
        -------
        int
            The number of unique environments.

        Raises
        ------
        RuntimeError
            If ``self.subenvs`` is not an attribute of the environment.
        """
        if hasattr(self, "_n_unique_envs"):
            return self._n_unique_envs
        if not hasattr(self, "envs_unique"):
            if not hasattr(self, "subenvs"):
                raise RuntimeError("The environment does not contain self.subenvs")
            envs_unique, _, _ = self._get_unique_environments(self.subenvs)
            self._n_unique_envs = len(envs_unique)
        else:
            self._n_unique_envs = len(self.envs_unique)
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
        self, idx_subenv: int, state: Optional[Dict] = None
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
        state : dict
            A state of the global composite environment.
        """
        assert idx_subenv < self.max_elements
        return self._get_unique_indices(state)[idx_subenv]

    def _get_unique_env_of_subenv(
        self, idx_subenv: int, state: Optional[Dict] = None
    ) -> GFlowNetEnv:
        """
        Returns the unique environment corresponding to the sub-environment at index
        ``idx_subenv``.

        Parameters
        ----------
        idx_subenv : int
            Index of a sub-environment (from 0 to ``self.max_elements``). Note that
            this is the index of a subenv, not of the unique environments.
        state : dict
            A state of the global composite environment.
        """
        return self._get_env_unique(self._get_unique_idx_of_subenv(idx_subenv, state))

    def _get_unique_indices(self, state: Optional[Dict] = None) -> int:
        """
        Returns the part of the state containing the unique indices.

        The unique indices identify the type of environment of each element in state of
        the composite environment.

        Parameters
        ----------
        state : dict
            A state of the global composite environment.
        """
        state = self._get_state(state)
        return state["_envs_unique"]

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs a list with all possible actions, including EOS.

        By default, the action space of a Composite environment consists of the
        concatenation of the actions of all unique environments.

        Certain composite environments may make use of additional actions, for example
        to toggle specific sub-environments.

        Sub-classes with additional actions should override this method.

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
        - :py:meth:`~gflownet.envs.composite.base.CompositeBase._pad_action`
        - :py:meth:`~gflownet.envs.composite.base.CompositeBase._depad_action`
        """
        action_space = []
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
        - :py:meth:`~gflownet.envs.composite.base.CompositeBase.get_action_space`

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
        - :py:meth:`~gflownet.envs.composite.base.CompositeBase._pad_action`

        If idx_unique does not match the prefix of the action, the action is returned
        as is.

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
            # If idx_unique does not match the action prefix, raise an error
            if idx_unique != action[0]:
                raise RuntimeError(
                    f"There is a mismatch between the input idx_unique ({idx_unique}) "
                    f"for de-padding and the unique index in the action ({action[0]})"
                )
        if idx_unique != -1:
            return action[1 : 1 + len(self._get_env_unique(idx_unique).eos)]
        return (action[1],)

    def _depad_action_batch(
        self,
        actions: TensorType["batch_size", "action_dim"],
        idx_unique: int,
    ) -> TensorType["batch_size", "action_dim_subenv"]:
        """
        Reverses the padding operation for a batch of actions.

        It is assumed that all actions correspond to the same unique environment or
        that all of them are meta-actions.

        See:
        - :py:meth:`~gflownet.envs.composite.base.CompositeBase._depad_action`

        Parameters
        ----------
        actions : tensor
            The batch of actions to be depadded.
        idx_unique : int
            The index of the unique environment or -1 for meta-actions (EOS and other
            actions of the composite environment)

        Returns
        -------
        tensor
            The depadded batch of actions. If idx_unique is -1 (meta-action), then the
            returned batch is a single-column tensor with the sub-environment index.
        """
        if idx_unique != -1:
            return actions[:, 1 : 1 + len(self._get_env_unique(idx_unique).eos)]
        return actions[:, 1]

    def set_state(self, state: Dict, done: Optional[bool] = False):
        """
        Sets a state and done.

        The correct state and done of each sub-environment are set too.

        Parameters
        ----------
        state : dict
            A state of the global composite environment.
        done : bool
            Whether the trajectory of the environment is done or not.
        """
        # If done is True, then all sub-environments must be done
        if done:
            dones = [True] * self.max_elements
        else:
            dones = self._get_dones(state)

        # Set state and done
        super().set_state(state, done)

        # Set state and done of each sub-environment
        for idx, (subenv, done_subenv) in enumerate(zip(self.subenvs, dones)):
            subenv.set_state(self._get_substate(self.state, idx), bool(done_subenv))

        # Apply constraints across sub-environments, in case they apply.
        self._apply_constraints(state=self.state, is_backward=None)

        return self

    def reset(self, env_id: Union[int, str] = None):
        """
        Resets the environment by resetting the sub-environments.
        """
        if self.subenvs is not None:
            for subenv in self.subenvs:
                subenv.reset()
        super().reset(env_id=env_id)

        # Apply constraints across sub-environments, in case they apply. is_backward is
        # set to True to bypass forward constraints.
        self._apply_constraints(state=self.state, is_backward=True)

        return self

    def get_policy_output(self, params: list[dict]) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model.

        By default, the policy output is the concatenation of the policy outputs of the
        unique environments.

        Sub-classes should override this method if the structure of the policy outputs
        changes, for example, if meta-actions are added.

        Parameters
        ----------
        params : list
            A list of distribution parameters. This list has as many elements as
            there are unique environments, since all sub-environments of the same
            environment type are expected to be identical.
        """
        return torch.cat(
            [
                self._get_env_unique(idx).get_policy_output(params_env_unique)
                for idx, params_env_unique in enumerate(params)
            ]
        )

    def _get_policy_outputs_of_env_unique(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        idx_unique: int,
    ):
        """
        Returns the columns of the policy outputs that correspond to the unique
        environment indicated by idx_unique.

        Since the policy outputs corresponding to each unique environment are
        concatenated across the columns of the input tensor, the outputs of a
        particular environment can be retrieved by iterating over the unique
        environments and calculating their output dimensions. In order to avoid this
        iteration at every request, the first call creates a dictionary of offsets as
        an attribute of the environment.

        Parameters
        ----------
        policy_outputs : tensor
            A tensor containing a batch of policy outputs. It is assumed that all the
            rows in the this tensor correspond to the same unique environment.
        idx_unique : int
            Index of the unique environment of which the corresponding columns of the
            policy outputs are to be extracted.
        """
        if hasattr(self, "_policy_outputs_offset_of_unique_env"):
            if idx_unique in self._policy_outputs_offset_of_unique_env:
                init_col = self._policy_outputs_offset_of_unique_env[idx_unique]
                end_col = init_col + self._get_env_unique(idx_unique).policy_output_dim
                return policy_outputs[:, init_col:end_col]
        else:
            self._policy_outputs_offset_of_unique_env = {}

        init_col = 0
        for idx in range(self.n_unique_envs):
            end_col = init_col + self._get_env_unique(idx).policy_output_dim
            if idx == idx_unique:
                self._policy_outputs_offset_of_unique_env[idx] = init_col
                return policy_outputs[:, init_col:end_col]
            init_col = end_col

    @property
    def has_constraints(self):
        """
        Whether the composite environment has constraints across sub-environments.

        Returns
        -------
        True if the composite environment has constraints across sub-environments.
        """
        return self._has_constraints

    def _check_has_constraints(self) -> bool:
        """
        Checks whether the composite environment has constraints across
        sub-environments.

        By default, composite environments do not have constraints (False).

        This method should be overriden in environments that incorporate constraints
        across sub-environmnents via ``_apply_constraints()``.

        Returns
        -------
        bool
            True if the composite environment has constraints, False otherwise
        """
        return False

    def _apply_constraints(
        self,
        action: Tuple = None,
        state: Optional[Dict] = None,
        is_backward: bool = None,
    ) -> bool:
        """
        Applies constraints across sub-environments.

        This method is called from the methods that can modify the state, namely:
            - :py:meth:`~gflownet.envs.composite.base.CompositeBase.step()`
            - :py:meth:`~gflownet.envs.composite.base.CompositeBase.step_backwards()`
            - :py:meth:`~gflownet.envs.composite.base.CompositeBase.set_state()`
            - :py:meth:`~gflownet.envs.composite.base.CompositeBase.reset()`

        Furthermore, it is also called from methods that receive an input state
        different to ``self.state``, in order to make sure that the sub-environments
        have the properties and constraints corresponding to the input state, rather
        than those of ``self.state``. For example:
            - ``get_mask_invalid_actions_forward()``
            - ``get_mask_invalid_actions_backward()``
            - ``get_valid_actions()``
            - ``get_parents()``

        In general, the application of constraints can be initiated by and depend on an
        action (for example, from ``step()` or ``step_backwards()``) or by a state
        (most other methods). In the former case, the input action is not ``None`` and the
        state may be None. Furthermore, ``is_backward`` should be ``True`` or
        ``False`` to indicate the direction of the action. In the latter case, the
        action is ``None`` and the input state may be not ``None``. Furthermore,
        ``is_backward`` is ``None``, indicating that no transition is involved in the
        application of constraints.

        This method simply calls
        :py:meth:`~gflownet.envs.composite.base.CompositeBase._apply_constraints_forward`
        and/or
        :py:meth:`~gflownet.envs.composite.base.CompositeBase._apply_constraints_backward`.

        In general, this method should _not_ be overriden. Instead, environments
        inheriting composite classes may override:
            - `_apply_constraints_forward`
            - `_apply_constraints_backward`

        Parameters
        ----------
        action : tuple (optional)
            An action, used to determine whether and which constraints should
            be applied and which should not, since the computations may be intensive.
            If the call of the method is not initiated by an actioa, then it is
            expected to be ``None``.
        state : dict (optional)
            A state in environment format used to indicate the state of the trajectory
            which should inform whether and which constraints should be applied. If
            ``None``, ``self.state`` may be used.
        is_backward : bool or None
            Boolean flag to indicate whether the constraint should be applied in the
            backward direction (True), meaning 'undoing' the constraint (this is the
            value when the call method is initiated by ``step_backwards()``; or in the
            forward direction, meaning 'applying' the constraint (if initiated by
            ``step()``). If the call of the method is not initiated by an action, then
            the value may be ``None``, indicating that the constraints depend on the
            input state. ``is_backward`` can be not None even if the action is None to
            indicate that either the forward or the backward constraints can be
            ignored. For example, ``reset()`` can pass ``is_backward=True`` to bypass
            the forward constraints.

        Returns
        -------
        bool
            True if any constraint was applied; False otherwise.
        """
        if not self.has_constraints:
            return False

        applied_constraints = False

        # Both forward and backward constraints are attempted if the call method is not
        # initiated by a transition (action is None), unless is_backward is True or
        # False, in which case one of the two directions may be ignored.
        # then is_backward must be None too, and vice versa
        # Forward constraints are applied as well if the call method is initiated
        # by a forward transition (action is not None and is_backward is False)
        # Backward constraints are applied as well if the call method is initiated
        # by a backward transition (action is not None and is_backward is True)
        if action is not None:
            assert isinstance(is_backward, bool)
        do_forward = is_backward is not True
        do_backward = is_backward is not False

        if do_forward:
            applied_constraints = self._apply_constraints_forward(action, state)
        if do_backward:
            applied_constraints = self._apply_constraints_backward(action, state)

        return applied_constraints

    def _apply_constraints_forward(
        self,
        action: Tuple = None,
        state: Optional[Dict] = None,
    ) -> bool:
        """
        Applies constraints across sub-environments in the forward direction.

        Environments inheriting composite classes may override this method if
        constraints across sub-environments must be applied. The method
        :py:meth:`~gflownet.envs.composite.base.CompositeBase._do_constraints_for_subenv`
        may be used as a helper to determine whether the constraints imposed by a
        sub-environment should be applied depending on the action.

        Parameters
        ----------
        action : tuple (optional)
            An action from the global composite environment. If the call of this method
            is not initiated by a transition, then ``action`` is None.
        state : dict (optional)
            A state of the global composite environment.

        Returns
        -------
        bool
            True if any constraint was applied; False otherwise.
        """
        return False

    def _apply_constraints_backward(
        self,
        action: Tuple = None,
        state: Optional[Dict] = None,
    ) -> bool:
        """
        Applies constraints across sub-environments in the backward direction.

        In the backward direction, in this case, means that the constraints between two
        sub-environments are undone and reset as in the source state.

        Environments inheriting composite classes may override this method if
        constraints across sub-environments must be applied. The method
        :py:meth:`~gflownet.envs.composite.base.CompositeBase._do_constraints_for_subenv`
        may be used as a helper to determine whether the constraints imposed by a
        sub-environment should be applied depending on the action.

        Parameters
        ----------
        action : tuple (optional)
            An action from the global composite environment. If the call of this method
            is not initiated by a transition, then ``action`` is None.
        state : dict (optional)
            A state of the global composite environment.

        Returns
        -------
        bool
            True if any constraint was applied; False otherwise.
        """
        return False

    def _do_constraints_for_subenv(
        self,
        state: Union[Dict],
        idx_subenv: int,
        action: Tuple = None,
        is_backward: bool = False,
    ) -> bool:
        """
        Returns True if constraints chould be applied given the state, relevant
        sub-environment, action and direction.

        This method is meant to be used by environments inheriting composite classes
        to determine whether the constraints imposed by a particular sub-environment
        should be applied. This depends on whether the environment is done or not,
        whether the constraints are to be done or undone, and whether they would be
        triggered by a transition or by a state. This method is
        meant to be called from:
        - :py:meth:`~gflownet.envs.composite.base.CompositeBase._apply_constraints_forward`
        - :py:meth:`~gflownet.envs.composite.base.CompositeBase._apply_constraints_backward`

        Additionally, composite environments may include other speciic checks before
        setting inter-environment constraints, besides the output of this method.

        Forward constraints could be applied if:
            - The condition environment is done, and
            - The action is either None or EOS
        Backward constraints could be applied if:
            - The condition environment is not done, and
            - The action is either None or EOS

        Parameters
        ----------
        state : dict
            A state of the global composite environment.
        idx_subenv : int
            Index of the sub-environment that would trigger constraints.
        action : tuple (optional)
            The action involved in the transition, or None if there is no transition.
        is_backward : bool
            Boolean flag to indicate whether the potential constraint is in the
            backward direction (True) or in the forward direction (False).
        """
        # If the index of the sub-environment is -1, then no sub-environment is
        # currently relevant and constraints should not be applied.
        if idx_subenv == -1:
            return False

        # If the action is not None, get the unique environment and depad the action
        if action is not None:
            idx_unique = self._get_unique_idx_of_subenv(idx_subenv, state)
            try:
                action = self._depad_action(action, idx_unique)
            except RuntimeError as e:
                # If there is a mismatch between idx_unique and the action index,
                # return False
                if str(e).startswith(
                    "There is a mismatch between the input idx_unique"
                ):
                    return False
            # If the action is not None, indicating that the check was initiated by a
            # transition, constraints are not applied if the action is not EOS.
            env_unique = self._get_env_unique(idx_unique)
            if action != env_unique.eos:
                return False

        subenv_is_done = self._get_subdone(idx_subenv, state)
        # Backward constraints should only be applied if the sub-environment is not done
        if is_backward:
            return not subenv_is_done
        # Forward constraints hould only be applied if the sub-environment is done
        else:
            return subenv_is_done
