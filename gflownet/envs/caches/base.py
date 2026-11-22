import gc
from sys import getsizeof
from typing import Tuple


class BaseMaskCache:
    """
    Implements a cache to store the (in)validity of specific action from given states.

    The notion of (in)validity used in the caches is slightly different than in the
    masks of invalid actions. In the masks of invalid actions, True indicates that an
    action is invalid and is interpreted as a hard-constraint or a confident decision.
    In contrast, False indicates that the action is not invalid and not necessarily
    that the action is valid. In some environments, it is possible that a non-invalid
    action (False according to the mask) leads to a dead-end trajectory, that is a
    trajectory that cannot reach a terminating staten from which an end-of-sequence
    action is valid. This cache encodes this notion of actions enabling a trajectory
    that can reach a terminating state. In particular, True still indicates that an
    action is invalid, in the sense that it cannot reach a terminating state.
    Similarly, False indicates that an action is not invalid, in the sense that it can
    reach a terminating state. In this case, both values (True and False) are meant to
    indicate a hard constraint or a guaranteed result.

    The base cache is a dictionary with states as keys. Each value is also a dictionary
    with actions as keys and a boolean value indicating whether a terminating state is
    reachable from the state by performing that action.

    Forward and backward transitions are stored in separate dictionaries.

    The states are converted by
    :py:meth:`~gflownet.envs.caches.base.BaseMaskCache._state2key` to be used as keys.
    """

    def __init__(self, store_valid_actions: bool = False):
        """
        Initializes the base mask cache.

        Parameters
        ----------
        store_valid_actions : bool, optional
            Whether valid actions should be stored in the cache, alongside invalid
            actions (which are always stored). Storing all valid actions can make the
            size of the cache explode very quickly. Therefore, valid actions are not
            stored by default (False).
        """
        self._cache_fw = {}
        self._cache_bw = {}
        self._store_valid_actions = store_valid_actions

    def _state2key(self, state) -> frozenset:
        """
        Converts the state into a suitable key for the cache dictionaries.

        This method should generally be overwritten by the specific environment cache
        in order to transform the state in a such that the transformation is valid and
        preserves the necessary information about the state.

        By default, the state is simply converted into a frozenset.

        Parameters
        ----------
        state
            A state in environment format.

        Returns
        -------
        frozenset
            The input state transformed into a frozenset.
        """
        return frozenset(state)

    def get(
        self,
        state: dict,
        action: tuple,
        env=None,
        backward: bool = False,
    ):
        """
        Retrieves an item from the cache if it is already stored or returns None.

        Parameters
        ----------
        state
            A state in environment format.
        action : tuple
            An action from an environment.
        env : GFlowNetEnv, optional
            An instance of an environment. It may be used by certain environments to
            use attributes of the environment in the organisation of the cache. None by
            default.
        backward : bool, optional
            True if the action corresponds to a backward transition. False (default)
            otherwise.

        Returns
        -------
        bool or None
            True if the applying the action from the state is invalid because a
            terminating state cannot be reached. False otherwise, indicating that a
            terminating state can be reached. If the state and action are not in the
            cache, None is returned.
        """
        cache = self._select_cache(backward, env)
        state_key = self._state2key(state)
        if state_key in cache.keys():
            if action in cache[state_key]:
                return cache[state_key][action]
        return None

    def set(
        self,
        is_invalid: bool,
        state,
        action: tuple,
        env=None,
        backward: bool = False,
    ):
        """
        Sets a new item in the cache.

        Parameters
        ----------
        is_invalid : bool
            The value to be stored in the cache: whether the action is invalid (True)
            or valid (False).
        state
            A state in environment format.
        action : tuple
            An action from an environment.
        env : GFlowNetEnv, optional
            An instance of an environment. It may be used by certain environments to
            use attributes of the environment in the organisation of the cache. None by
            default.
        backward : bool, optional
            True if the action corresponds to a backward transition. False (default)
            otherwise.
        """
        if not is_invalid and not self.store_valid_actions(state, env):
            return
        cache = self._select_cache(backward, env)
        state_key = self._state2key(state)
        if state_key not in cache.keys():
            cache[state_key] = {}
        cache[state_key][action] = is_invalid

    def _select_cache(self, backward: bool, env=None) -> dict:
        """
        Returns the cache attribute (forward or backward) corresponding to the input
        parameter.

        Parameters
        ----------
        backward : bool
            True if the action corresponds to a backward transition. False (default)
            otherwise.
        env : GFlowNetEnv, optional
            An instance of an environment. It may be used by certain environments to
            use attributes of the environment in the organisation of the cache. None by
            default.

        Returns
        -------
        dict
            `self._cache_bw` if `backward` is True; `self._cache_fw` if `backward` is
            False.
        """
        if backward:
            return self._cache_bw
        else:
            return self._cache_fw

    def store_valid_actions(self, state, env=None) -> bool:
        """
        Determines wheher a valid action should be stored in the cache for the input
        state.

        If `self._store_valid_actions` is True, then the method returns True
        immediately. Otherwise, it may apply different criteria depending on the input
        state.

        Parameters
        ----------
        state
            A state in environment format.
        env : GFlowNetEnv, optional
            An instance of an environment. It may be used by certain environments to
            use attributes of the environment in the decision. None by default.

        Returns
        -------
        bool
            True if a valid action for the input state should be stored in the cache;
            False otherwise.
        """
        return self._store_valid_actions

    def reset(self, force_garbage_collection=False):
        """
        Clears the cache variables.

        Parameters
        ----------
        force_garbage_collection : bool
            If True, the garbage collector is explicitly called.
        """
        self._cache_fw = {}
        self._cache_bw = {}
        if force_garbage_collection:
            gc.collect()

    def get_size(self) -> int:
        """
        Calculates the total size in bytes of the cache dictionaries.

        Returns
        -------
        int
            The sum of the size in bytes of the forward and the backward caches.
        """
        return getsizeof(self._cache_fw) + getsizeof(self._cache_bw)

    def get_caches(self) -> Tuple[dict, dict]:
        """
        Returns the forward and backward caches.

        Returns
        -------
        cache_fw : dict
            The cache of forward actions.
        cache_bw : dict
            The cache of backward actions.
        """
        return self._cache_fw, self._cache_bw

    def load_caches(self, cache_fw: dict, cache_bw: dict):
        """
        Sets the forward and backward caches passed as arguments as new caches.

        Parameters
        ----------
        cache_fw : dict
            The cache of forward actions.
        cache_bw : dict
            The cache of backward actions.
        """
        if cache_fw is not None:
            self._cache_fw = cache_fw
        if cache_bw is not None:
            self._cache_bw = cache_bw
