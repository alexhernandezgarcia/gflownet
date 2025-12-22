"""
An environment to sample a selection of elements from a given set of options.

The configuration of the environment must determine not only the set of options, but
the maximum number of elements to be sampled, whether fewer than the maximum can be
sampled and whether the selection must proceed with or without replacement.

If the selection is with replacement, then the environment operates as a SetFlex
(without constraints). If the selection is without replacement, then the environment
operates as a SetFix with constraints, such that options that have been already
selected are made unavailable in the remaining environments.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch.nn.functional as F
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.choice import Choice
from gflownet.envs.set import SetFix, SetFlex, make_set
from gflownet.utils.common import tfloat, tlong


def Choices(
    options: Iterable = None,
    n_options: int = 3,
    max_selection: int = 2,
    can_select_fewer_than_max: bool = False,
    with_replacement: bool = True,
    source_readable: str = "<source>",
    **kwargs,
):
    """
    Factory method to instantiate a Choices environment.

    Parameters
    ----------
    options : iterable (optional)
        The descrption of the options. If None, the options are simply described by
        their indices. In this case, ``n_options`` must be not None.
    n_options : int
        The number of options, if ``options`` is None. Ignored otherwise.
    max_selection : int
        The maximum number of options that may be selected.
    can_select_fewer_than_max : bool
        Whether fewer options than the maximum can be selected.
    with_replacement : bool
        Whether the selection proceeds with replacement (True, the same option can be
        selected more than once) or without replacement (False, each option can be
        selected only once).
    source_readable : str
        The string to be used to represent the source state as a human-readable
        string. By default: <source>
    """
    if can_select_fewer_than_max:
        raise NotImplementedError(
            "Selection of fewer than the maximum is currently not implemented"
        )
    return ChoicesSetFix(
        options,
        n_options,
        max_selection,
        can_select_fewer_than_max,
        with_replacement,
        source_readable,
    )


class ChoicesBase:
    """
    ChoicesBase class.

    This class is the base of Choices environments, which inherit from either SetFix or
    SetFlex as well. If the configuration allows for selecting fewer elements than the
    maximum, then the environment becomes a SetFlex. If the number of elements is fixed
    (the maximum), then the environment becomes a SetFix.

    This class determines the inputs that are passed to initialize the Set environment
    (SetFix or SetFlex) allows for overriding the methods that implement functionality
    that is common to both versions, in particular the constraints across environments,
    which depend on whether the selection is with or without replacement.

    If sampling is without replacement, the options that have been already selected are
    made unavailable in the remaining environments.

    ``can_alternate_subenvs`` is always passed as False, since the Choice
    sub-environments only have one meaningful action - selecting the option and then
    EOS.

    Attributes
    ----------
    options : iterable
        The description of the options as an iterable of strings. These strings are
        used as readable representation. By default, the string <source> is reserved
        for the source state.
    n_options : int
        The total number of different options.
    max_selection : int
        The maximum number of options that may be selected.
    can_select_fewer_than_max : bool
        Whether fewer options than the maximum can be selected.
    with_replacement : bool
        Whether the selection proceeds with replacement (True, the same option can be
        selected more than once) or without replacement (False, each option can be
        selected only once).
    source_readable : str
        The string to be used to represent the source state in the Choice environments
        as a human-readable string.
    """

    def __init__(
        self,
        options: Iterable = None,
        n_options: int = 3,
        max_selection: int = 2,
        can_select_fewer_than_max: bool = False,
        with_replacement: bool = True,
        source_readable: str = "<source>",
        **kwargs,
    ):
        """
        Initializes a Choices environment.

        Parameters
        ----------
        options : iterable (optional)
            The descrption of the options. If None, the options are simply described by
            their indices. In this case, ``n_options`` must be not None.
        n_options : int
            The number of options, if ``options`` is None. Ignored otherwise.
        max_selection : int
            The maximum number of options that may be selected.
        can_select_fewer_than_max : bool
            Whether fewer options than the maximum can be selected.
        with_replacement : bool
            Whether the selection proceeds with replacement (True, the same option can be
            selected more than once) or without replacement (False, each option can be
            selected only once).
        source_readable : str
            The string to be used to represent the source state as a human-readable
            string. By default: <source>
        """
        self.max_selection = max_selection
        self.can_select_fewer_than_max = can_select_fewer_than_max
        self.with_replacement = with_replacement

        # Initialize parent class:
        #   - SetFlex if can_select_fewer_than_max is True (not implemented yet)
        #   - SetFix if can_select_fewer_than_max is False
        if self.can_select_fewer_than_max:
            raise NotImplementedError(
                "Selection of fewer than the maximum is currently not implemented"
            )
            env_unique = Choice(
                options=options, n_options=n_options, source_readable=source_readable
            )
            super().__init__(
                envs_unique=(env_unique),
                max_elements=self.max_selection,
                can_alternate_subenvs=False,
                **kwargs,
            )
        else:
            subenvs = (
                Choice(
                    options=options,
                    n_options=n_options,
                    source_readable=source_readable,
                )
                for _ in range(self.max_selection)
            )
            super().__init__(subenvs=subenvs, can_alternate_subenvs=False, **kwargs)

        # Get attributes from sub-environment
        self.options = self.subenvs[0].options
        self.n_options = len(self.options)
        self.source_readable = self.subenvs[0].source_readable

    @property
    def choice_env(self) -> Choice:
        """
        Returns the unique Choice environment.

        Returns
        -------
        Choice
            The Choice environment that serves as unique environment of the Set.
        """
        return self.envs_unique[0]

    def _check_has_constraints(self) -> bool:
        """
        Checks whether the composite environment has constraints across
        sub-environments.

        Constraints need to be applied if the selection is without replacement, since
        the available options of the remaining sub-environments need to be restricted.

        Returns
        -------
        bool
            True if the selection is without replacement, False otherwise
        """
        return not self.with_replacement

    def get_options(self, state: Dict = None) -> Set[int]:
        """
        Returns all the options that have already been chosen from the state.

        Parameters
        ----------
        state : dict
            A state of the global set environment.

        Returns
        -------
        The set of options, as a set of integers.
        """
        if state is None:
            state = self.state
        states = self._get_substates(state)
        return {state[0] for state in states}.difference({0})

    def _apply_constraints_forward(
        self,
        action: Tuple = None,
        state: Optional[Dict] = None,
    ):
        """
        Applies constraints across sub-environments in the forward direction.

        This method is called when ``step()`` and ``set_state()`` are called.

        The available options of the sub-environments that are still to be set are
        restricted to the options that have not been selected yet. This is done by
        restricting the options of the unique environment, which is common to all
        sub-environments.

        Parameters
        ----------
        action : tuple (optional)
            An action from the global set environment. If the call of this method
            is initiated by ``set_state()``, then ``action`` is None.
        state : dict (optional)
            A state of the global set environment.
        """
        idx_subenv = self._get_active_subenv(state)
        if self._do_constraints_for_subenv(state, idx_subenv, action, False):
            options = self.get_options(state)
            options_available = set(self.choice_env.options_indices).difference(options)
            self.choice_env.set_available_options(options_available)

    def _apply_constraints_backward(
        self,
        action: Tuple = None,
        state: Optional[Dict] = None,
    ):
        """
        Applies constraints across sub-environments in the backward direction.

        In the backward direction, in this case, means that the constraints between two
        sub-environments are undone and reset as in the source state.

        This method is called when ``step_backwards()``, ``set_state()`` and
        ``reset()`` are called.

        The available options of the sub-environments that are restricted to the
        options that are not part of the state. Additionally, the option of the
        currently active sub-environment is also addedto the available options, since,
        in the backward sense, it will be unselected and then it will be available.
        This is done by restricting the options of the unique environment, which is
        common to all sub-environments.

        Parameters
        ----------
        action : tuple
            An action from the global composite environment.
        state : dict (optional)
            A state of the global composite environment.
        """
        idx_subenv = self._get_active_subenv(state)
        if self._do_constraints_for_subenv(state, idx_subenv, action, True):
            options = self.get_options(state)
            options_available = set(self.choice_env.options_indices).difference(options)
            # Add option of currently active sub-environment since its option is
            # currently part of the state and thus not available in the forward sense
            # but it should be available assuming it will be unselected in the forward
            # sense.
            option_of_active_subenv = self._get_substate(state, idx_subenv)[0]
            if option_of_active_subenv != 0:
                options_available.add(option_of_active_subenv)
            self.choice_env.set_available_options(options_available)


class ChoicesSetFix(ChoicesBase, SetFix):
    """
    ChoicesSetFix environment.

    This environment is the version of the Choices environments for the configuration
    where the number of elements is fixed (``max_selection``), since fewer elements
    than the maximum are not allowed. This version inherits the SetFix composite
    environment.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Initializes a ChoicesSetFix environment inheriting from the SetFix.
        """
        super().__init__(*args, **kwargs)


class ChoicesSetFlex(ChoicesBase, SetFlex):
    """
    ChoicesSetFlex environment.

    This environment is the version of the Choices environments for the configuration
    where the number of elements is variable, since fewer elements than the maximum are
    allowed. This version inherits the SetFlex composite environment.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Initializes a ChoicesSetFlex environment inheriting from the SetFlex.
        """
        super().__init__(*args, **kwargs)
