"""
A very simple environment to sample one element from a given set of options.

Given a set of options, the environment proceeds to select one of the options from the
source state and then only the end-of-sequence action is valid.
"""

from typing import Iterable, List, Optional, Tuple

import torch.nn.functional as F
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import tfloat, tlong


class Choice(GFlowNetEnv):
    """
    Choice environment.

    States are represented by a single-element list, indicating the index of the
    selected element (starting from 1), or 0 if no option has been selected yet (source
    state).

    The actions of the environment are single-element tuples indicating the index of
    the element to be selected, or -1 for the EOS.

    Attributes
    ----------
    options : iterable
        The description of the options as an iterable of strings. These strings are
        used as readable representation. By default, the string <source> is reserved
        for the source state.
    n_options : int
        The number of options.
    options_available : tuple
        A tuple of option indices (starting from 1) corresponding to the subset of
        options that is available for the environment instance.
    source_readable : str
        The string to be used to represent the source state as a human-readable string.
    """

    def __init__(
        self,
        options: Iterable = None,
        n_options: int = 3,
        source_readable: str = "<source>",
        options_available: Iterable = None,
        **kwargs,
    ):
        """
        Initializes a Choice environment.

        Parameters
        ----------
        options : iterable (optional)
            The descrption of the options. If None, the options are simply described by
            their indices. In this case, ``n_options`` must be not None.
        n_options : int
            The number of options, if ``options`` is None. Ignored otherwise.
        source_readable : str
            The string to be used to represent the source state as a human-readable
            string. By default: <source>
        options_available : iterable (optional)
            A subset of the options to restrict the available options for the
            environment instance. The elements of the iterable are integers
            referring to the option indices.
        """
        self.source_readable = source_readable
        if options is None:
            assert n_options > 0
            options = [str(el) for el in range(1, n_options + 1)]
        else:
            assert self.source_readable not in options
        self.options = options
        self.n_options = len(self.options)
        self.options_indices = set(range(1, self.n_options + 1))
        # Available options
        if options_available is None:
            self.options_available = tuple(self.options_indices)
        else:
            self.options_available = tuple(options_available)
        # Source state: [0]
        self.source = [0]
        # End-of-sequence action
        self.eos = (-1,)
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs list with all possible actions, including EOS.

        Actions are represented by one element, namely the index of the option to be
        selected, starting from 1. The end of sequence action is (-1,).

        Returns
        -------
        list
            A list of tuples representing the actions.
        """
        return [(el,) for el in self.options_indices] + [self.eos]

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
    ) -> List[bool]:
        """
        Returns which actions are invalid (True) and which are not invalid (False).

        Parameters
        ----------
        state : list
            Input state. If None, self.state is used.
        done : bool
            Whether the trajectory is done. If None, self.done is used.

        Returns
        -------
        A list of boolean values.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        # If done is True, return all invalid
        if done:
            return [True] * self.action_space_dim

        # If the state is the source state, all actions of available options are valid
        # except EOS
        if self.is_source(state):
            mask = [
                False if idx in self.options_available else True
                for idx in self.options_indices
            ] + [True]
            return mask
        # Otherwise, only EOS is valid
        else:
            return [True] * self.n_options + [False]

    def get_parents(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        There are only three types of states:
            - Done trajectories: the only parent is the state itself with action EOS.
            - Source state: no parents
            - Option selected: the only parent is the source state.

        Parameters
        ----------
        state : list
            Input state. If None, self.state is used.
        done : bool
            Whether the trajectory is done. If None, self.done is used.
        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format.

        actions : list
            List of actions that lead to state for each parent in parents.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        if done:
            return [state], [self.eos]
        elif self.is_source(state):
            return [], []
        else:
            return [self.source], [(state[0],)]

    def step(
        self, action: Tuple[int, int], skip_mask_check: bool = False
    ) -> [List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Parameters
        ----------
        action : tuple
            Action to be executed. An action is a tuple with a single element indicating
            the the index of the option to be set.
        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The state after executing the action.
        action : tuple
            Action executed.
        valid : bool
            False, if the action is not allowed for the current state.
        """
        # Generic pre-step checks
        do_step, self.state, action = self._pre_step(
            action=action,
            backward=False,
            skip_mask_check=skip_mask_check or self.skip_mask_check,
        )
        if not do_step:
            return self.state, action, False
        valid = True
        self.n_actions += 1

        # If action is EOS, set done to True and return state as is
        if action == self.eos:
            self.done = True
            return self.state, action, valid

        # Update state: simply set the index indicated by the action
        self.state[0] = action[0]

        return self.state, action, valid

    def set_available_options(self, options: Iterable):
        """
        Updates the attribute
        :py:meth:`~gflownet.envs.choice.Choice.options_available`.
        """
        self.options_available = options

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment.

        The maximum and fixed trajectory length is 2: one action to the select the
        optio plus EOS.
        """
        return 2

    def states2policy(self, states: List) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: states
        are one-hot encoded.

        Parameters
        ----------
        states : list
            A batch of states in environment format

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        n_states = len(states)
        states_policy = tlong(states, device=self.device)
        states_policy = F.one_hot(states_policy, num_classes=self.n_options + 1)
        return tfloat(
            states_policy.reshape(n_states, -1),
            float_type=self.float,
            device=self.device,
        )

    def readable2state(self, readable, alphabet={}):
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        if readable == self.source_readable:
            return self.source
        return [self.options.index(readable) + 1]

    def state2readable(self, state: Optional[List] = None, alphabet={}):
        """
        Converts a state into a human-readable string representing a state.

        The readable representation is taken from ``self.options``, except if the state
        is the source state in which case ``self.source_readable`` is returned.
        """
        state = self._get_state(state)
        if self.is_source(state):
            return self.source_readable
        return self.options[state[0] - 1]

    def get_all_terminating_states(self) -> List[List[int]]:
        """
        Returns a list with all the terminating states in environment format.

        Returns
        -------
        list
            The list of all terminating states.
        """
        return [[idx] for idx in self.options_available]

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[List[int]]:
        """
        Constructs a batch of n states uniformly sampled in the sample space of the
        environment.

        Parameters
        ----------
        n_states : int
            The number of states to sample.
        seed : int
            Random seed.
        """
        random.seed(seed)
        return random.choices(self.get_all_terminating_states(), k=n_states)
