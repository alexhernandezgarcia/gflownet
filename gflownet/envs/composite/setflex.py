# TODO: The docstrings need a major rewrite to clarify the new functioning with _keys
# and non-alternating subenvs.
"""
Classes implementing the family of Set meta-environments, which allow to combine
multiple sub-environments without any specific order.
"""

import uuid
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.composite.setbase import BaseSet
from gflownet.utils.common import copy


class SetFlex(BaseSet):
    """
    Base class to create new environments by arranging a variable number (up to a
    maximum) of sub-environments.

    While the (more basic) Set environment is limited to a pre-defined set of
    sub-environments, the SetFlex is a more flexible set. In particular, one can define
    the following properties:
        - The maximum number of elements (sub-environments) in the SetFlex
        - The possible elements (different types of sub-environments) that can be part
          of the SetFlex.

    This flexible implementation enables training a GFlowNet that considers
    environments (SetFlex) whose constituents are a variable number of
    sub-environments.

    For example, we can consider sets of points in an Euclidean space, such as the
    environment Points. Each point is modelled by a ContinuousCube sub-environment. If
    we use the (more basic) Set environment, we need to define the (fixed) number of
    points of the set. Alternatively, we could use the SetFlex to sample a variable
    number of points, from 1 to  self.max_elements.
    """

    def __init__(
        self,
        max_elements: int,
        envs_unique: Iterable[GFlowNetEnv] = None,
        subenvs: Iterable[GFlowNetEnv] = None,
        do_random_subenvs: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        max_elements : int
            The maximum number of enviroments that can be included in the set. Note
            that this number does not refer to the number of unique environments, but
            the number of elements (instances of a sub-environment) that can form a
            set. For example, a SetFlex of up to 10 2D points would contain a single
            unique environment (2D ContinuousCube) with max_elements equal to 10.
        envs_unique : iterable
            An iterable containing the set of unique environments that can make part of
            the set. This iterable is meant to contain unique environments, unique
            meaning that the both the type and the action space are unique.  "Repeated"
            sub-environmens are discarded. If it is None, the unique environments may
            be determined from the argument subenvs.
        subenvs : iterable
            An iterable used to initialize the SetFlex with a specific set of
            sub-environments. This list of environments plays the role of a condition
            for a specific trajectory. If it is None, the set of sub-environments for a
            trajectory may be set via
            :py:meth:`~gflownet.envs.composite.setflex.SetFlex.set_subenvs`.
        do_random_subenvs : bool
            If True, the environment is initialized with a set of random
            sub-environments. If True, also the reset method will resample the set of
            sub-environments. First, the number of elements is sampled uniformly from 1
            to max_elements, then the set of sub-environments is sampled with
            replacement from the set of unique environments. This is practical for
            testing purposes.
        """
        # If both envs_unique and subenvs are None, the environment cannot be
        # initialized
        if envs_unique is None:
            if subenvs is None:
                raise ValueError(
                    "Both envs_unique and subenvs are None. At least one of the "
                    "two variables must contain a set of environments."
                )
            else:
                # Determine the unique environments from the list of sub-environments
                # to determine a trajectory.
                envs_unique = subenvs
        # Determine the unique environments
        (
            self.envs_unique,
            self.envs_unique_keys,
            _,
        ) = self._get_unique_environments(envs_unique)
        self.max_elements = max_elements
        self._n_unique_envs = len(self.envs_unique)
        self.do_random_subenvs = do_random_subenvs

        # Allocate a cache for env instances of each of the unique environments. These
        # instance pools are used in the event of a call to
        # get_env_instances_by_unique_indices() to avoid performing env copies every
        # time these methods are called.
        self.envs_unique_cache = {idx: [] for idx in range(self.n_unique_envs)}

        # States are represented as a dictionary with the following keys and values:
        # - Meta-data about the Set
        #   - "_active":  The index of the currently active sub-environment, or -1 if
        #   none is active.
        #   - "_toggle": A flag indicating whether a sub-environment is active before
        #   (1) or after (0) a sub-environment action (in the forward sense).
        #   - "_done": A list of flags indicating whether the sub-environments are done
        #   (1) or not (0). The flag of environments that do not correspond to a subenv
        #   is set to 1. In the source state, the list is set to all 1s.
        #   - "_envs_unique": A list of indices identifying the unique environment. -1
        #   if there is no sub-environment in that position. All -1 in the source.
        #   - "_keys": A list of unique integers containing the key at which each
        #   substate is stored. The i-th substate is stored under the key at the i-th
        #   position of this list.
        # - States of the sub-environments, with keys the indices of the subenvs. In
        # the source state, there is not any.
        # The only meta-data key specific to the Set (not part of composite
        # environments by default) is "_toggle".
        self.source = {
            "_active": -1,
            "_toggle": 0,
            "_dones": [1] * self.max_elements,
            "_envs_unique": [-1] * self.max_elements,
            "_keys": list(range(self.max_elements)),
        }

        # Set sub-environments
        # - If subenvs is not None, set them as sub-environments
        # - If do_random_subenvs is True, sample a random set of sub-environments
        # - If subenvs is None and do_random_subenvs is False, set the unique
        # environments as sub-environments.
        if subenvs is not None:
            self.set_subenvs(subenvs)
        elif self.do_random_subenvs:
            subenvs = self._sample_random_subenvs()
            self.set_subenvs(subenvs)
        else:
            self.subenvs = None

        # Get action dimensionality by computing the maximum action length among all
        # sub-environments, and adding 1 to indicate the sub-environment.
        self.action_dim = max([len(subenv.eos) for subenv in self.envs_unique]) + 1

        # EOS is a tuple of -1's
        self.eos = (-1,) * self.action_dim

        # Policy distributions parameters
        kwargs["fixed_distr_params"] = [
            env.fixed_distr_params for env in self.envs_unique
        ]
        kwargs["random_distr_params"] = [
            env.random_distr_params for env in self.envs_unique
        ]
        # Base class init
        super().__init__(**kwargs)

        # The set is continuous if any subenv is continuous
        self.continuous = any([subenv.continuous for subenv in self.envs_unique])

    def _get_unique_indices(
        self, state: Optional[Dict] = None, exclude_nonpresent: bool = True
    ) -> int:
        """
        Returns the part of the state containing the unique indices.

        This method is overriden to include the option to exclude the indices of
        non-present sub-environments.

        Parameters
        ----------
        state : dict
            A state of the global composite environment.
        exclude_nonpresent : bool
            If True, return only the indices of sub-environments that are present in
            the state, that is exclude indices with -1.
        """
        unique_indices = super()._get_unique_indices(state)
        if exclude_nonpresent:
            return [idx for idx in unique_indices if idx != -1]
        return unique_indices

    def _compute_unique_indices_of_subenvs(
        self, subenvs: Iterable[GFlowNetEnv]
    ) -> List[int]:
        """
        Identifies the unique environment corresponding to each sub-environment in
        subenvs and returns the list of unique indices.

        Parameters
        ----------
        subenvs : iterable
            Set of sub-environments to be matched with the unique environments.

        Returns
        -------
        list
           A list of indices of the unique environments, with the same length as
           subenvs.
        """
        indices_unique = []
        for env in subenvs:
            try:
                indices_unique.append(
                    self.envs_unique_keys.index((type(env), tuple(env.action_space)))
                )
            except:
                raise ValueError(
                    "The list of subenvs contains a sub-environment that could not "
                    "be matched to one of the existing unique environments"
                )
        return indices_unique

    def _sample_random_subenvs(self) -> List[GFlowNetEnv]:
        """
        Samples randomly the unique indices of a set of sub-environments.

        First, the number of elements is sampled uniformly from 1 to self.max_elements.
        Then, the set of unique indices is sampled with replacement from the set of
        unique environments.

        This method can be practical for testing purposes.

        Returns
        -------
        list
            A list of sub-environments, each with a unique id.
        """
        n_subenvs = np.random.randint(low=1, high=self.max_elements + 1)
        indices_unique = np.random.choice(
            a=self.n_unique_envs, size=n_subenvs, replace=True
        )
        subenvs = self.get_env_instances_by_unique_indices(indices_unique)
        return subenvs

    def set_subenvs(self, subenvs: Iterable[GFlowNetEnv]):
        """
        Sets the sub-environments of the Set and applies the rest of necessary changes
        to the environment.

        - Sets self.subenvs
        - Determines the indices of the unique environments for each subenv.
        - Sets self.state by setting the correct dones, unique indices source states of
          the sub-envs.

        The sub-environments can be thought as the conditioning variables of a specific
        trajectory, since the specific sub-environments are expected to be variable in
        a SetFlex, unlike in the (simpler) Set.

        Parameters
        ----------
        subenvs : iterable
            The list of sub-environments to condition a trajectory.
        """
        self.subenvs = tuple(subenvs)
        n_subenvs = len(self.subenvs)
        # Obtain indices of unique environments and pad with -1's.
        unique_indices = self._compute_unique_indices_of_subenvs(self.subenvs)
        unique_indices += [-1] * (self.max_elements - n_subenvs)
        # Set done of sub-environments to 0 and pad with 1's.
        dones = [0] * n_subenvs + [1] * (self.max_elements - n_subenvs)
        keys = [*range(n_subenvs)] + [-1] * (self.max_elements - n_subenvs)
        # Set self.state
        self.state = {
            "_active": -1,
            "_toggle": 0,
            "_dones": dones,
            "_envs_unique": unique_indices,
            "_keys": keys,
        }
        self.state.update({idx: subenv.source for idx, subenv in enumerate(subenvs)})

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment, including the EOS
        action.

        The maximum length of a trajectory is the maximum of the maximum trajectory
        lengths of the unique environments, times 3 because each action requires
        two Set actions, times :py:const:`self.max_elements`, plus one (EOS).

        Returns
        -------
        int
            The maximum possible length of a trajectory.
        """
        return (
            max([subenv.max_traj_length for subenv in self.envs_unique])
            * 3
            * self.max_elements
            + 1
        )

    def reset(self, env_id: Union[int, str] = None):
        """
        Resets the environment by resetting the sub-environments.

        If self.do_random_subenvs is True, the set of sub-environments is re-sampled.
        """
        if self.do_random_subenvs:
            subenvs = self._sample_random_subenvs()
        elif self.subenvs is not None:
            for subenv in self.subenvs:
                subenv.reset()
            subenvs = self.subenvs
        else:
            subenvs = None

        # If subenv is None, simply call super()'s reset.
        if subenvs is None:
            super().reset(env_id=env_id)
        # Otherwise, reset the environment manually and call set_subenvs. super()'s
        # reset is not called to avoid setting self.state = copy(self.source).
        else:
            self.set_subenvs(subenvs)
            self.n_actions = 0
            self.done = False
            if env_id is None:
                self.id = str(uuid.uuid4())
            else:
                self.id = env_id
        return self

    def get_env_instances_by_unique_indices(self, unique_indices: List):
        """
        Returns a list of env instances corresponding to the requested unique
        environments. The instances have already been reset and their ID set.

        Parameters
        ----------
        unique_indices : list
            Indices of the unique environments

        Returns
        -------
        A list containing instances of the requested environments
        """

        # Allocate counter of how many instances of each unique env have been used
        # to fulfill the request
        envs_counter = {idx: 0 for idx in self.envs_unique_cache.keys()}

        # Go through requested envs, only making new env copies if there aren't already
        # enough in the cache.
        envs = []
        for idx, idx_unique in enumerate(unique_indices):
            # If too few instances of the requested unique env are available, create
            # one more.
            env_instances_used = envs_counter[idx_unique]
            env_instances_available = len(self.envs_unique_cache[idx_unique])
            if env_instances_available <= env_instances_used:
                # Allocate new env instance and add it to the cache
                new_env_instance = self._get_env_unique(idx_unique).copy()
                self.envs_unique_cache[idx_unique].append(new_env_instance)

            # Use one available instance of the requested unique env
            selected_instance = self.envs_unique_cache[idx_unique][env_instances_used]
            selected_instance.reset().set_id(idx)
            envs.append(selected_instance)
            envs_counter[idx_unique] += 1

        return envs

    def set_state(self, state: Dict, done: Optional[bool] = False):
        """
        Sets a state and done.

        It also sets the sub-environments as specified in the unique indices of the
        state by making copies of the unique environments.

        Parameters
        ----------
        state : dict
            A state of the Set environment.
        done : bool
            Whether the trajectory of the environment is done or not.
        """
        # Obtain the sub-environments from the unique indices from the state
        unique_indices = self._get_unique_indices(state)
        subenvs = self.get_env_instances_by_unique_indices(unique_indices)

        # Set sub-environments
        self.set_subenvs(subenvs)

        # If done is True, then the done flags in the set should all be 1
        dones = [bool(el) for el in self._get_dones(state)]
        if done:
            assert all(dones)

        # Call set_state from the parent to set the global state
        super().set_state(state, done)

        # Set state and done of each sub-environment
        for idx, (subenv, done_subenv) in enumerate(zip(self.subenvs, dones)):
            subenv.set_state(self._get_substate(self.state, idx), done_subenv)

        return self

    # TODO: This method is currently implemented with a basic (constant) representation
    # of the states. In order to learn a GFlowNet invariant to permutations of the
    # elements of the Set, the representation needs to be invariant to permutation of
    # the elements.  A simple but potentially effective representation could involve
    # randomly permuting the elements of the set in the policy representation.
    def states2policy(
        self, states: List[Dict]
    ) -> TensorType["batch", "state_policy_dim"]:
        """
        Prepares a batch of states in environment format for the policy model.

        The default policy representation is the concatenation of the following
        elements:
        - One-hot encoding of the active sub-environment
        - Toggle flag
        - Done flag of each sub-environment
        - A vector indicating the number of sub-environments of each unique environment
          present in the state.
        - For each unique environment:
            - A concatenation of the policy-format states of the sub-environments,
              padded up to self.max_elements with the policy representation of the
              source states.

        Parameters
        ----------
        states : list
            A batch of states in environment format.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        n_states = len(states)

        # Obtain torch tensors for the active sub-environments, the toggle flags,
        # the done indicators and the number of subenvs per unique environment
        active_subenvs = torch.zeros((n_states, self.max_elements), dtype=self.float)
        dones = []
        n_subenvs_per_unique_env = []
        for idx_state, state in enumerate(states):
            active_subenv = self._get_active_subenv(state)
            if active_subenv != -1:
                active_subenvs[idx_state, active_subenv] = 1.0
            dones.append(self._get_dones(state))
            n_subenvs = np.zeros(self.n_unique_envs)
            indices, counts = np.unique(
                self._get_unique_indices(state), return_counts=True
            )
            if len(indices) != 0:
                n_subenvs[indices] = counts
            n_subenvs_per_unique_env.append(n_subenvs.tolist())
        dones = torch.tensor(dones, dtype=self.float)
        n_subenvs_per_unique_env = torch.tensor(
            n_subenvs_per_unique_env, dtype=self.float
        )

        # Obtain the torch tensor containing the toggle flags
        toggle_flags = torch.tensor(
            [self._get_toggle_flag(s) for s in states], dtype=self.float
        ).reshape(
            (-1, 1)
        )  # reshape to (n_states, 1)

        # Initialize the policy representation of the states with self.max_elements
        # source states in their policy representation per unique environment.
        substates = torch.tile(
            torch.cat(
                [
                    subenv.state2policy(subenv.source).tile((self.max_elements,))
                    for subenv in self.envs_unique
                ],
                dim=0,
            ),
            (n_states, 1),
        )

        # Obtain the policy representation of the states that are present.
        for idx_state, state in enumerate(states):
            indices_unique = np.array(self._get_unique_indices(state))
            # Obtain the states of each unique environment
            offset = 0
            for idx_unique in range(self.n_unique_envs):
                subenv = self._get_env_unique(idx_unique)
                indices = np.where(indices_unique == idx_unique)[0]
                if len(indices) == 0:
                    offset += subenv.policy_input_dim * self.max_elements
                    continue
                substates_idx = subenv.states2policy(
                    [self._get_substate(state, idx) for idx in indices]
                ).flatten()
                substates[idx_state, offset : offset + substates_idx.shape[0]] = (
                    substates_idx
                )
                offset += subenv.policy_input_dim * self.max_elements

        return torch.cat(
            [active_subenvs, toggle_flags, dones, n_subenvs_per_unique_env, substates],
            dim=1,
        )

    # TODO: this implementation may be useles for flexible sets.
    def states2proxy(
        self, states: List[Dict]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in environment format for a proxy.

        The default proxy format is similar to the environment format, except that the
        states of the sub-enviroments in the dictionary are converted into their
        corresponding proxy formats.

        Parameters
        ----------
        states : list
            A batch of states in environment format.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states_proxy = copy(states)
        for state in states_proxy:
            for idx, substate in enumerate(self._get_substates(state)):
                subenv = self._get_unique_env_of_subenv(idx, state)
                self._set_substate(idx, subenv.state2proxy(substate)[0], state)
        return states_proxy

    def state2readable(self, state: Optional[Dict] = None) -> str:
        """
        Converts a state into human-readable representation.

        It concatenates the readable representations of each sub-environment preceded
        by the index of its unique environment (idx: ...), separated by "; " and all
        preceded by Set meta-data: active sub-environment and toggle flag.  If a
        sub-environment is done, it is indicanted with " | done" after the state.

        Parameters
        ----------
        state : dict
            A state in environment format.

        Returns
        -------
        str
            The state in readable format.
        """

        def _done2str(done: bool):
            """
            Converts a boolean done variable into a string suitable for the readable
            representation.

            Parameters
            ----------
            done : bool
                A boolean variable indicating whether a trajectory is done.

            Returns
            -------
            str
                " | done" if done is True; "" otherwise.
            """
            if done:
                return " | done"
            else:
                return ""

        state = self._get_state(state)
        indices_unique = self._get_unique_indices(state)
        dones = self._get_dones(state)
        substates = self._get_substates(state)
        readable = (
            f"Active subenv {self._get_active_subenv(state)}; "
            + f"Toggle flag {self._get_toggle_flag(state)};\n"
            + "".join(
                [
                    f"{idx}: "
                    + self._get_env_unique(idx).state2readable(substate)
                    + _done2str(done)
                    + ";\n"
                    for idx, done, substate in zip(indices_unique, dones, substates)
                ]
            )
        )
        readable = readable[:-2]
        return readable

    def readable2state(self, readable: str) -> List[int]:
        """
        Converts a human-readable representation of a state into the standard format.
        """
        state = copy(self.source)
        readables = readable.split(";")
        self._set_active_subenv(int(readables[0].split(" ")[-1]), state)
        self._set_toggle_flag(int(readables[1].split(" ")[-1]), state)
        self._set_keys(
            [*range(len(self.subenvs))]
            + [-1] * (self.max_elements - len(self.subenvs)),
            state,
        )
        readables = [readable.strip() for readable in readables[2:]]
        for idx, (subenv, readable) in enumerate(zip(self.subenvs, readables)):
            idx_unique, readable = readable.split(": ")
            self._set_unique_index(idx, int(idx_unique), state)
            self._set_substate(
                idx, subenv.readable2state(readable.split(" | ")[0]), state
            )
            self._set_subdone(idx, " | done" in readable, state)
        return state

    def is_source(self, state: Optional[Dict] = None) -> bool:
        """
        Returns True if the environment's state or the state passed as parameter (if
        not None) is the source state of the environment.

        This method is overriden to check the meta-data about non-present states, which
        is special of the SetFlex..

        Parameters
        ----------
        state : dict
            None, or a state in environment format.

        Returns
        -------
        bool
            Whether the state is the source state of the environment
        """
        # First, check if the state is source, considering the generic SetBase metadata
        if not super().is_source(state):
            return False

        # Otherwise, check the SetFlex specific metadata
        state = self._get_state(state)
        n_subenvs = 0
        for idx in range(self.max_elements):
            if idx in state:
                n_subenvs += 1
        n_left = self.max_elements - n_subenvs
        if self._get_dones(state) != [0] * n_subenvs + [1] * n_left:
            return False
        if self._get_unique_indices(state, False)[n_subenvs:] != [-1] * n_left:
            return False
        return True

    # TODO: Try to find a better solution to this issue when sampling random subenvs
    def __eq__(self, other, ignored_keys: List[str] = []) -> bool:
        """
        Checks whether the current environment instance is equal to the input
        environment instance.

        This method is overriden to ignore the keys:
            - ``subenvs`` if ``self.do_random_subenvs`` is True, because in this case
              resetting the environment will resample the sub-environments.
            - ``state`` if ``self.do_random_subenvs`` is True, because in this case
              resetting the environment will resample the sub-environments and set the
              state accordingly.

        Parameters
        ----------
        other : GFlowNetEnv
            The environment instance to be compared.
        ignored_keys : list
            A list of keys (strings) to be ignored in the comparison. This parameter
            may be used by subclasses that may need to ignore certain keys.

        Returns
        -------
        bool
            True if the environments's attributes are considered equal; False otherwise.
        """
        if self.do_random_subenvs:
            ignored_keys = ignored_keys + ["subenvs", "state"]
        return super().__eq__(other, ignored_keys=ignored_keys)
