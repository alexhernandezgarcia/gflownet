# TODO: The docstrings need a major rewrite to clarify the new functioning with _keys
# and non-alternating subenvs.
"""
Classes implementing the family of Set meta-environments, which allow to combine
multiple sub-environments without any specific order.
"""

from typing import Iterable, List, Optional

import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.composite.setbase import BaseSet
from gflownet.utils.common import copy


class SetFix(BaseSet):
    """
    Base class to create new environments by arranging (fixed) sets of multiple
    environments.

    Unlike the SetFlex meta-environment, the SetFix works with a fixed set of
    sub-environments. That is, all trajectories will perform actions from all the
    sub-environments defined at the initialization of the environment.

    A practical use case of the SetFix environment is the creation of an environment
    comprising a set of an arbitrary number N of Cube environments, representing for
    example points in an Euclidean space. The actions of the different Cubes can be
    sampled in any order by first sampling the action that selects the corresponding
    Cube. While the trajectories will be longer because of actions to select the
    specific Cube, all Cubes will share the same action space (and mask, and policy
    output), which is desirable since all of them represent the same kind of object.

    Note that the SetFix (as well as the SetFlex) also admits diverse environments in
    the set (for example, Cubes and Grids).
    """

    def __init__(
        self,
        subenvs: Iterable[GFlowNetEnv],
        **kwargs,
    ):
        """
        Parameters
        ----------
        subenvs : iterable
            An iterable containing the set of the sub-environments.
        """
        self.subenvs = tuple(subenvs)
        self.n_subenvs = len(self.subenvs)
        self.max_elements = self.n_subenvs

        # Determine the unique environments
        (
            self.envs_unique,
            _,
            self.unique_indices,
        ) = self._get_unique_environments(self.subenvs)
        self._n_unique_envs = len(self.envs_unique)

        # States are represented as a dictionary with the following keys and values:
        # - Meta-data about the Set
        #   - "_active":  The index of the currently active sub-environment, or -1 if
        #   none is active.
        #   - "_toggle": A flag indicating whether a sub-environment is active before
        #   (1) or after (0) a sub-environment action (in the forward sense).
        #   - "_done": A list of flags indicating whether the sub-environments are done
        #   (1) or not (0).
        #   - "_envs_unique": A list of indices identifying the unique environment
        #   corresponding to each subenv. -1 if there is no sub-environment in that
        #   position. All -1 in the source.
        #   - "_keys": A list of unique integers containing the key at which each
        #   substate is stored. The i-th substate is stored under the key at the i-th
        #   position of this list.
        # - States of the sub-environments, with keys the indices of the subenvs.
        # The only meta-data key specific to the Set (not part of composite
        # environments by default) is "_toggle".
        self.source = {
            "_active": -1,
            "_toggle": 0,
            "_dones": [0] * self.max_elements,
            "_envs_unique": self.unique_indices,
            "_keys": list(range(self.max_elements)),
        }
        self.source.update(
            {idx: subenv.source for idx, subenv in enumerate(self.subenvs)}
        )

        # Get action dimensionality by computing the maximum action length among all
        # sub-environments, and adding 1 to indicate the sub-environment.
        self.action_dim = max([len(subenv.eos) for subenv in self.subenvs]) + 1

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
        self.continuous = any([subenv.continuous for subenv in self.subenvs])

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment, including the EOS
        action.
        """
        return sum([subenv.max_traj_length for subenv in self.subenvs]) * 3 + 1

    # TODO: The current representation is not permutation invariant. In order to
    # properly let a GFlowNet on a Set environment, the representation should be
    # invariant to the permutation of states from the same environment. For example the
    # state of a Set of M d-dimensional points should be represented such that the
    # permutation of the points is invariant. A dummy representation could rely on the
    # randomisation of the subsets of sub-environments that correspond to the same
    # unique environment.
    def states2policy(
        self, states: List[List]
    ) -> TensorType["batch", "state_policy_dim"]:
        """
        Prepares a batch of states in environment format for the policy model.

        The default policy representation is the concatenation of the following
        elements:
        - One-hot encoding of the active sub-environment
        - Toggle flag
        - Done flag of each sub-environment
        - A concatenation of the policy-format states of the sub-environments

        Parameters
        ----------
        states : list
            A batch of states in environment format.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        n_states = len(states)

        # Obtain torch tensors for the active subenvironments, the toggle flags and
        # the done indicators
        active_subenvs = torch.zeros((n_states, self.n_subenvs), dtype=self.float)
        dones = []
        for idx, state in enumerate(states):
            active_subenv = self._get_active_subenv(state)
            if active_subenv != -1:
                active_subenvs[idx, active_subenv] = 1.0
            dones.append(self._get_dones(state))
        dones = torch.tensor(dones, dtype=self.float)

        # Obtain the torch tensor containing the toggle flags
        toggle_flags = torch.tensor(
            [self._get_toggle_flag(s) for s in states], dtype=self.float
        ).reshape(
            (-1, 1)
        )  # reshape to (n_states, 1)

        # Obtain the torch tensor containing the states2policy of the sub-environments
        substates = []
        for idx_subenv, subenv in enumerate(self.subenvs):
            # Collect all substates for the current subenv
            subenv_states = [self._get_substate(s, idx_subenv) for s in states]

            # Convert the subenv_states to policy format
            substates.append(subenv.states2policy(subenv_states))
        substates = torch.cat(substates, dim=1)

        return torch.cat([active_subenvs, toggle_flags, dones, substates], dim=1)

    def states2proxy(
        self, states: List[List]
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
            for idx, subenv in enumerate(self.subenvs):
                self._set_substate(
                    idx, subenv.state2proxy(self._get_substate(state, idx))[0], state
                )
        return states_proxy

    def state2readable(self, state: Optional[List[int]] = None) -> str:
        """
        Converts a state into human-readable representation.

        It concatenates the readable representations of each sub-environment, separated
        by "; " and preceded by Set meta-data: active sub-environment and toggle flag.
        If a sub-environment is done, it is indicanted with " | done" after the state.

        Parameters
        ----------
        state : list
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
        dones = self._get_dones(state)
        readable = (
            f"Active subenv {self._get_active_subenv(state)}; "
            + f"Toggle flag {self._get_toggle_flag(state)};\n"
            + "".join(
                [
                    subenv.state2readable(self._get_substate(state, idx))
                    + _done2str(dones[idx])
                    + ";\n"
                    for idx, subenv in enumerate(self.subenvs)
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
        readables = [readable.strip() for readable in readables[2:]]
        for idx, (subenv, readable) in enumerate(zip(self.subenvs, readables)):
            self._set_substate(
                idx, subenv.readable2state(readable.split(" | ")[0]), state
            )
            self._set_subdone(idx, " | done" in readable, state)
        return state
