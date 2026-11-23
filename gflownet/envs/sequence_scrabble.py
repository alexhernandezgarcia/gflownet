"""
Environment to test sequence scrabble
"""
from typing import Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import torch
from torchtyping import TensorType
from tqdm import tqdm

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.composite.sequence import Sequence
from gflownet.envs.scrabble import Scrabble
from gflownet.utils.common import copy

from itertools import product
from string import ascii_uppercase

class OneLetterScrabble(Scrabble):
    def get_mask_invalid_actions_forward(self, state=None, done=None):
        state = self._get_state(state)
        mask = super().get_mask_invalid_actions_forward(state, done)
        if state == self.source:
            mask[self.action_space.index(self.eos)] = True
        return mask

# THIS IS THE COMPOSITE ENV
class SequenceScrabble(Sequence):
    """
    Environment to test multiple scrabble
    """

    def __init__(
        self,
        right_only=False,
        sequence_length=4,
        min_sequence_length=4,
        **kwargs,
    ):
        # Initialize list of subenvs:
        #subenvs = [
        #        OneLetterScrabble(letters=["A"], max_length=1),
        #        OneLetterScrabble(["B"], max_length=1),
        #        OneLetterScrabble(["C"], max_length=1),
        #        OneLetterScrabble(["D"], max_length=1)
        #    ]
        subenvs = [
            OneLetterScrabble(letters=[letter], max_length=1)
            for letter in ascii_uppercase[:sequence_length]
        ]
        # Initialize base Stack environment
        super().__init__(
            envs_unique=tuple(subenvs),
            merge_representations=True,
            max_sequence_length=sequence_length,
            min_sequence_length=min_sequence_length,
            right_only=right_only,
            left_only=False,
            **kwargs,
        )

    def _get_unique_environments(self, subenvs):
        envs_unique = []
        envs_unique_keys = []
        unique_indices = []
        for env in subenvs:
            env_key = (type(env), tuple(env.action_space), tuple(env.letters))
            if env_key not in envs_unique_keys:
                envs_unique_keys.append(env_key)
                envs_unique.append(env)
            unique_indices.append(envs_unique_keys.index(env_key))
        return envs_unique, tuple(envs_unique_keys), unique_indices

    def _compute_unique_indices_of_subenvs(
        self, subenvs: Iterable[GFlowNetEnv]
    ) -> List[int]:
        """
        Returns the list of unique-environment indices corresponding to each
        sub-environment in ``subenvs``.
        """
        indices_unique = []
        for env in subenvs:
            try:
                indices_unique.append(
                    self.envs_unique_keys.index(
                        (type(env), tuple(env.action_space), tuple(env.letters))
                    )
                )
            except ValueError:
                raise ValueError(
                    "The list of subenvs contains a sub-environment that could not "
                    "be matched to one of the existing unique environments"
                )
        return indices_unique

    def state2readable(self, state: Optional[Dict] = None) -> str:
        """Converts a state into a human-readable representation."""
        state = self._get_state(state)
        indices_seq = state["_indices"]
        body = ""
        for key in indices_seq:
            idx_unique = state["_envs_unique"][key]
            # get the class of the subenv so that we can get the state2readable function
            subenv = self._get_env_unique(idx_unique)
            # this part is to remove spaces between the letters in the subenv, this is needed
            body += str(subenv.state2readable(self._get_substate(state, key))).replace(" ", "")
        return body

    def states2proxy(self, states):
        return [self.state2readable(state) for state in states]

    def get_all_states(self):
        """enumerate all intermedia states"""
        states = []
        env = self.copy()

        for length in range(self.max_sequence_length + 1):
            for indices in product(range(self.n_unique_envs), repeat=length):
                env.reset()
                for index in indices:
                    direction =0 if env.is_source() else 1

                    env.step((-1, index, direction))
                    env.step((index, 1, 0))
                    env.step((index, -1, 0))
                    env.step((-1, index, direction))

                states.append(copy(env.state))
        
        return states
