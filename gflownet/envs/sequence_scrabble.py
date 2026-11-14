"""
Environment to test sequence sentence using word subenv
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.composite.sequence import Sequence
from gflownet.envs.scrabble import Scrabble


# Here we will use exactly the same scrabble but we can insert each letter left or right
class SequenceScrabble(Sequence):
    """
    Environment to test multiple scrabble
    """

    def __init__(
        self,
        max_sequence_length: Optional[int] = 7,
        left_only: Optional[bool] = False,
        right_only: Optional[bool] = False,
        **kwargs,
    ):

        # only one subenv lol with one letter each
        subenvs = [Scrabble(max_length=1)]

        # Initialize base Sequence environment
        # CHANGE THIS LATER!
        super().__init__(
            envs_unique=tuple(subenvs),
            merge_representations=True,
            max_sequence_length=max_sequence_length,  # possibly, we want to add this to the config file
            right_only=right_only,
            left_only=left_only,
            **kwargs,
        )

    def state2readable(self, state: Optional[Dict] = None) -> str:
        """Converts a state into a human-readable representation."""
        state = self._get_state(state)
        indices_seq = state["_indices"]
        letters = []
        for key in indices_seq:
            idx_unique = state["_envs_unique"][key]
            # get the class of the subenv so that we can get the state2readable function
            subenv = self._get_env_unique(idx_unique)
            letters.append(subenv.state2readable(self._get_substate(state, key)))
        body = " ".join(letters)
        return body

    # make a readable2state
    def readable2state(self, scrabble_word: str) -> Dict:
        # it is expecting for words (subenvs) to be space separated
        letters = scrabble_word.split(" ")
        # get the subenv object, we know it is only the Word() lol
        scrabble_env = self._get_env_unique(0)
        # this is the format of the composite sequence state
        state = {"_active": 0, "_dones": [], "_envs_unique": [], "_indices": []}
        for key, letter in enumerate(letters):
            state["_envs_unique"].append(0)
            state["_dones"].append(1)
            state["_indices"].append(key)
            state[key] = scrabble_env.readable2state(letter)
        return state
