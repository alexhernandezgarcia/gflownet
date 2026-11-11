"""
Environment to test sequence sentence using word subenv
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.composite.sequence import Sequence
from gflownet.envs.word import Word


# THIS IS THE COMPOSITE ENV
class Sentence(Sequence):
    """
    Environment to test multiple scrabble
    """

    def __init__(
        self,
        max_sequence_length: int = 5,
        **kwargs,
    ):

        # only one subenv lol
        subenvs = [Word()]

        # Initialize base Sequence environment
        # CHANGE THIS LATER!
        super().__init__(
            envs_unique=tuple(subenvs),
            merge_representations=True,
            max_sequence_length=max_sequence_length,  # possibly, we want to add this to the config file
            right_only=False,
            left_only=False,
            **kwargs,
        )

    def state2readable(self, state: Optional[Dict] = None) -> str:
        """Converts a state into a human-readable representation."""
        state = self._get_state(state)
        indices_seq = state["_indices"]
        words = []
        for key in indices_seq:
            idx_unique = state["_envs_unique"][key]
            # get the class of the subenv so that we can get the state2readable function
            subenv = self._get_env_unique(idx_unique)
            words.append(subenv.state2readable(self._get_substate(state, key)))
        body = " ".join(words).capitalize()
        return body + "." if self.done else body

    # make a readable2state
    def readable2state(self, sentence: str) -> Dict:
        # it is expecting for words (subenvs) to be space separated
        words = sentence.strip().rstrip(".").split()
        # get the subenv object, we know it is only the Word() lol
        word_env = self._get_env_unique(0)
        # this is the format of the composite sequence state
        state = {"_active": 0, "_dones": [], "_envs_unique": [], "_indices": []}
        for key, word in enumerate(words):
            word = word.upper()  # upper everything based on Word/scrabble environment
            if len(word) > word_env.max_length:
                # catch some errors
                raise ValueError(f"Word too long for max_length: {word}")
            state["_envs_unique"].append(0)
            state["_dones"].append(1)
            state["_indices"].append(key)
            state[key] = word_env.readable2state(word)
        return state
