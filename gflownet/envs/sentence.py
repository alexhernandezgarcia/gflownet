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
        **kwargs,
    ):

        # only one subenv lol
        subenvs = [
            Word()
        ]

        # Initialize base Sequence environment
        # CHANGE THIS LATER!
        super().__init__(
            envs_unique=tuple(subenvs),
            merge_representations=True,
            # max_sequence_length=5,
            right_only=False,
            left_only=False,
            **kwargs,
        )

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