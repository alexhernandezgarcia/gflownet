"""
Environment to test sequence sentence using word subenv
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.composite.sequence import Sequence
from gflownet.envs.scrabble import Scrabble
from gflownet.utils.common import tlong


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

        # here we add attributes from the original scrabble that will be needed by the proxy
        letter_env = self._get_env_unique(0)  # we know we only have 1 env
        # things needed
        self.pad_token = letter_env.pad_token
        self.pad_idx = letter_env.pad_idx
        self.idx2token = letter_env.idx2token
        self.token2idx = letter_env.token2idx
        self.max_length = self.max_sequence_length  # word length

    def states2proxy(self, states: List[Dict]):
        # for the scrabble proxy
        out = []
        for state in states:
            seq = [self._get_substate(state, key)[0] for key in state["_indices"]]
            seq += [self.pad_idx] * (self.max_length - len(seq))
            out.append(seq)
        return tlong(out, device=self.device)

    def state2readable(self, state: Optional[Dict] = None) -> str:
        """Converts a state into a human-readable representation."""
        state = self._get_state(state)
        letter_env = self._get_env_unique(0)
        body = " ".join(
            letter_env.state2readable(self._get_substate(state, key))
            for key in state["_indices"]
        )
        return body

    # careful! for this part, we use a readable2state that will output a state used by the scrabble proxy
    # def readable2state(self, scrabble_word: str) -> Dict:
    def readable2state(self, scrabble_word: str) -> List[int]:
        # # it is expecting for words (subenvs) to be space separated
        # letters = scrabble_word.split(" ")
        # # get the subenv object, we know it is only the Word() lol
        # scrabble_env = self._get_env_unique(0)
        # # this is the format of the composite sequence state
        # state = {"_active": 0, "_dones": [], "_envs_unique": [], "_indices": []}
        # for key, letter in enumerate(letters):
        #     state["_envs_unique"].append(0)
        #     state["_dones"].append(1)
        #     state["_indices"].append(key)
        #     state[key] = scrabble_env.readable2state(letter)
        # think about how to make it general
        idx = (
            [self.token2idx[t] for t in scrabble_word.split(" ")]
            if scrabble_word
            else []
        )
        state = idx + [self.pad_idx] * (self.max_length - len(idx))
        return state
