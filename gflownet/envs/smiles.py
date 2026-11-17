"""
Environment to test sequence sentence using word subenv
"""

from typing import Dict, Iterable, List, Optional, Tuple, Union

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.composite.sequence import Sequence
from gflownet.envs.sequences.smiles import SmilesBase
from gflownet.envs.word import Word


# THIS IS THE COMPOSITE ENV
class Smiles(Sequence):
    """
    Environment to test multiple scrabble
    """

    def __init__(
        self,
        max_sequence_length: Optional[int] = 30,
        smiles_vocab: Optional[List[str]] = None,
        left_only: Optional[bool] = False,
        right_only: Optional[bool] = False,
        pad_token: str = "<unk>",
        **kwargs,
    ):

        # only one subenv lol
        subenvs = [
            SmilesBase(
                smiles_vocab=smiles_vocab,
                min_length=1,
                max_length=1,
                pad_token=pad_token,
                device=kwargs.get("device", "cpu"),
            )
        ]
        self.max_n_tokens = max_sequence_length

        # we need to sort the vocab for each subenv
        self._vocab_by_len = sorted(
            (tok for tok in subenvs[0].smiles_vocab if tok), key=len, reverse=True
        )

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

    def states2proxy(self, states: List[Dict]) -> List[str]:
        # for the SEH proxy
        token_env = self._get_env_unique(0)
        smiles = []
        for state in states:
            substates = [self._get_substate(state, key) for key in state["_indices"]]
            if not substates:
                smiles.append("")
                continue
            smiles.append("".join(token_env.states2proxy(substates)))
        return smiles

    def state2readable(self, state: Optional[Dict] = None) -> str:
        state = self._get_state(state)
        tokens = []
        for key in state["_indices"]:
            idx_unique = state["_envs_unique"][key]
            subenv = self._get_env_unique(idx_unique)
            tokens.append(subenv.state2readable(self._get_substate(state, key)))
        return " ".join(tokens)

    def readable2state(self, readable: str) -> Dict:
        tokens = readable.split()
        token_env = self._get_env_unique(0)
        state = {"_active": 0, "_dones": [], "_envs_unique": [], "_indices": []}
        for key, token in enumerate(tokens):
            if token not in token_env.smiles_vocab:
                raise ValueError(f"Unknown SMILES token: {token}")
            state["_envs_unique"].append(0)
            state["_dones"].append(1)
            state["_indices"].append(key)
            state[key] = token_env.readable2state(token)
        return state
