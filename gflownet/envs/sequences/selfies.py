"""
Class to represent SELFIES molecules.
"""

from typing import List

from gflownet.envs.sequences.base import SequenceBase

SELFIES_VOCAB_SMALL = [
    "[#Branch1]",
    "[#Branch2]",
    "[#C]",
    "[#N]",
    "[=Branch1]",
    "[=Branch2]",
    "[=C]",
    "[=N]",
    "[=O]",
    "[=Ring1]",
    "[=Ring2]",
    "[=S]",
    "[B]",
    "[Br]",
    "[Branch1]",
    "[Branch2]",
    "[C]",
    "[Cl]",
    "[F]",
    "[NH1]",
    "[N]",
    "[O]",
    "[P]",
    "[Ring1]",
    "[Ring2]",
    "[S]",
]
PAD_TOKEN = "[nop]"


class SelfiesEnv(SequenceBase):
    """
    A SELFIES sequence environment whose tokens are valid SELFIES tokens.

    This basic implementation fully relies on the base sequence class Sequence.
    """

    def __init__(
        self,
        selfies_vocab: List[str] | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        selfies_vocab : List[str] | None
            The list of SELFIES tokens to use as the vocabulary.
            If None (default), the small vocabulary defined in
            SELFIES_VOCAB_SMALL is used.
        """
        if selfies_vocab is None:
            selfies_vocab = SELFIES_VOCAB_SMALL
        self.selfies_vocab = selfies_vocab
        super().__init__(tokens=self.selfies_vocab, pad_token=PAD_TOKEN, **kwargs)
