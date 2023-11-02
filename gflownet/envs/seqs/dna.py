"""
Class to represent DNA sequences.
"""
from gflownet.envs.seqs.sequence import Sequence

NUCLEOBASES = ("A", "C", "T", "G")
PAD_TOKEN = "0"


class DNA(Sequence):
    """
    A DNA sequence environment, where the tokens are the nucleobases A, C, T and G.

    This basic implementation fully relies on the base sequence class Sequence.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(tokens=NUCLEOBASES, pad_token=PAD_TOKEN, **kwargs)
