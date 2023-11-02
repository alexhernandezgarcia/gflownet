"""
Class to represent a Scrabble-inspired environment.
"""
from gflownet.envs.seqs.sequence import Sequence
from gflownet.utils.scrabble.utils import read_alphabet

PAD_TOKEN = "0"


class Scrabble(Sequence):
    """
    A Scrabble-inspired environment based on the Sequence, where the tokens are the
    letters in the English alphabet.

    This basic implementation fully relies on the base sequence class Sequence.
    """

    def __init__(
        self,
        **kwargs,
    ):
        alphabet_dict = read_alphabet()
        super().__init__(tokens=alphabet_dict.keys(), pad_token=PAD_TOKEN, **kwargs)
