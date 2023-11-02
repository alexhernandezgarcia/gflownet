"""
Class to represent a Scrabble-inspired environment.
"""
from collections import OrderedDict
from pathlib import Path
import yaml
from gflownet.envs.seqs.sequence import Sequence

ALPHABET = None
PAD_TOKEN = "0"

def _read_alphabet():
    global ALPHABET
    if ALPHABET is None:
        with open(Path(__file__).parent / "alphabet.yaml", "r") as f:
            ALPHABET = OrderedDict(yaml.safe_load(f))
    return ALPHABET

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
        alphabet_dict = _read_alphabet()
        super().__init__(tokens=alphabet_dict.keys(), pad_token=PAD_TOKEN, **kwargs)
