from collections import OrderedDict
from pathlib import Path
import yaml

ALPHABET = None

def read_alphabet():
    global ALPHABET
    if ALPHABET is None:
        with open(Path(__file__).parent / "alphabet.yaml", "r") as f:
            ALPHABET = OrderedDict(yaml.safe_load(f))
    return ALPHABET

