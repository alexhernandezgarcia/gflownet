from collections import OrderedDict
from pathlib import Path

import yaml

ALPHABET = None
VOCABULARY = None


def read_alphabet():
    global ALPHABET
    if ALPHABET is None:
        with open(Path(__file__).parent / "alphabet.yaml", "r") as f:
            ALPHABET = OrderedDict(yaml.safe_load(f))
    return ALPHABET


def read_vocabulary():
    global VOCABULARY
    if VOCABULARY is None:
        with open(Path(__file__).parent / "vocabulary_7letters_en", "r") as f:
            VOCABULARY = set(f.read().splitlines())
    return VOCABULARY
