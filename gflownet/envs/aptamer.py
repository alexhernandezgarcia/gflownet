"""
Classes to represent aptamers environments
"""
import itertools
import time
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd

from gflownet.envs.sequence import Sequence
from gflownet.utils.sequence.aptamers import NUCLEOTIDES


class Aptamer(Sequence):
    """
    Aptamer sequence environment
    """

    def __init__(
        self,
        **kwargs,
    ):
        special_tokens = ["[PAD]", "[EOS]"]
        self.vocab = NUCLEOTIDES + special_tokens
        super().__init__(
            **kwargs,
            special_tokens=special_tokens,
        )
