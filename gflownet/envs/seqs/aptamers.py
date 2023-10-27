"""
Classes to represent aptamers environments
"""
import itertools
import time
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
import time
from gflownet.utils.sequence.aptamers import NUCLEOTIDES
from gflownet.envs.sequence import Sequence


class Aptamers(Sequence):
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
