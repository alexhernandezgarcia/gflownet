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

        # if (
        #     hasattr(self, "proxy")
        #     and self.proxy is not None
        #     and hasattr(self.proxy, "setup")
        # ):
        #     self.proxy.setup(self.max_seq_length)
