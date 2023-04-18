"""
Classes to represent aptamers environments
"""
import itertools
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from polyleven import levenshtein
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.sequence import Sequence
from gflownet.utils.sequence.amp import AMINO_ACIDS


class AMP(Sequence):
    """
    Anti-microbial peptide sequence environment
    """

    def __init__(
        self,
        **kwargs,
    ):
        special_tokens = ["[PAD]", "[EOS]"]
        self.vocab = AMINO_ACIDS + special_tokens
        super().__init__(
            **kwargs,
            special_tokens=special_tokens,
        )
