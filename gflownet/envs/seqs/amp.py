"""
Classes to represent aptamers environments
"""
from typing import List, Tuple
import itertools
import numpy as np
from gflownet.envs.base import GFlowNetEnv
import itertools
from polyleven import levenshtein
import numpy.typing as npt
from torchtyping import TensorType
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from gflownet.utils.sequence.amp import AMINO_ACIDS
from gflownet.envs.sequence import Sequence


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
