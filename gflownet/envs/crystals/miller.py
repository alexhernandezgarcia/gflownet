"""
Class to represent an environment to sample Miller indices (hkl).
"""
import itertools
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchtyping import TensorType

from gflownet.envs.grid import Grid


class MillerIndices(Grid):
    """
    The Miller indices are represented by either 3 parameters (h, k, l) if the
    structure is cubic, or by 4 parameters (h, k, i, l) if the structure is hexagonal
    or rhobohedral. Each parameter can take values in the set {-2, -1, 0, 1, 2}.
    Therefore, we can represent the Miller indices environment by a hyper cube of
    length 5, with dimensionality 3 or 4 depending on the structure.

    Attributes
    ----------
    is_cubic : bool
        True if the structure is cubic, False if the structure is hexagonal or
        rhombohedral.

    max_increment : int
        Maximum increment of each dimension by the actions.

    max_dim_per_action : int
        Maximum number of dimensions to increment per action. If -1, then
        max_dim_per_action is set to n_dim.
    """

    def __init__(
        self,
        is_cubic: bool,
        max_increment: int = 1,
        max_dim_per_action: int = 1,
        **kwargs,
    ):
        if is_cubic:
            n_dim = 3
        else:
            n_dim = 4
        super().__init__(n_dim=n_dim, length=5, cell_min=-2, cell_max=2, **kwargs)
