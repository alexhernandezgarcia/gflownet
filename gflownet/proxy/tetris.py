import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


class Tetris(Proxy):
    def __init__(self, normalize, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

    def setup(self, env=None):
        if env:
            self.height = env.height
            self.width = env.width

    @property
    def norm(self):
        if self.normalize:
            return -(self.height * self.width)
        else:
            return -1.0

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        if states.dim() == 2:
            return torch.sum(states) / self.norm

        elif states.dim() == 3:
            return torch.sum(states, axis=(1, 2)) / self.norm
        else:
            raise ValueError


class DensityTetris(Proxy):
    """Tetris proxy with rewards based on density

    In this proxy, the rewards are computed based on occupied_space / used_space with :
    - occupied_space being the number of cells occupied by a Tetris piece
    - used_space being the area of the smallest rectangle encompassing all Tetris pieces
      present in the state.

    The goal of this reward structure is to allow model that can use intermediate rewards to learn
    to densely pack Tetris pieces together from the beginning instead of placing them far across the
    board.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, env=None):
        if env:
            self.height = env.height
            self.width = env.width

    def get_2d_state_density(self, state):
        """
        Compute the density of Tetris pieces in a 2D state tensor

        This density is equal to ratio between the number of cells occupied by a Tetris piece and
        the area of the smallest rectangle containing all the placed Tetris pieces.

        Args
        ----
        state : tensor

        Returns
        -------
        torch.tensor
            Density of Tetris pieces in the state tensor (scalar)
        """
        # Compute the area used
        rows_used = (state.sum(1) > 0).nonzero()
        if len(rows_used) == 0:
            nb_rows_used = torch.tensor(0)
        else:
            nb_rows_used = rows_used.max() - rows_used.min() + 1

        cols_used = (state.sum(0) > 0).nonzero()
        if len(cols_used) == 0:
            nb_cols_used = torch.tensor(0)
        else:
            nb_cols_used = cols_used.max() - cols_used.min() + 1

        area_used = nb_rows_used * nb_cols_used

        # Compute the number of cells occupied by tetris pieces
        area_occupied = state.sum()

        # Compute the density of cells in the area used
        return (area_occupied + 1e-6) / (area_used.float() + 1e-6)

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        if states.dim() == 2:
            return self.get_2d_state_density(states)

        elif states.dim() == 3:
            density_per_sample = [self.get_2d_state_density(s) for s in states]
            return torch.stack(density_per_sample)

        else:
            raise ValueError
