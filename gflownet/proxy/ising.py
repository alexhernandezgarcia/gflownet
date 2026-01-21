import itertools
from typing import Optional, Union

import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


def nn_adjacency(
    n_dim: int, length: int, J_nn: float = 1.0, periodic: bool = True
) -> torch.Tensor:
    """
    Construct adjacency matrix for a hypercubic lattice of size length**n_dim with
    nearest-neighbor coupling J_nn.
    """
    N = length**n_dim
    J = torch.zeros((N, N), dtype=torch.float)

    coords = list(itertools.product(range(length), repeat=n_dim))
    coords_to_index = {c: i for i, c in enumerate(coords)}

    for c in coords:
        i = coords_to_index[c]
        for axis in range(n_dim):
            for shift in [-1, 1]:  # For each axis, we consider both directions :
                # the neighbor on the left (-1) and the neighbor on the right (+1).
                neighbor = list(c)
                neighbor[axis] += shift

                if periodic:
                    neighbor[axis] %= length
                elif neighbor[axis] < 0 or neighbor[axis] >= length:
                    continue  # if no periodic conditions, this is not a valid neighbor

                j = coords_to_index[tuple(neighbor)]
                J[i, j] = J_nn
    return J


class Ising(Proxy):
    def __init__(
        self,
        n_dim: Optional[int] = None,  # No need to provide n_dim if J is provided
        length: Optional[int] = None,  # No need to provide length if J is provided
        J_nn: float = 1.0,
        periodic: bool = True,
        J: Optional[torch.Tensor] = None,
        h: Union[float, torch.Tensor] = 0.0,
        **kwargs,
    ):
        """
        Implements the Ising energy model.

        The class represents an Ising Hamiltonian of the form

            E(s) = -1/2 * sᵀ J s  -  h · s

        where:
            - `s` is a spin state
            - `J` is either a user-provided coupling matrix or a nearest-neighbor
            lattice automatically constructed from (n_dim, length, J_nn)
            - `h` is an optional external magnetic field (scalar or per-site)

        The factor 1/2 ensures each interaction pair (i, j) is counted once.

        Args
        ----
            n_dim, length, J_nn, periodic:
                Used to build a nearest-neighbor lattice if J is not provided.
            J:
                Optional full coupling matrix. Overrides lattice construction.
            h:
                External magnetic field (scalar or site-dependent tensor).
        """
        super().__init__(**kwargs)

        if J is not None:
            # Use provided arbitrary coupling matrix
            self.J = tfloat(
                J,
                float_type=self.float,
                device=self.device,
            )
        elif n_dim is not None and length is not None:
            # Construct nearest-neighbor adjacency matrix automatically
            self.J = nn_adjacency(n_dim, length, J_nn=J_nn, periodic=periodic).to(
                self.device, self.float
            )
        else:
            raise ValueError(
                "Either provide J or both n_dim and length to construct nearest-neighbor lattice."
            )

        self.h = torch.as_tensor(h, device=self.device, dtype=self.float)

    def __call__(
        self, states: Union[list, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch"]:
        """
        Compute the Ising energy for a batch of spin configurations.

            - Flattens input to shape (batch, state_dim).
            - Computes the quadratic interaction term:
                    -0.5 * sum( (s @ J) * s )
            - Computes the field term:
                    -h * sum(s)                (if h is scalar)
              or  -sum(h * s)                  (if h is per-site)

        Returns:
            Tensor of shape (batch,) containing energies.
        """
        if isinstance(states, list):
            states = torch.stack(states)
        # Flatten all but batch dimension    
        states = states.view(states.size(0), -1)  
        # Energy computation. Note the factor 0.5!!
        quadratic_term = -0.5 * torch.sum(states @ self.J * states, dim=1)

        if self.h.numel() == 1:
            field_term = -self.h * torch.sum(states, dim=1)
        else:
            field_term = -torch.sum(states * self.h, dim=1)

        return quadratic_term + field_term
