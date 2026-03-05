import itertools
from typing import Optional, Union

import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


class Ising(Proxy):
    def __init__(
        self,
        n_dim: Optional[int] = None,
        length: Optional[int] = None,
        J_nn: float = 1.0,
        periodic: bool = True,
        J: Optional[torch.Tensor] = None,
        h: Union[float, torch.Tensor] = 0.0,
        **kwargs,
    ):
        r"""
        Ising energy model.

        Implements an Ising Hamiltonian of the form

        .. math::

            E(s) = -\frac{1}{2} s^T J s - h \cdot s

        where ``s`` is a spin configuration, ``J`` is the coupling matrix, and ``h`` is
        an external magnetic field.

        The factor ``1/2`` ensures that each pairwise interaction is counted once.

        Parameters
        ----------
        n_dim : int
            Number of spatial dimensions of the lattice. Not needed if J is provided.
        length : int
            Number of sites along each dimension. Not needed if J is provided.
        J_nn : float, default=1.0
            Nearest-neighbor coupling strength.
        periodic : bool, default=True
            Whether to use periodic boundary conditions.
        J :
            Optional full coupling matrix. Overrides Nearest-neighbor coupling matrix
            construction.
        h :
            External magnetic field (scalar or site-dependent tensor).
        """
        super().__init__(**kwargs)

        self.n_dim = n_dim
        self.length = length
        self.J_nn = J_nn
        self.periodic = periodic

        if J is not None:
            # Use provided arbitrary coupling matrix
            self.J = tfloat(
                J,
                float_type=self.float,
                device=self.device,
            )
        elif n_dim is not None and length is not None:
            # Construct nearest-neighbor adjacency matrix automatically
            self.J = self.nn_adjacency()
        else:
            raise ValueError(
                "Either provide J or both n_dim and length to construct "
                "nearest-neighbor lattice."
            )

        self.h = torch.as_tensor(h, device=self.device, dtype=self.float)

    def nn_adjacency(self) -> torch.Tensor:
        """
        Build the nearest-neighbor adjacency matrix for a hypercubic lattice.

        Constructs a coupling matrix ``J`` for an ``n_dim``-dimensional hypercubic
        lattice of linear size ``length`` (total number of sites ``length**n_dim``),
        where each site is coupled to its nearest neighbors with strength ``J_nn``.

        Returns
        -------
        A tensor of shape ``(N, N)``, where ``N = length**n_dim``, representing the
        nearest-neighbor coupling matrix.
        """
        N = self.length**self.n_dim
        J = torch.zeros((N, N), dtype=self.float, device=self.device)

        coords = list(itertools.product(range(self.length), repeat=self.n_dim))
        coords_to_index = {c: i for i, c in enumerate(coords)}

        for c in coords:
            i = coords_to_index[c]
            for axis in range(self.n_dim):
                # For each axis, we consider both directions : the neighbor on the left
                # (-1) and the neighbor on the right (+1).
                for shift in [-1, 1]:
                    neighbor = list(c)
                    neighbor[axis] += shift

                    if self.periodic:
                        neighbor[axis] %= self.length
                    # if no periodic conditions, the following are not valid neighbors
                    elif neighbor[axis] < 0 or neighbor[axis] >= self.length:
                        continue

                    j = coords_to_index[tuple(neighbor)]
                    J[i, j] = self.J_nn
        return J

    def __call__(
        self, states: Union[list, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch"]:
        """
        Compute the Ising energy for a batch of spin configurations.

        - Flattens input to shape (batch, state_dim).
        - Computes the quadratic interaction term:
                ``-0.5 * sum( (s @ J) * s)``
        - Computes the field term:
                ``-h * sum(s)``                (if h is scalar)
          or ``-sum(h * s)``                   (if h is per-site)


        Parameters
        ----------
        states : list or tensor
            Batch of spin configurations.

        Returns
        -------
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
