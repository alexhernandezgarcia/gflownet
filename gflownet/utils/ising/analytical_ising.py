#!/usr/bin/env python3
"""
Compute exact partition function (Z), average magnetization (⟨M⟩),
and specific heat (Cv) for an Ising lattice by full enumeration.

Usage:
    python gflownet/utils/ising/exact_stats.py --length 3 --beta 0.4
"""

import argparse
import itertools

import numpy as np
import torch

from gflownet.proxy.ising import Ising


def exact_ising_stats(
    n_dim: int = 2,
    length: int = 3,
    beta: float = 1.0,
    J_nn: float = 1.0,
    h: float = 0.0,
    periodic: bool = True,
    device: str = "cpu",
    threshold: int = 2**30,
):
    """Compute exact Z, ⟨E⟩, ⟨M⟩, and Cv by enumerating all spin configurations."""
    N = length**n_dim
    if 2**N > threshold:
        msg = f"Number of configurations 2^{N} = {2**N} exceeds threshold {THRESHOLD}."
        if n_dim == 2:
            warnings.warn(msg + " Using 2D thermodynamic limit expressions for ⟨|M|⟩")
            # 2D Ising thermodynamic limit formulas (Onsager solution)
            K = beta * J_nn
            # Magnetization (non-zero only for T<Tc)
            Tc = 2.0 / np.log(1 + np.sqrt(2))
            if beta > 1 / Tc:
                avg_abs_M = (1 - (math.sinh(2 * K)) ** (-4)) ** 0.125
            else:
                avg_abs_M = 0.0
            # Heat capacity

    else:
        # Instantiate Ising model
        ising = Ising(n_dim=n_dim, length=length, J_nn=J_nn, h=h, periodic=periodic)
        # Generate all spin configurations
        all_states = torch.tensor(
            list(itertools.product([-1, 1], repeat=N)), dtype=torch.float, device=device
        )

        # Compute energies
        energies = ising(all_states)  # shape (2**N,)
        log_weights = -beta * energies
        max_log_w = torch.max(log_weights)  # Avoid overflow when log_Z is too large.
        # Still having overflow issues for Z.
        Z = torch.exp(log_weights - max_log_w).sum() * torch.exp(max_log_w)
        log_Z = max_log_w + torch.log(torch.exp(log_weights - max_log_w).sum())
        weights = torch.exp(log_weights - log_Z)  # normalized weights

        # Average energy and magnetization
        avg_E = (weights * energies).sum()
        avg_E2 = (weights * energies**2).sum()
        magnetizations = all_states.sum(dim=1)
        abs_magnetizations = magnetizations.abs()
        avg_M = (weights * magnetizations).sum()
        avg_abs_M = (weights * abs_magnetizations).sum()
        avg_abs_M2 = (weights * abs_magnetizations**2).sum()

        # (per site) Magentization, Specific heat and susceptibility, assuming kB = 1
        avg_M_per_site = avg_M / N
        avg_abs_M_per_site = avg_abs_M / N
        Cv_per_site = 1 / N * beta**2 * (avg_E2 - avg_E**2)  # specific heat
        chi_per_site = (
            1 / N * beta * (avg_abs_M2 - avg_abs_M**2)
        )  # magnetic susceptibility

    return dict(
        Z=Z.item(),
        log_Z=log_Z.item(),
        avg_M_per_site=avg_M_per_site.item(),
        avg_abs_M_per_site=avg_abs_M_per_site.item(),
        Cv_per_site=Cv_per_site.item(),
        chi_per_site=chi_per_site.item(),
        N=N,
    )


def main():
    parser = argparse.ArgumentParser(description="Exact Ising lattice statistics")
    parser.add_argument("--n_dim", type=int, default=2, help="Lattice dimensionality")
    parser.add_argument("--length", type=int, default=3, help="Lattice side length")
    parser.add_argument(
        "--beta", type=float, default=0.4, help="Inverse temperature 1/T"
    )
    parser.add_argument(
        "--J_nn", type=float, default=1.0, help="Nearest-neighbor coupling"
    )
    parser.add_argument("--h", type=float, default=0.0, help="External magnetic field")
    parser.add_argument(
        "--periodic", action="store_true", help="Use periodic boundaries"
    )
    parser.add_argument("--device", type=str, default="cpu", help="device")
    parser.add_argument(
        "--threshold",
        type=int,
        default=10**25,
        help="max number of configurations to enumerate",
    )
    args = parser.parse_args()

    res = exact_ising_stats(
        n_dim=args.n_dim,
        length=args.length,
        beta=args.beta,
        J_nn=args.J_nn,
        h=args.h,
        periodic=args.periodic,
        device=args.device,
        threshold=args.threshold,
    )

    print(f"\nExact Ising statistics for {args.n_dim}D lattice of length {args.length}")
    print(f"β = {args.beta:.3f}, J = {args.J_nn:.3f}, h = {args.h:.3f}")
    print(f"Z  = {res['Z']:.6e}")
    print(f"log_Z = {res['log_Z']:.6e}")
    print(f"per site ⟨M⟩ = {res['avg_M_per_site']:.6f}")
    print(f"per site ⟨|M|⟩ = {res['avg_abs_M_per_site']:.6f}")
    print(f"per site Cv = {res['Cv_per_site']:.6f}")
    print(f"per site χ  = {res['chi_per_site']:.6f}")
    print(f"Total spins: {res['N']} (2^{res['N']} configurations)\n")


if __name__ == "__main__":
    main()
