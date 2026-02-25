"""
Generates representative 2D ferromagnetic Ising model test configurations.

The test set includes the following categories:

Fixed single configurations (1 each):
    - all_up
    - checkerboard
    - domain_wall_vertical
    - domain_wall_horizontal
    - horizontal_stripe
    - vertical_stripe

Parameterized categories (user-controlled counts):
    - partial_block_up
    - partial_block_checkerboard
    - near_ground_random
    - high_energy_random
    - mid_low_energy_random
    - mid_high_energy_random
    - random (fully random)

For every category, spin-flip symmetric counterparts are automatically added,
doubling the number of configurations per category.

Duplicate configurations (if any) are removed before saving.

The script:
    1. Generates configurations for a lattice of size L x L.
    2. Saves unique configurations to a pickle file.
    3. Optionally plots example configurations and their energy distribution.

Use the `--help` option to see how to specify lattice size and the
    number of configurations per category.
"""

import os
import pickle
import random
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gflownet.proxy.ising import Ising

# --- Fix random seed for reproducibility ---
SEED = 4
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------
# Configuration generation
# ----------------------------


def all_up(L):
    r"""
    Generate the fully aligned all up configuration.
    This configuration corresponds to one of the two  ground states of the
    ferromagnetic Ising Hamiltonian.

    Parameters
    ----------
    L : int
        Linear lattice size ($L \times L$ system).

    Returns
    -------
    np.ndarray
        Integer array of shape $(L, L)$ with all spins equal to $+1$.
    """
    return np.ones((L, L), dtype=int)


def checkerboard(L):
    r"""
    Generate a checkerboard configuration.

    For a ferromagnetic coupling ($J>0$), this configuration
    corresponds to a the highest energy state.

    Parameters
    ----------
    L : int
        Linear lattice size ($L \times L$ system).

    Returns
    -------
    np.ndarray
        Integer array of shape $(L, L)$ with alternating $\pm 1$ spins.
    """
    s = np.ones((L, L), dtype=int)
    s[::2, 1::2] = -1
    s[1::2, ::2] = -1
    return s


def get_base_state(L, base):
    if base == "up":
        return all_up(L)
    elif base == "checkerboard":
        return checkerboard(L)
    else:
        raise ValueError(f"Unknown base state '{base}'. Use 'up' or 'checkerboard'.")


def partial_block(L, base, n_states=1, max_clusters=None, max_block_size=None):
    r"""
    Generate configurations by flipping multiple square blocks
    on a base configuration.

    For each configuration:

    1. A base state is selected (all-up or checkerboard).
    2. A random number of square clusters is drawn.
    3. For each cluster, a random size is chosen
       and all spins in that block are flipped to $-1$.

    Number of blocks and block sizes are sampled uniformly
    in the ranges:

    $$
    1 \leq n_{\text{clusters}} \leq \text{max\_clusters}
    $$

    $$
    1 \leq \text{size} \leq \text{max\_block\_size}
    $$

    Overlapping blocks are allowed.

    Parameters
    ----------
    L : int
        Linear lattice size.
    base : str
        Base configuration type (``"up"`` or ``"checkerboard"``).
    n_states : int, optional
        Number of configurations to generate. Default is 1.
    max_clusters : int or None, optional
        Maximum number of flipped blocks per configuration.
        If None, defaults to $L/2$.
    max_block_size : int or None, optional
        Maximum side length of square blocks.
        If None, defaults to $L/2$.

    Returns
    -------
    list of np.ndarray
        2D configurations.
    """

    if max_clusters is None:
        max_clusters = max(1, L // 2)
    if max_block_size is None:
        max_block_size = max(1, L // 2)

    states = []

    for _ in range(n_states):
        s = get_base_state(L, base).copy()

        # Random number of clusters
        n_clusters = np.random.randint(1, max_clusters + 1)

        for _cluster in range(n_clusters):

            # Random block size
            size = np.random.randint(1, max_block_size + 1)

            # Valid positions where this block fits
            positions = [
                (i, j) for i in range(L - size + 1) for j in range(L - size + 1)
            ]

            # Choose one position (allows overlay)
            i0, j0 = positions[np.random.randint(len(positions))]

            # Flip block
            s[i0 : i0 + size, j0 : j0 + size] = -1

        states.append(s)

    return states


def domain_wall_vertical(L):
    r"""
    Generate a vertical domain wall configuration.
    Parameters
    ----------
    L : int
        Linear lattice size ($L \times L$ system).

    Returns
    -------
    np.ndarray
        1 vertical domain wall configuration.
    """
    s = np.ones((L, L), dtype=int)
    s[:, : L // 2] = -1
    return s


def domain_wall_horizontal(L):
    r"""
    Generate a horizontal domain wall configuration.
    Parameters
    ----------
    L : int
        Linear lattice size ($L \times L$ system).

    Returns
    -------
    np.ndarray
        1 horizontal domain wall configuration.
    """

    s = np.ones((L, L), dtype=int)
    s[: L // 2, :] = -1
    return s


def stripe_patterns(L):
    r"""
    Generate horizontal and vertical stripe configurations.

    Parameters
    ----------
    L : int
        Linear lattice size.

    Returns
    -------
    list of np.ndarray
        [horizontal_stripe, vertical_stripe].
    """

    horizontal = np.ones((L, L), dtype=int)
    horizontal[::2, :] = -1
    vertical = np.ones((L, L), dtype=int)
    vertical[:, ::2] = -1
    return [horizontal, vertical]


def random_states(L, n_states, bias=None, spread=1):
    r"""
    Generate random 2D Ising configurations targeting different energy regimes.

    Depending on the value of ``bias``, configurations are constructed by
    perturbing a low-energy reference (ferromagnetic) state, a high-energy
    reference (checkerboard) state, or by sampling spins independently.

    Parameters
    ----------
    L : int
        Linear lattice size (system size is $L \times L$).
    n_states : int
        Number of configurations to generate.
    bias : str or None, optional
        Determines the approximate energy regime:

        - ``"low"``:
          Start from the all-up ferromagnetic ground state
          and flip at most $L$ spins.

        - ``"high"``:
          Start from the checkerboard configuration (maximally frustrated
          for ferromagnetic coupling) and flip at most $L$ spins.

        - ``"mid_low"``:
          Start from all-up and flip approximately

          $$
          \frac{L^2}{2} \times (1 \pm \text{spread})
          $$

          spins.

        - ``"mid_high"``:
          Start from the checkerboard state and flip approximately

          $$
          \frac{L^2}{2} \times (1 \pm \text{spread})
          $$

          spins.

        - ``None``:
          Each spin is sampled independently from $\{-1, +1\}$ with
          uniform probability.

    spread : float, optional
        Controls the relative fluctuation in the number of flipped spins
        for the mid-energy regimes. Default is ``1``.

    Returns
    -------
    list of np.ndarray
        2D configurations.

    Notes
    -----
    The energy biasing is heuristic and does not explicitly enforce
    a target energy value.
    """

    states = []
    for _ in range(n_states):
        if bias == "low":
            # flip up to L spins from all up config
            s = all_up(L).copy()
            flips = random.sample(range(L * L), random.randint(1, L))
            for f in flips:
                i, j = divmod(f, L)
                s[i, j] *= -1
        elif bias == "high":
            # higher energy random states
            s = checkerboard(L).copy()
            flips = random.sample(range(L * L), random.randint(1, L))
            for f in flips:
                i, j = divmod(f, L)
                s[i, j] *= -1
        elif bias == "mid_low":  # mid_low energy
            s = all_up(L).copy()
            n_flip = int(L * L / 2 * (1 + random.uniform(-spread, spread)))
            flips = random.sample(range(L * L), n_flip)
            for f in flips:
                i, j = divmod(f, L)
                s[i, j] *= -1
        elif bias == "mid_high":  # mid_high energy
            s = checkerboard(L).copy()
            n_flip = int(L * L / 2 * (1 + random.uniform(-spread, spread)))
            flips = random.sample(range(L * L), n_flip)
            for f in flips:
                i, j = divmod(f, L)
                s[i, j] *= -1
        else:
            s = np.random.choice([-1, 1], size=(L, L)).astype(int)
        states.append(s)
    return states


def generate_testset(
    L=4,
    n_partial_block_up=10,
    n_partial_block_checkerboard=10,
    n_random_low=10,
    n_random_high=10,
    n_random_mid_low=10,
    n_random_mid_high=10,
    n_random=10,
):
    """Generate testset. The categories with with suffix ``"_flipped"``
    account for global spin-flipped counterparts. Note that potential redundancies can occur.
    """
    categories = {
        "all_up": [all_up(L)],
        "partial_block_up": partial_block(L, base="up", n_states=n_partial_block_up),
        "partial_block_checkerboard": partial_block(
            L, base="checkerboard", n_states=n_partial_block_checkerboard
        ),
        "domain_wall_vertical": [domain_wall_vertical(L)],
        "domain_wall_horizontal": [domain_wall_horizontal(L)],
        "checkerboard": [checkerboard(L)],
        "horizontal_stripe": [stripe_patterns(L)[0]],
        "vertical_stripe": [stripe_patterns(L)[1]],
        "near_ground_random": random_states(L, n_random_low, bias="low"),
        "high_energy_random": random_states(L, n_random_high, bias="high"),
        "mid_low_energy_random": random_states(L, n_random_mid_low, bias="mid_low"),
        "mid_high_energy_random": random_states(L, n_random_mid_high, bias="mid_high"),
        "random": random_states(L, n_random),
    }
    flipped_categories = {
        f"{k}_flipped": [-s for s in v] for k, v in categories.items()
    }
    categories.update(flipped_categories)
    return categories

# ----------------------------
# Plotting
# ----------------------------

def plot_examples(categories, L):
    # Filter out empty categories
    non_empty = [(cat, states) for cat, states in categories.items() if len(states) > 0]
    n = len(non_empty)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    # scale up for readability
    figsize = (min(4 * ncols, 20), min(4 * nrows, 20))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, (cat, states) in enumerate(non_empty):
        ax = axes[i]
        ax.imshow(states[0], cmap="gray", vmin=-1, vmax=1)
        ax.set_title(cat, fontsize=10)
        ax.axis("off")

    # Hide extra axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(f"ising_ferro_testset_examples_L{L}.png", dpi=200)
    plt.close()
    print(f"Saved example configurations to ising_testset_examples_L{L}.png")


def plot_energy_distribution(configs, L, J_nn=1.0, h=0.0, periodic=True):
    # Prepare Ising model
    ising = Ising(n_dim=2, length=L, J_nn=J_nn, h=h, periodic=periodic)
    energies = []
    # Flatten all configurations and compute their energies
    for config in configs:
        flat_config = torch.from_numpy(config.flatten()).float().unsqueeze(0)
        E = ising(flat_config).numpy()
        energies.extend(E)
    energies = np.array(energies)
    # Theoretical bounds (for ferrmoagnetic Ising)
    E_min_theoretical = -2 * J_nn * L**2
    E_max_theoretical = 2 * J_nn * L**2

    # --- Plot histogram ---
    plt.figure(figsize=(6, 4))
    plt.hist(energies, bins=40, alpha=0.7, edgecolor="black")
    plt.axvline(
        E_min_theoretical,
        color="blue",
        linestyle="--",
        label=f"Ground state = {E_min_theoretical:.0f}",
    )
    plt.axvline(
        E_max_theoretical,
        color="red",
        linestyle="--",
        label=f"Max energy = {E_max_theoretical:.0f}",
    )
    plt.xlabel("Energy")
    plt.ylabel("Count")
    plt.title(f"Energy distribution of Ising test set (L={L})")
    plt.tight_layout()
    plt.savefig(f"ising_ferro_testset_energy_dist_L{L}.png", dpi=200)


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "Generate a representative 2D ferromagnetic Ising model test set, "
            "including fixed configurations (e.g., all-up, domain walls, stripes) "
            "and random configurations at different energy biases. Duplicate configurations (if any) are removed before saving."
        )
    )
    parser.add_argument("--length", type=int, default=4, help="Lattice size")
    parser.add_argument(
        "--n_partial_block_up",
        type=int,
        default=10,
        help="Number of partial-block-flip from up. By spin flip symmetry 2n configurations will be generated.",
    )
    parser.add_argument(
        "--n_partial_block_checkerboard",
        type=int,
        default=10,
        help="Number of partial-block-flip from checkerboard. By spin flip symmetry 2n configurations will be generated.",
    )
    parser.add_argument(
        "--n_random_low",
        type=int,
        default=10,
        help="Number of  states with near ground state energy.",
    )
    parser.add_argument(
        "--n_random_high",
        type=int,
        default=10,
        help="Number of  states with high energy.",
    )
    parser.add_argument(
        "--n_random_mid_low",
        type=int,
        default=50,
        help="Number of  states with mid low energy.",
    )
    parser.add_argument(
        "--n_random_mid_high",
        type=int,
        default=50,
        help="Number of  states with mid high energy.",
    )
    parser.add_argument(
        "--n_random", type=int, default=10, help="Number of  completely random states."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot examples of selected configurations and energy distribution.",
    )
    args = parser.parse_args()

    cats = generate_testset(
        L=args.length,
        n_partial_block_up=args.n_partial_block_up,
        n_partial_block_checkerboard=args.n_partial_block_checkerboard,
        n_random_low=args.n_random_low,
        n_random_high=args.n_random_high,
        n_random_mid_low=args.n_random_mid_low,
        n_random_mid_high=args.n_random_mid_high,
        n_random=args.n_random,
    )

    # Save test set as pickle
    unique_states = []
    for states in cats.values():
        for s in states:
            if not any(np.array_equal(s, u) for u in unique_states):
                unique_states.append(s)

    with open(f"ising_ferro_testset_L{args.length}.pkl", "wb") as f:
        pickle.dump({"samples": unique_states}, f)

    # Plot
    if args.plot:
        plot_examples(cats, L=args.length)
        plot_energy_distribution(
            unique_states, L=args.length, J_nn=1.0, h=0.0, periodic=True
        )

    print(f"Saved {len(unique_states)} unique test configurations in pickle file.")
