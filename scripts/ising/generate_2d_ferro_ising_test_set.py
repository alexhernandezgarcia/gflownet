import argparse
import random
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import torch
from gflownet.proxy.ising import Ising

# --- Fix random seed for reproducibility ---
SEED = 4
np.random.seed(SEED)
random.seed(SEED)


def all_up(L):
    return np.ones((L, L), dtype=int)


def checkerboard(L):
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
    """
    Generate configurations by flipping multiple random blocks of random sizes, with:
      - uniform random block sizes up to L//2
      - uniform random number of clusters up to L//2
      - overlaps allowed 
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
    s = np.ones((L, L), dtype=int)
    s[:, : L // 2] = -1
    return s


def domain_wall_horizontal(L):
    s = np.ones((L, L), dtype=int)
    s[: L // 2, :] = -1
    return s


def stripe_patterns(L):
    horizontal = np.ones((L, L), dtype=int)
    horizontal[::2, :] = -1
    vertical = np.ones((L, L), dtype=int)
    vertical[:, ::2] = -1
    return [horizontal, vertical]


def random_states(L, n_states, bias=None, spread=1):
    """bias='low' → mostly aligned, 'high' → mostly not aligned, None → completely random"""
    states = []
    for _ in range(n_states):
        if bias == "low":  # flip up to L spins from all up config
            s = all_up(L).copy()
            flips = random.sample(range(L * L), random.randint(1, L))
            for f in flips:
                i, j = divmod(f, L)
                s[i, j] *= -1
        elif bias == "high":  # higher energy random states
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
    """Generate one representative configuration per category."""
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
        "M0_random": random_states(L, n_random),
    }
    flipped_categories = {
        f"{k}_flipped": [-s for s in v] for k, v in categories.items()
    }
    categories.update(flipped_categories)
    return categories


def plot_examples(categories, L):
    # Filter out empty categories
    non_empty = [(cat, states) for cat, states in categories.items() if len(states) > 0]
    n = len(non_empty)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    figsize = (min(4 * ncols, 20), min(4 * nrows, 20))  # scale up for readability

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
    plt.savefig(f"ising_testset_examples_L{L}.png", dpi=200)
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
    plt.savefig(f"ising_testset_energy_dist_L{L}.png", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate a representative 2D ferromagnetic Ising model test set, "
            "including fixed configurations (e.g., all-up, domain walls, stripes) "
            "and random configurations at different energy biases."
        )
    )
    parser.add_argument("--length", type=int, default=4, help="Lattice size")
    parser.add_argument(
        "--n_partial_block_up",
        type=int,
        default=1,
        help="Number of partial-block-flip from up. By spin flip symmetry 2n configurations will be generated.",
    )
    parser.add_argument(
        "--n_partial_block_checkerboard",
        type=int,
        default=1,
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
        default=10,
        help="Number of  states with mid low energy.",
    )
    parser.add_argument(
        "--n_random_mid_high",
        type=int,
        default=10,
        help="Number of  states with mid low energy.",
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
    import pickle

    unique_states = []
    for states in cats.values():
        for s in states:
            if not any(np.array_equal(s, u) for u in unique_states):
                unique_states.append(s)

    with open(f"ising_testset_length{args.length}_n{args.n_random}.pkl", "wb") as f:
        pickle.dump({"samples": unique_states}, f)

    # Plot
    if args.plot:
        plot_examples(cats, L=args.length)
        plot_energy_distribution(
            unique_states, L=args.length, J_nn=1.0, h=0.0, periodic=True
        )

    print(f"Saved {len(unique_states)} unique test configurations in pickle file.")
