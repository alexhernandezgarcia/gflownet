import numpy as np
import random
import matplotlib.pyplot as plt
import argparse

# --- Fix random seed for reproducibility ---
SEED = 44
np.random.seed(SEED)
random.seed(SEED)
def all_up(L):
    return np.ones((L, L), dtype=int)

def single_flip_from_up(L):
    s = all_up(L).copy()
    i, j = 0, 0  # just one example (top-left flipped)
    s[i, j] = -1
    return [s]

def two_adjacent_flips_from_up(L):
    s = all_up(L).copy()
    s[0, 0] = s[0, 1 % L] = -1
    return [s]

def two_opposite_flips_from_up(L):
    s = all_up(L).copy()
    s[0, 0] = s[L//2, L//2] = -1
    return [s]

def domain_wall_vertical(L):
    s = np.ones((L, L), dtype=int)
    s[:, :L//2] = -1
    return [s]

def domain_wall_horizontal(L):
    s = np.ones((L, L), dtype=int)
    s[:L//2, :] = -1
    return [s]

def checkerboard(L):
    s = np.ones((L, L), dtype=int)
    s[::2, 1::2] = -1
    s[1::2, ::2] = -1
    return [s]

def stripe_patterns(L):
    horizontal = np.ones((L, L), dtype=int)
    horizontal[::2, :] = -1
    vertical = np.ones((L, L), dtype=int)
    vertical[:, ::2] = -1
    return [horizontal, vertical]

def block_up(L):
    """Make a block of +1 spins in a -1 background, scaled with L."""
    s = -np.ones((L, L), dtype=int)
    block_size = max(2, L // 4)
    start = (L - block_size) // 2
    s[start:start+block_size, start:start+block_size] = 1
    return [s]

def random_states(L, n, bias=None):
    """bias='low' → mostly aligned, 'high' → mostly not aligned, None → completely random"""
    states = []
    for _ in range(n):
        if bias == "low": #flip up to L spins from all up config
            s = np.ones((L, L))
            flips = random.sample(range(L*L), random.randint(1, L))
            for f in flips:
                i, j = divmod(f, L)
                s[i, j] = -1
        elif bias == "high": # higher energy rnadom states
            s = np.random.choice([-1, 1], size=(L, L))
            while abs(s.mean()) > 0.5:  # ensure roughly balanced magnetization
                #Absolute value > 0.5 means more than 75% of spins are aligned in one direction
                #So lattices with roughly 25%-75% balance are allowed.
                s = np.random.choice([-1, 1], size=(L, L))
        else:
            s = np.random.choice([-1, 1], size=(L, L))
        states.append(s)
    return states

def generate_testset(L=4, n=30):
    """Generate one representative configuration per category."""
    categories = {
        "all_up": [all_up(L)],
        "single_flip_from_up": [single_flip_from_up(L)[0]],
        "two_adjacent_flips_from_up": [two_adjacent_flips_from_up(L)[0]],
        "two_opposite_flips_from_up": [two_opposite_flips_from_up(L)[0]],
        "domain_wall_vertical": [domain_wall_vertical(L)[0]],
        "domain_wall_horizontal": [domain_wall_horizontal(L)[0]],
        "checkerboard": [checkerboard(L)[0]],
        "horizontal_stripe": [stripe_patterns(L)[0]],
        "vertical_stripe": [stripe_patterns(L)[1]],
        "2x2_block_up": [block_up(L)[0]],
        "near_ground_random": random_states(L, n, bias="low"),
        "high_energy_random": random_states(L, n, bias="high"),
        "M0_random": random_states(L, n),
    }
    return categories

def plot_examples(categories, L):
    n = len(categories)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    figsize = (min(4*ncols, 20), min(4*nrows, 20))  # scale up for readability

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, (cat, states) in enumerate(categories.items()):
        ax = axes[i]
        ax.imshow(states[0], cmap="gray", vmin=-1, vmax=1)
        ax.set_title(cat, fontsize=10)
        ax.axis("off")

    # Hide extra axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(f"ising_testset_examples_L{L}.png", dpi=200)
    plt.close()
    print(f"✅ Saved example configurations to ising_testset_examples_L{L}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate a representative 2D ferromagnetic Ising model test set, "
            "including fixed configurations (e.g., all-up, domain walls, stripes) "
            "and random configurations at different energy biases."
        )
    )
    parser.add_argument("--length", type=int, default=4, help="Lattice size")
    parser.add_argument("--n_random", type=int, default=30, help="Number of  states per random category.")
    parser.add_argument("--plot", action="store_true",  help="Plot examples of selected configurations.")
    args = parser.parse_args()

    cats = generate_testset(L=args.length, n=args.n_random)
    if args.plot:
        plot_examples(cats, L=args.length)

    # Save test set as pickle
    import pickle
    unique_states = []
    for states in cats.values():
        for s in states:
            if not any(np.array_equal(s, u) for u in unique_states):
                unique_states.append(s)  

    with open(f"ising_testset_length{args.length}_n{args.n_random}.pkl", "wb") as f:
        pickle.dump({"samples": unique_states}, f)

    print(f"✅ Saved {len(unique_states)} unique test configurations in pickle file.")
