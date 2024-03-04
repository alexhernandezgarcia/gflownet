"""
Script for fitting KDE on a dataset of unit cell lengths and angles of materials with
triclinic lattice system. Used for the dummy lattice parameter proxy.
"""

import itertools
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gflownet.envs.crystals.lattice_parameters import LatticeParameters
from gflownet.proxy.crystals.lattice_parameters import PICKLE_PATH

DATASET_PATH = (
    Path(__file__).parents[1] / "data" / "crystals" / "matbench_mp_e_form_lp_stats.csv"
)


if __name__ == "__main__":
    X = pd.read_csv(DATASET_PATH).sample(1000, random_state=0).values
    kde = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kde", KernelDensity(kernel="gaussian", bandwidth=5.0)),
        ]
    ).fit(X)

    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(kde, f)

    # Plot histogram of scores for the original dataset.
    dataset_scores = kde.score_samples(X)
    plt.hist(dataset_scores, bins=100)
    plt.title("Original dataset")
    plt.show()

    # Plot histogram for a grid of samples.
    env = LatticeParameters(
        lattice_system="triclinic",
        min_length=1.0,
        max_length=350.0,
        min_angle=50.0,
        max_angle=150.0,
    )
    linspaces = [env.lengths_tensor.numpy() for _ in range(3)] + [
        env.angles_tensor.numpy() for _ in range(3)
    ]
    grid = np.array(list(itertools.product(*linspaces)))
    grid_scores = kde.score_samples(grid)
    plt.hist(grid_scores, bins=100)
    plt.title("Grid of environment samples")
    plt.show()
