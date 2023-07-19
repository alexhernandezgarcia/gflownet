"""
Calculates the compatibility between space groups and stoichiometries for all the
combinations spanned by the N_SYMMETRY_GROUPS, N_SPECIES and MAX_N_ATOMS. The results
are printed to stdout.
"""
import itertools
import time

import numpy as np
from pyxtal.symmetry import Group
from tqdm import tqdm

N_SYMMETRY_GROUPS = 230
N_SPECIES = 3
MAX_N_ATOMS = 20

n_compatible = 0
n_not_compatible = 0
times = []
for idx in tqdm(range(1, N_SYMMETRY_GROUPS + 1)):
    sg = Group(idx)
    times_sg = []
    for n_atoms_withzeros in itertools.product(
        range(0, MAX_N_ATOMS + 1), repeat=N_SPECIES
    ):
        n_atoms = [n for n in n_atoms_withzeros if n > 0]
        time0 = time.time()
        is_compatible, _ = sg.check_compatible(list(n_atoms))
        times_sg.append(time.time() - time0)
        if is_compatible is True:
            n_compatible += 1
        else:
            n_not_compatible += 1
    times.append((np.mean(times_sg), np.std(times_sg)))

n_total = n_compatible + n_not_compatible
pct_compatible = n_compatible / n_total * 100
pct_not_compatible = n_not_compatible / n_total * 100
print(f"Number compatible: {n_compatible}/{n_total} ({pct_compatible} %)")
print(f"Number not compatible: {n_not_compatible}/{n_total} ({pct_not_compatible} %)")

time_per_sg_mean = np.mean([t[0] for t in times]) * 1000
time_per_sg_std = np.mean([t[1] for t in times]) * 1000
n_iters_per_sg = n_total / N_SYMMETRY_GROUPS
time_per_check_mean = time_per_sg_mean / n_iters_per_sg
time_per_check_std = time_per_sg_std / n_iters_per_sg
print(f"Mean (std) time per space group: {time_per_sg_mean} ms ({time_per_sg_std})")
print(f"Mean (std) time per check: {time_per_check_mean} ms ({time_per_check_std})")
