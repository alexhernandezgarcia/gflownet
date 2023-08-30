"""
Calculates the compatibility between space group --space_group and all the stoichiometries
spanned by the --max_n_atoms and --max_n_species. The results are written to a file in
--output_dir.
"""
import itertools
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from pyxtal.symmetry import Group
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument(
    "--output_dir",
    default="/home/mila/h/hernanga/crystal-gfn/data/crystals/compatibility_composition_spacegroup",
    type=str,
    help="Output directory",
)
parser.add_argument(
    "--space_group",
    default=1,
    type=int,
    help="Space group",
)
parser.add_argument(
    "--max_n_species",
    default=5,
    type=int,
    help="Maximum number of different elements",
)
parser.add_argument(
    "--max_n_atoms",
    default=20,
    type=int,
    help="Maximum number of atoms per species",
)
args = parser.parse_args()

space_group = args.space_group
assert space_group > 0 and space_group <= 230
max_n_species = args.max_n_species
assert max_n_species > 0 and max_n_species < 10
max_n_atoms = args.max_n_atoms
assert max_n_atoms > 0 and max_n_atoms < 50

filename = f"sg-{space_group}_max_els-{max_n_species}_max_atoms-{max_n_atoms}.csv"
f = open(Path(args.output_dir) / filename, "w")


n_compatible = 0
n_not_compatible = 0
times = []
sg = Group(space_group)
for n_atoms_withzeros in tqdm(
    itertools.product(range(0, max_n_atoms + 1), repeat=max_n_species)
):
    n_atoms = [n for n in n_atoms_withzeros if n > 0]
    time0 = time.time()
    is_compatible, _ = sg.check_compatible(list(n_atoms))
    times.append(time.time() - time0)
    if is_compatible is True:
        n_compatible += 1
    else:
        n_not_compatible += 1

n_total = n_compatible + n_not_compatible
time_mean = np.mean(times)
time_std = np.std(times)

f.write("space_group,n_compatible,n_not_compatible,n_total,time_mean,time_std\n")
f.write(
    f"{space_group},{n_compatible},{n_not_compatible},{n_total},{time_mean},{time_std}\n"
)
