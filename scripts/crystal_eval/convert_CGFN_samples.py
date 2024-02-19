# This script converts the GFN samples (state space) to standard format for evaluation.
# Other datasets (GNOME, Alexiandria, ...) will be parsed to the same format.
# Standard format: pickled dataframe with columns: "structure", "symmetry", "eform"
# structure: pymatgen Structure Object
# symmetry: dictionnary with key "spacegroup". (other keys such as Wyckoff may be added later)
# eform: formation energy in eV

import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
from pyxtal import pyxtal
from pyxtal.lattice import Lattice as pyxtal_lattice

# encoded elements in CGFN pickle
IDX2ELEM = {
    0: "H",
    1: "Li",
    2: "C",
    3: "N",
    4: "O",
    5: "F",
    6: "Mg",
    7: "Si",
    8: "P",
    9: "S",
    10: "Cl",
    11: "Fe",
}
SPACEGROUP_IDX = 15
LATTICE_IDX_START = 16
LENGTH_SCALE = (0.9, 100)
ANGLE_SCALE = (50, 150)


def make_crystal(
    spacegroup: int = 1,
    species: list = None,
    num_atoms: list = None,
    lattice: list = None,
):
    """Makes a random crystal with PyXtal given a spacegroup, species and their amount and a lattice.

    Parameters
    ----------
    spacegroup : int, optional
        Spacegroup number, by default 1
    species : list, optional
       List of species, e.g ["Li","O"], by default None
    num_atoms : list, optional
        List of nuber of atoms per species, e.g [2, 1], by default None
    lattice : list, optional
        list [a, b, c, alpha, beta, gamma], by default None

    Returns
    -------
    _type_
        Pymatgen Structure object
    """
    struct_xtal = pyxtal()
    try:
        struct_xtal.from_random(
            3,
            group=spacegroup,
            species=species,
            numIons=num_atoms,
            lattice=pyxtal_lattice.from_para(*lattice),
        )
        return struct_xtal.to_pymatgen()
    except Exception as e:
        print(e)
        return None  # if fails, we simpy ommit the sample


def encoded_to_comp(sample: list):
    """Converts GFN sample in state space to a composition dictionary."""
    comp_dic = {
        IDX2ELEM[idx]: amount for idx, amount in enumerate(sample[1:13]) if amount != 0
    }
    return comp_dic


def scale(x, scaler):
    """Min-max scale back to real domain, min: scaler[0], max: scaler[1]"""
    return x * (scaler[1] - scaler[0]) + scaler[0]


def encoded_to_crystal(sample: list):
    """Convert CGFN state space sample to pymatgen structure object, using pyXtal (randomness involved)."""
    comp = encoded_to_comp(sample)
    species = list(comp.keys())
    amount = list(comp.values())
    spacegroup = sample[SPACEGROUP_IDX]
    lattice = sample[LATTICE_IDX_START : LATTICE_IDX_START + 6]
    lattice[0:3] = [scale(l, LENGTH_SCALE) for l in lattice[0:3]]
    lattice[3:6] = [scale(a, ANGLE_SCALE) for a in lattice[3:6]]

    return make_crystal(
        spacegroup=spacegroup, species=species, num_atoms=amount, lattice=lattice
    )


def encoded_to_container_structure(sample: list):
    """Convert CGFN state space sample to pymatgen structure object, only setting known info (lattice)"""
    lattice = sample[LATTICE_IDX_START : LATTICE_IDX_START + 6]
    lattice[0:3] = [scale(l, LENGTH_SCALE) for l in lattice[0:3]]
    lattice[3:6] = [scale(a, ANGLE_SCALE) for a in lattice[3:6]]
    lattice = Lattice.from_parameters(
        a=lattice[0],
        b=lattice[1],
        c=lattice[2],
        alpha=lattice[3],
        beta=lattice[4],
        gamma=lattice[5],
    )
    species = []
    composition = encoded_to_comp(sample)
    for element, amount in composition.items():
        for _ in range(amount):
            species.append(element)
    coords = np.zeros((len(species), 3))
    return Structure(lattice=lattice, species=species, coords=coords)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--file_path",
        default="data/crystals/crystalgfn_sgfirst_10ksamples_20231130.pkl",
        type=str,
        help="File path containing pickled CGFN samples",
    )
    parser.add_argument(
        "--out_dir",
        default="data/crystals/eval_data/",
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--n_rand_struct",
        default=0,
        type=int,
        help="Number of structures to generate with pyXtal",
    )

    args = parser.parse_args()
    print("Starting...")
    data = pd.read_pickle(Path(args.file_path))

    df_data = pd.DataFrame({"encoded": data["x"], "eform": data["energy"]})
    # df_data = df_data.sample(n=5) # uncomment for testing
    df_data[f"structure"] = df_data["encoded"].map(encoded_to_container_structure)

    if args.n_rand_struct > 0:
        print("Generating positions with PyXtal")
        for struct_idx in range(args.n_rand_struct):
            df_data[f"structure_{struct_idx}"] = df_data["encoded"].map(
                encoded_to_crystal
            )

    df_data["symmetry"] = df_data["encoded"].map(
        lambda x: {"spacegroup": x[SPACEGROUP_IDX]}
    )
    out_dir = Path(args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df_data.to_pickle(
        os.path.join(Path(args.out_dir), os.path.basename(Path(args.file_path)))
    )
    print(df_data)
    print("Done")


if __name__ == "__main__":
    main()
