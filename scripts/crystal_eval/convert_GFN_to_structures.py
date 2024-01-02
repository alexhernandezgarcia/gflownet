import pandas as pd
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from argparse import ArgumentParser
from pathlib import Path
import os

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


def make_crystal(spacegroup=1, species=None, num_atoms=None, lattice=None):
    # return None  # too slow and not accurate for now ...
    struct_xtal = pyxtal()
    try:
        struct_xtal.from_random(
            3,
            group=spacegroup,
            species=species,
            numIons=num_atoms,
            lattice=Lattice.from_para(*lattice),
        )
        return struct_xtal.to_pymatgen()
    except Exception as e:
        print(e)
        return None


def encoded_to_comp(sample):
    comp_dic = {
        IDX2ELEM[idx]: amount for idx, amount in enumerate(sample[1:13]) if amount != 0
    }
    return comp_dic


def scale(x, scaler):
    return x * (scaler[1] - scaler[0]) + scaler[0]


def encoded_to_crystal(sample):
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
    args = parser.parse_args()
    data = pd.read_pickle(Path(args.file_path))

    df_data = pd.DataFrame({"encoded": data["x"], "energy": data["energy"]})
    df_data = df_data.tail(n=5)  # given how slow it is, reduce for development phase
    df_data["composition"] = df_data["encoded"].map(encoded_to_comp)
    for struct_idx in range(5):
        df_data[f"structure_{struct_idx}"] = df_data["encoded"].map(encoded_to_crystal)

    df_data["spacegroup"] = df_data["encoded"].map(lambda x: x[SPACEGROUP_IDX])
    out_dir = Path(args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df_data.to_pickle(
        os.path.join(Path(args.out_dir), os.path.basename(Path(args.file_path)))
    )


if __name__ == "__main__":
    main()
