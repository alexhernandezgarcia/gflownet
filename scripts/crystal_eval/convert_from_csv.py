import os
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Lattice, Structure


def encoded_to_container_structure(sample: str):
    """Convert CGFN readable sample to pymatgen structure object and extract space group,
    only setting known info (lattice)."""
    pattern = r"Stage 2;\s*(\d+)\s*\|\s*[\w-]+.*?;\s*([\w\d]+[\w\dFOSi]+);\s*\(([\d.]+), ([\d.]+), ([\d.]+)\), \(([\d.]+), ([\d.]+), ([\d.]+)\)"
    match = re.search(pattern, sample)
    if match:
        space_group = match.group(1)
        composition = match.group(2)
        a, b, c, alpha, beta, gamma = [float(x) for x in match.groups()[2:8]]

        lattice = Lattice.from_parameters(
            a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma
        )
        species = []
        composition_dict = Composition(composition).as_dict()
        for element, amount in composition_dict.items():
            for _ in range(int(amount)):
                species.append(element)
        coords = np.zeros((len(species), 3))
        structure = Structure(lattice=lattice, species=species, coords=coords)
        return structure, {
            "spacegroup": space_group
        }  # Return both structure and symmetry info
    else:
        return None, None  # Handle samples that don't match the pattern


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--file_path",
        default="data/crystals/gfn_samples.csv",
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
    print("Starting...")
    df_data = pd.read_csv(args.file_path)
    df_data = df_data.filter(items=["readable", "energies"])
    df_data[["structure", "symmetry"]] = df_data.apply(
        lambda row: encoded_to_container_structure(row["readable"]),
        axis=1,
        result_type="expand",
    )
    df_data.drop(["readable"], axis=1, inplace=True)
    df_data.rename(columns={"energies": "eform"}, inplace=True)
    out_file_path = os.path.join(
        Path(args.out_dir),
        os.path.splitext(os.path.basename(Path(args.file_path)))[0] + ".pkl",
    )
    df_data.to_pickle(out_file_path)
    print(df_data)
    print(f"File saved to {out_file_path}")
    print("Done")


if __name__ == "__main__":
    main()
