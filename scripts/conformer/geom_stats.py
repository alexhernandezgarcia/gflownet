import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from gflownet.utils.molecule.geom import (all_same_graphs, get_all_confs_geom,
                                          get_conf_geom, get_rd_mol)
from gflownet.utils.molecule.rotatable_bonds import (get_rotatable_ta_list,
                                                     has_hydrogen_tas)

"""
Here we use rdkit_folder format of the GEOM dataset 
Tutorial and downloading links are here: https://github.com/learningmatter-mit/geom/tree/master
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geom_dir",
        type=str,
        default="/home/mila/a/alexandra.volokhova/scratch/datasets/geom",
    )
    parser.add_argument("--output_file", type=str, default="./geom_stats.csv")
    args = parser.parse_args()

    base_path = Path(args.geom_dir)
    drugs_file = base_path / "rdkit_folder/summary_drugs.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    smiles = []
    self_consistent = []
    rdkit_consistent = []
    n_tas_geom = []
    n_tas_rdkit = []
    unique_confs = []
    n_atoms = []
    n_heavy_atoms = []
    hydrogen_tas = []
    for sm, sub_dic in tqdm(list(drugs_summ.items())):
        confs = get_all_confs_geom(base_path, sm, summary_file=drugs_summ)
        if not confs is None:
            rd_mol = get_rd_mol(sm)
            smiles.append(sm)
            unique_confs.append(len(confs))
            n_atoms.append(confs[0].GetNumAtoms())
            n_heavy_atoms.append(confs[0].GetNumHeavyAtoms())
            self_consistent.append(all_same_graphs(confs))
            rdkit_consistent.append(all_same_graphs(confs + [rd_mol]))
            n_tas_geom.append(len(get_rotatable_ta_list(confs[0])))
            n_tas_rdkit.append(len(get_rotatable_ta_list(rd_mol)))
            hydrogen_tas.append(has_hydrogen_tas(confs[0]))

    data = {
        "smiles": smiles,
        "self_consistent": self_consistent,
        "rdkit_consistent": rdkit_consistent,
        "n_rotatable_torsion_angles_geom": n_tas_geom,
        "n_rotatable_torsion_angles_rdkit": n_tas_rdkit,
        "has_hydrogen_tas": hydrogen_tas,
        "n_confs": unique_confs,
        "n_heavy_atoms": n_heavy_atoms,
        "n_atoms": n_atoms,
    }
    df = pd.DataFrame(data)
    df.to_csv(args.output_file)
