import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from gflownet.utils.molecule.rotatable_bonds import (
    get_rotatable_ta_list,
    is_hydrogen_ta,
)


def get_conf_geom(base_path, smiles, conf_idx=0, summary_file=None):
    if summary_file is None:
        drugs_file = base_path / "rdkit_folder/summary_drugs.json"
        with open(drugs_file, "r") as f:
            summary_file = json.load(f)

    pickle_path = base_path / "rdkit_folder" / summary_file[smiles]["pickle_path"]
    if os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as f:
            dic = pickle.load(f)
        mol = dic["conformers"][conf_idx]["rd_mol"]
        return mol


def get_all_confs_geom(base_path, smiles, summary_file=None):
    if summary_file is None:
        drugs_file = base_path / "rdkit_folder/summary_drugs.json"
        with open(drugs_file, "r") as f:
            summary_file = json.load(f)
    try:
        pickle_path = base_path / "rdkit_folder" / summary_file[smiles]["pickle_path"]
        if os.path.isfile(pickle_path):
            with open(pickle_path, "rb") as f:
                dic = pickle.load(f)
            conformers = [x["rd_mol"] for x in dic["conformers"]]
            return conformers
    except KeyError:
        print("No pickle_path file for {}".format(smiles))
        return None


def get_rd_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    return mol


def has_same_can_smiles(mol1, mol2):
    sm1 = Chem.CanonSmiles(Chem.MolToSmiles(mol1))
    sm2 = Chem.CanonSmiles(Chem.MolToSmiles(mol2))
    return sm1 == sm2


def all_same_graphs(mols):
    ref = mols[0]
    same = []
    for mol in mols:
        same.append(has_same_can_smiles(ref, mol))
    return np.all(same)
