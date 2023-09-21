import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd

from rdkit import Chem
from pathlib import Path
from tqdm import tqdm

from gflownet.utils.molecule.rotatable_bonds import get_rotatable_ta_list, is_hydrogen_ta

"""
Here we use rdkit_folder format of the GEOM dataset 
Tutorial and downloading links are here: https://github.com/learningmatter-mit/geom/tree/master
"""

def get_conf_geom(base_path, smiles, conf_idx=0, summary_file=None):
    if summary_file is None:
        drugs_file = base_path / 'rdkit_folder/summary_drugs.json'
        with open(drugs_file, "r") as f:
            summary_file = json.load(f)

    pickle_path = base_path / "rdkit_folder" / summary_file[smiles]['pickle_path']
    if os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as f:
            dic = pickle.load(f)
        mol = dic['conformers'][conf_idx]['rd_mol']
        return mol

def get_all_confs_geom(base_path, smiles, summary_file=None):
    if summary_file is None:
        drugs_file = base_path / 'rdkit_folder/summary_drugs.json'
        with open(drugs_file, "r") as f:
            summary_file = json.load(f)
    try:
        pickle_path = base_path / "rdkit_folder" / summary_file[smiles]['pickle_path']
        if os.path.isfile(pickle_path):
            with open(pickle_path, "rb") as f:
                dic = pickle.load(f)
            conformers = [x['rd_mol'] for x in dic['conformers']]
            return conformers
    except KeyError:
        print('No pickle_path file for {}'.format(smiles))

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

def has_hydrogen_tas(mol):
    tas = get_rotatable_ta_list(mol)
    hydrogen_flags = []
    for t in tas:
        hydrogen_flags.append(is_hydrogen_ta(mol, t))
    return np.any(hydrogen_flags)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geom_dir', type=str, default='/home/mila/a/alexandra.volokhova/scratch/datasets/geom')
    parser.add_argument('--output_dir', type=str, default='./')
    args = parser.parse_args()

    base_path = Path(args.geom_dir)
    drugs_file = base_path / 'rdkit_folder/summary_drugs.json'
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
    'smiles': smiles,
    'self_consistent': self_consistent,
    'rdkit_consistent': rdkit_consistent,
    'n_rotatable_torsion_angles_geom' : n_tas_geom,
    'n_rotatable_torsion_angles_rdkit' : n_tas_rdkit,
    'has_hydrogen_tas': hydrogen_tas, 
    'n_confs': unique_confs,
    'n_heavy_atoms': n_heavy_atoms,
    'n_atoms': n_atoms,
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(args.output_dir, 'geom_stats.csv'))


