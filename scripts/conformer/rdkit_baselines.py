import argparse
import numpy as np
import pickle
import os

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign as MA
from tqdm import tqdm

from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

def gen_multiple_conf_rdkit(smiles, n_confs=1000):
    mols = []
    for _ in range(n_confs):
        mols.append(get_single_conf_rdkit(smiles))
    return mols

def get_single_conf_rdkit(smiles):
    mol =  Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    try:
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    except Exception as e:
        print(e)
    return mol

def write_conformers(conf_list, smiles, output_dir, prefix=''):
    conf_dict = {'conformer': conf_list}
    filename = output_dir / f'{prefix}conformer_{smiles}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(conf_dict, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./samples')
    parser.add_argument('--method', type=str, default='rdkit')
    parser.add_argument('--target_smiles', type=str, 
                        default='/home/mila/a/alexandra.volokhova/projects/gflownet/results/conformer/target_smiles.pkl')
    parser.add_argument('--n_confs', type=int, default=300)  
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        print("Output dir already exists! Exit generation")  
    else:
        os.mkdir(output_dir)
        with open(args.target_smiles, 'rb') as file:
            smiles_list = pickle.load(file)
        if args.method == 'rdkit':
            for smiles in tqdm(smiles_list):
                confs = gen_multiple_conf_rdkit(smiles, args.n_confs)
                write_conformers(confs, smiles, output_dir, prefix='rdkit_') 
