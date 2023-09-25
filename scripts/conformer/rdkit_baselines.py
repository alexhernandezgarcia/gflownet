import argparse
import numpy as np
import pickle
import os
import copy
import pandas as pd

from datetime import datetime
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry.rdGeometry import Point3D
from tqdm import tqdm

from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule

def gen_multiple_conf_rdkit(smiles, n_confs, optimise=True, randomise_tas=False):
    mols = []
    for _ in range(n_confs):
        mols.append(get_single_conf_rdkit(smiles, optimise=optimise, 
                                          randomise_tas=randomise_tas))
    return mols

def get_single_conf_rdkit(smiles, optimise=True, randomise_tas=False):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    if optimise:
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=0)
        except Exception as e:
            print(e)
    if randomise_tas:
        rotable_bonds = get_torsions(mol)
        values = 3.1415926 * 2 * np.random.rand(len(rotable_bonds))
        conf = mol.GetConformers()[0]
        for rb, val in zip(rotable_bonds, values):
            rdMolTransforms.SetDihedralRad(conf, rb[0], rb[1], rb[2], rb[3], val)
        Chem.rdMolTransforms.CanonicalizeConformer(conf)
    return mol

def write_conformers(conf_list, smiles, output_dir, prefix='', idx=None):
    conf_dict = {'conformer': conf_list, 'smiles': smiles}
    filename = output_dir / f'{prefix}conformer_{idx}.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(conf_dict, file)

def get_torsions(m):
    # taken from https://gist.github.com/ZhouGengmo/5b565f51adafcd911c0bc115b2ef027c
    m = Chem.RemoveHs(copy.deepcopy(m))
    torsionList = []
    torsionSmarts = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"
    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    matches = m.GetSubstructMatches(torsionQuery)
    for match in matches:
        idx2 = match[0]
        idx3 = match[1]
        bond = m.GetBondBetweenAtoms(idx2, idx3)
        jAtom = m.GetAtomWithIdx(idx2)
        kAtom = m.GetAtomWithIdx(idx3)
        for b1 in jAtom.GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            for b2 in kAtom.GetBonds():
                if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                # skip 3-membered rings
                if idx4 == idx1:
                    continue
                # skip torsions that include hydrogens
                if (m.GetAtomWithIdx(idx1).GetAtomicNum() == 1) or (
                    m.GetAtomWithIdx(idx4).GetAtomicNum() == 1
                ):
                    continue
                if m.GetAtomWithIdx(idx4).IsInRing():
                    torsionList.append((idx4, idx3, idx2, idx1))
                    break
                else:
                    torsionList.append((idx1, idx2, idx3, idx4))
                    break
            break
    return torsionList


def clustering(smiles, M=1000, N=100):
    # adapted from https://gist.github.com/ZhouGengmo/5b565f51adafcd911c0bc115b2ef027c
    total_sz = 0
    rdkit_coords_list = []

    # add MMFF optimize conformers, 20x
    rdkit_mols = gen_multiple_conf_rdkit(smiles, n_confs=M, 
                                         optimise=True, randomise_tas=False)
    rdkit_mols = [Chem.RemoveHs(x) for x in rdkit_mols]
    sz = len(rdkit_mols)
    # normalize
    tgt_coords = rdkit_mols[0].GetConformers()[0].GetPositions().astype(np.float32)
    tgt_coords = tgt_coords - np.mean(tgt_coords, axis=0)
    
    for item in rdkit_mols:
        _coords = item.GetConformers()[0].GetPositions().astype(np.float32)
        _coords = _coords - _coords.mean(axis=0)  # need to normalize first
        _R, _score = Rotation.align_vectors(_coords, tgt_coords)
        rdkit_coords_list.append(np.dot(_coords, _R.as_matrix()))
    total_sz += sz

    # add no MMFF optimize conformers, 5x
    rdkit_mols = gen_multiple_conf_rdkit(smiles, n_confs=int(M // 4), 
                                         optimise=False, randomise_tas=False)
    rdkit_mols = [Chem.RemoveHs(x) for x in rdkit_mols]
    sz = len(rdkit_mols)

    for item in rdkit_mols:
        _coords = item.GetConformers()[0].GetPositions().astype(np.float32)
        _coords = _coords - _coords.mean(axis=0)  # need to normalize first
        _R, _score = Rotation.align_vectors(_coords, tgt_coords)
        rdkit_coords_list.append(np.dot(_coords, _R.as_matrix()))
    total_sz += sz

    ### add uniform rotation bonds conformers, 5x
    rdkit_mols = gen_multiple_conf_rdkit(smiles, n_confs=int(M // 4), 
                                         optimise=False, randomise_tas=True)
    rdkit_mols = [Chem.RemoveHs(x) for x in rdkit_mols]
    sz = len(rdkit_mols)

    for item in rdkit_mols:
        _coords = item.GetConformers()[0].GetPositions().astype(np.float32)
        _coords = _coords - _coords.mean(axis=0)  # need to normalize first
        _R, _score = Rotation.align_vectors(_coords, tgt_coords)
        rdkit_coords_list.append(np.dot(_coords, _R.as_matrix()))
    total_sz += sz

    # clustering
    rdkit_coords_flatten = np.array(rdkit_coords_list).reshape(total_sz, -1)
    cluster_size = N
    kmeans = KMeans(n_clusters=cluster_size, random_state=42).fit(rdkit_coords_flatten)
    ids = kmeans.predict(rdkit_coords_flatten)
    # get cluster center
    center_coords = kmeans.cluster_centers_
    coords_list = [center_coords[i].reshape(-1,3) for i in range(cluster_size)]
    mols = []
    for coord in coords_list:
        mol = get_single_conf_rdkit(smiles, optimise=False, randomise_tas=False)
        mol = set_atom_positions(mol, coord)
        mols.append(copy.deepcopy(mol))
    return mols

def set_atom_positions(mol, atom_positions):
    """
    mol: rdkit mol with a single embeded conformer
    atom_positions: 2D np.array of shape [n_atoms, 3]
    """
    conf = mol.GetConformers()[0]
    for idx, pos in enumerate(atom_positions):
        conf.SetAtomPosition(idx, Point3D(*pos))
    return mol

def gen_multiple_conf_rdkit_cluster(smiles, n_confs):
    M = min(10 * n_confs, 2000)
    mols = clustering(smiles, N=n_confs, M=M)
    return mols


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_base_dir', type=str, default='/home/mila/a/alexandra.volokhova/projects/gflownet/results/conformer/samples')
    parser.add_argument('--method', type=str, default='rdkit')
    parser.add_argument('--target_smiles', type=str, 
                        default='/home/mila/a/alexandra.volokhova/projects/gflownet/results/conformer/target_smiles/target_smiles_4_initial.csv')
    parser.add_argument('--n_confs', type=int, default=300)  
    args = parser.parse_args()

    output_base_dir = Path(args.output_base_dir)
    if not output_base_dir.exists():
        os.mkdir(output_base_dir) 
    
    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    ts = Path(args.target_smiles).name[:-4]
    output_dir = output_base_dir / f'{args.method}_samples_{ts}_{timestamp}'
    if output_dir.exists():
        print("Output dir already exisits! Exit generation")
    else:
        os.mkdir(output_dir)
        target_smiles = pd.read_csv(args.target_smiles, index_col=0)
        for idx, (_, item) in tqdm(enumerate(target_smiles.iterrows()), total=len(target_smiles)):
            if args.method == 'rdkit':
                confs = gen_multiple_conf_rdkit(item.smiles, 2 * item.n_confs, optimise=True)
            if args.method == 'rdkit_cluster':
                confs = gen_multiple_conf_rdkit_cluster(item.smiles, 2 * item.n_confs)
            write_conformers(confs, item.smiles, output_dir, prefix=f'{args.method}_', idx=idx)
        print("Finished generation, results are in {}".format(output_dir))
