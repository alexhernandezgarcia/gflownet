# some functions inspired by: https://gist.github.com/ZhouGengmo/5b565f51adafcd911c0bc115b2ef027c

import copy

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolAlign as MA
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Geometry.rdGeometry import Point3D


def get_best_rmsd(gen_mol, ref_mol):
    gen_mol = Chem.RemoveHs(gen_mol)
    ref_mol = Chem.RemoveHs(ref_mol)
    rmsd = MA.GetBestRMS(gen_mol, ref_mol)
    return rmsd


def get_cov_mat(ref_mols, gen_mols, threshold=1.25):
    rmsd_mat = np.zeros([len(ref_mols), len(gen_mols)], dtype=np.float32)
    for i, gen_mol in enumerate(gen_mols):
        gen_mol_c = copy.deepcopy(gen_mol)
        for j, ref_mol in enumerate(ref_mols):
            ref_mol_c = copy.deepcopy(ref_mol)
            rmsd_mat[j, i] = get_best_rmsd(gen_mol_c, ref_mol_c)
    rmsd_mat_min = rmsd_mat.min(-1)
    return (rmsd_mat_min <= threshold).mean(), rmsd_mat_min.mean()


def normalise_positions(mol):
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    pos = pos - pos.mean(axis=0)
    for idx, p in enumerate(pos):
        conf.SetAtomPosition(idx, Point3D(*p))
    return mol
