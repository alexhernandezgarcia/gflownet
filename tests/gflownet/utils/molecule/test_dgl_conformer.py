import pytest
import torch
import dgl

from gflownet.utils.molecule import constants
from gflownet.utils.molecule.dgl_conformer import DGLConformer
from gflownet.utils.molecule.rdkit_utils import get_rdkit_atom_positions

def test_simple_conformer():
    atom_positions = get_rdkit_atom_positions(constants.ad_smiles, extra_opt=False, add_hydrogens=False)
    conformer = DGLConformer(atom_positions, constants.ad_smiles, torsion_indices=[0, 1, 2, 3], add_hydrogens=False)
    import ipdb; ipdb.set_trace()
    