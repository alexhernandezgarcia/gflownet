import pytest
import torch
import dgl
import numpy as np

from gflownet.utils.molecule import constants
from gflownet.utils.molecule.dgl_conformer import DGLConformer
from gflownet.utils.molecule.rdkit_utils import (get_rdkit_atom_positions, get_rdkit_conformer, 
                                                 get_rdkit_torsion_angles_values, set_rdkit_atom_positions)
from gflownet.utils.molecule import torsions

@pytest.mark.parametrize(
        "torsion_indices, expected_n_rotatable_ta, add_hydrogens",
        [
            (
                [0,1,2,3],
                4,
                False
            ),
            (
                None,
                4,
                False
            ),
            (
                [0,1,2,3],
                4,
                True
            ),
            (
                None,
                7,
                True
            ),
            (
                [1,2],
                2,
                False
            ),
            (
                [1,2],
                2,
                True
            ),
        ]
)

def test_n_rotatale_tas(torsion_indices, expected_n_rotatable_ta, add_hydrogens):
    atom_positions = get_rdkit_atom_positions(constants.ad_smiles, extra_opt=False, add_hydrogens=add_hydrogens)
    conformer = DGLConformer(atom_positions, constants.ad_smiles, torsion_indices=torsion_indices, add_hydrogens=add_hydrogens)
    assert conformer.rotatable_torsion_angles.shape[0] == expected_n_rotatable_ta

@pytest.mark.parametrize(
        "float_type, device",
        [
            (
                torch.float32, 
                torch.device("cpu")
            ),
            (
                torch.float64,
                torch.device("cpu")
            ),
            (
                torch.float32, 
                torch.device("cuda:0")
            ),
            (
                torch.float64,
                torch.device("cuda:0")
            ),
        ]
)
def test_simple_compute_ta(float_type, device):
    rdk_conf = get_rdkit_conformer(constants.ad_smiles, add_hydrogens=True)
    positions = rdk_conf.GetPositions()
    conformer = DGLConformer(positions, constants.ad_smiles, torsion_indices=None, add_hydrogens=True, 
                             float_type=float_type, device=device)
    ta_values = conformer.compute_rotatable_torsion_angles()
    ta_names = conformer.rotatable_torsion_angles.tolist()
    expected = torch.tensor(get_rdkit_torsion_angles_values(rdk_conf, ta_names), 
                            dtype=float_type).to(device)
    assert torch.all(torch.isclose(ta_values, expected, atol=1e-6))

@pytest.mark.parametrize(
        "float_type, device",
        [
            (
                torch.float32, 
                torch.device("cpu")
            ),
            (
                torch.float64,
                torch.device("cpu")
            ),
            (
                torch.float32, 
                torch.device("cuda:0")
            ),
            (
                torch.float64,
                torch.device("cuda:0")
            ),
        ]
)
def test_stress_compute_ta(float_type, device):
    rdk_conf = get_rdkit_conformer(constants.ad_smiles, add_hydrogens=True)
    positions = rdk_conf.GetPositions()
    conformer = DGLConformer(positions, constants.ad_smiles, torsion_indices=None, add_hydrogens=True, 
                             float_type=float_type, device=device)
    for _ in range(100):
        ta_values = conformer.compute_rotatable_torsion_angles()
        ta_names = conformer.rotatable_torsion_angles.tolist()
        expected = torch.tensor(get_rdkit_torsion_angles_values(rdk_conf, ta_names), 
                            dtype=float_type).to(device)
        assert torch.all(torch.isclose(ta_values, expected, atol=1e-6))
        conformer.randomise_torsion_angles()
        new_pos = conformer.get_atom_positions()
        rdk_conf = set_rdkit_atom_positions(rdk_conf, np.array(new_pos.tolist()))

@pytest.mark.parametrize(
        "float_type, device",
        [
            (
                torch.float32, 
                torch.device("cpu")
            ),
            (
                torch.float64,
                torch.device("cpu")
            ),
            (
                torch.float32, 
                torch.device("cuda:0")
            ),
            (
                torch.float64,
                torch.device("cuda:0")
            ),
        ]
)
def test_stress_set_ta(float_type, device):
    rdk_conf = get_rdkit_conformer(constants.ad_smiles, add_hydrogens=True)
    positions = rdk_conf.GetPositions()
    conformer = DGLConformer(positions, constants.ad_smiles, torsion_indices=None, add_hydrogens=True, 
                             float_type=float_type, device=device)

    for _ in range(100):
        expected = torch.rand(conformer.n_rotatable_bonds, dtype=float_type, device=device) * 2 * torch.pi - torch.pi 
        conformer.set_rotatable_torsion_angles(expected)
        ta_values = conformer.compute_rotatable_torsion_angles()
        assert torch.all(torch.isclose(ta_values, expected, atol=1e-6))