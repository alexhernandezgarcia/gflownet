import pytest
import torch
import dgl

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry.rdGeometry import Point3D

from gflownet.utils.molecule.torsions import get_rotation_masks, apply_rotations
from gflownet.utils.molecule import constants
from gflownet.utils.molecule.featurizer import MolDGLFeaturizer
from gflownet.utils.molecule.rdkit_conformer import get_torsion_angles_values

def test_four_nodes_chain():
    graph = dgl.graph(([0,1,1,2,2,3], [1,0,2,1,3,2]))
    edges_mask, nodes_mask, rotation_signs = get_rotation_masks(graph)
    correct_edges_mask = torch.tensor([False, False,  True,  True, False, False])
    correct_nodes_mask = torch.tensor([[False, False, False, False],
                                       [False, False, False, False],
                                       [ True, False, False, False],
                                       [ True, False, False, False],
                                       [False, False, False, False],
                                       [False, False, False, False]])
    correct_rotation_signs = torch.tensor([ 0.,  0., -1., -1.,  0.,  0.])
    assert torch.all(edges_mask == correct_edges_mask)
    assert torch.all(nodes_mask == correct_nodes_mask)
    assert torch.all(rotation_signs == correct_rotation_signs)

def test_choose_smallest_component():
    graph = dgl.graph(([0, 2, 1, 2, 2, 3, 3, 4], [2, 0, 2, 1, 3, 2, 4, 3]))
    edges_mask, nodes_mask, rotation_signs = get_rotation_masks(graph)
    correct_edges_mask = torch.tensor([False, False, False, False,  True,  True, False, False])
    correct_nodes_mask = torch.tensor([[False, False, False, False, False],
                                       [False, False, False, False, False],
                                       [False, False, False, False, False],
                                       [False, False, False, False, False],
                                       [False, False, False, False,  True],
                                       [False, False, False, False,  True],
                                       [False, False, False, False, False],
                                       [False, False, False, False, False]])
    correct_rotation_signs = torch.tensor([0., 0., 0., 0., 1., 1., 0., 0.]) 
    assert torch.all(edges_mask == correct_edges_mask)
    assert torch.all(nodes_mask == correct_nodes_mask)
    assert torch.all(rotation_signs == correct_rotation_signs)

@pytest.mark.parametrize(
    "angle, exp_result",
    [
        (
            torch.pi / 2,
            torch.tensor(
                [[1., 0., -1.],                                                         
                 [1., 0., 0.],                                                         
                 [1., 1., 0.],                                                         
                 [2., 1., 0.]])
        ),
        (
            torch.pi,
            torch.tensor(
                [[2., 0., 0.],                                                         
                 [1., 0., 0.],                                                         
                 [1., 1., 0.],                                                         
                 [2., 1., 0.]])
        ),
        (
            torch.pi * 3 / 2,
            torch.tensor(
                [[1., 0., 1.],                                                         
                 [1., 0., 0.],                                                         
                 [1., 1., 0.],                                                         
                 [2., 1., 0.]])
        ),
        (
            torch.pi * 2,
            torch.tensor(
                [[0., 0., 0.],                                                         
                 [1., 0., 0.],                                                         
                 [1., 1., 0.],                                                         
                 [2., 1., 0.]])
        ),

    ]

)
def test_apply_rotations_simple(angle, exp_result):
    graph = dgl.graph(([0,1,1,2,2,3], [1,0,2,1,3,2]))
    graph.ndata[constants.atom_position_name] = torch.tensor([
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [2., 1., 0.]
    ])
    edges_mask, nodes_mask, rotation_signs = get_rotation_masks(graph)
    graph.edata[constants.rotatable_edges_mask_name] = edges_mask
    graph.edata[constants.rotation_affected_nodes_mask_name] = nodes_mask
    graph.edata[constants.rotation_signs_name] = rotation_signs
    rotations = torch.tensor([0., angle, 0.])
    result = apply_rotations(graph, rotations).ndata[constants.atom_position_name]
    assert torch.allclose(result, exp_result,  atol=1e-6)


@pytest.mark.parametrize(
    "angle, exp_result",
    [
        (
            torch.pi / 2,
            torch.tensor(
                [[1., 0., -1.],                                                         
                 [1., 0., 0.],                                                         
                 [1., 1., 0.],                                                         
                 [2., 1., 0.]])
        ),
        (
            torch.pi,
            torch.tensor(
                [[2., 0., 0.],                                                         
                 [1., 0., 0.],                                                         
                 [1., 1., 0.],                                                         
                 [2., 1., 0.]])
        ),
        (
            torch.pi * 3 / 2,
            torch.tensor(
                [[1., 0., 1.],                                                         
                 [1., 0., 0.],                                                         
                 [1., 1., 0.],                                                         
                 [2., 1., 0.]])
        ),
        (
            torch.pi * 2,
            torch.tensor(
                [[0., 0., 0.],                                                         
                 [1., 0., 0.],                                                         
                 [1., 1., 0.],                                                         
                 [2., 1., 0.]])
        ),

    ]

)
def test_apply_rotations_ignore_nonrotatable(angle, exp_result):
    graph = dgl.graph(([0,1,1,2,2,3], [1,0,2,1,3,2]))
    graph.ndata[constants.atom_position_name] = torch.tensor([
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [2., 1., 0.]
    ])
    edges_mask, nodes_mask, rotation_signs = get_rotation_masks(graph)
    graph.edata[constants.rotatable_edges_mask_name] = edges_mask
    graph.edata[constants.rotation_affected_nodes_mask_name] = nodes_mask
    graph.edata[constants.rotation_signs_name] = rotation_signs
    rotations = torch.tensor([2., angle, -1.])
    result = apply_rotations(graph, rotations).ndata[constants.atom_position_name]
    assert torch.allclose(result, exp_result,  atol=1e-6)

def stress_test_apply_rotation_alanine_dipeptide():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry.rdGeometry import Point3D
    from gflownet.utils.molecule.featurizer import MolDGLFeaturizer
    from gflownet.utils.molecule.rdkit_conformer import get_torsion_angles_values

    mol = Chem.MolFromSmiles(constants.ad_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    rconf = mol.GetConformer()
    start_pos = rconf.GetPositions()

    featurizer = MolDGLFeaturizer(constants.ad_atom_types)
    graph = featurizer.mol2dgl(mol)
    graph.ndata[constants.atom_position_name] = torch.from_numpy(start_pos)

    torsion_angles = [
        (10, 0, 1, 6),
        (0, 1, 2, 3),
        (1, 2, 4, 14),
        (2, 4, 5, 15),
        (0, 1, 6, 7),
        (18, 6, 7, 8),
        (8, 7, 9, 19)
    ]
    n_edges = graph.edges()[0].shape[-1]
    for _ in range (100):
        ta_initial_values = torch.tensor(get_torsion_angles_values(rconf, torsion_angles))
        
        rotations = torch.rand(n_edges // 2) * torch.pi * 2
        graph = apply_rotations(graph, rotations)
        new_pos = graph.ndata[constants.atom_position_name].numpy()
        for idx, pos in enumerate(new_pos):
            rconf.SetAtomPosition(idx, Point3D(*pos))
        ta_updated_values = torch.tensor(get_torsion_angles_values(rconf, torsion_angles))
        valid_rotations = rotations[graph.edata[constants.rotatable_edges_mask_name][::2]]
        diff = (ta_updated_values - ta_initial_values - valid_rotations) % (2*torch.pi)
        assert torch.logical_or(torch.isclose(diff, torch.zeros_like(diff), atol=1e-6), torch.isclose(diff, torch.ones_like(diff)*2*torch.pi, atol=1e-5)).all()