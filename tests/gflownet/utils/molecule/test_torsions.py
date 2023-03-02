import pytest
import torch
import dgl

from gflownet.utils.molecule.torsions import get_rotation_masks, apply_rotations
from gflownet.utils.molecule import constants

def test_four_nodes_chain():
    graph = dgl.graph(([0,1,1,2,2,3], [1,0,2,1,3,2]))
    edges_mask, nodes_mask = get_rotation_masks(graph)
    correct_edges_mask = torch.tensor([False, False,  True,  True, False, False])
    correct_nodes_mask = torch.tensor([[False, False, False, False],
                                       [False, False, False, False],
                                       [ True, False, False, False],
                                       [ True, False, False, False],
                                       [False, False, False, False],
                                       [False, False, False, False]])
    assert torch.all(edges_mask == correct_edges_mask)
    assert torch.all(nodes_mask == correct_nodes_mask)

def test_choose_smallest_component():
    graph = dgl.graph(([0, 2, 1, 2, 2, 3, 3, 4], [2, 0, 2, 1, 3, 2, 4, 3]))
    edges_mask, nodes_mask = get_rotation_masks(graph)
    correct_edges_mask = torch.tensor([False, False, False, False,  True,  True, False, False])
    correct_nodes_mask = torch.tensor([[False, False, False, False, False],
                                       [False, False, False, False, False],
                                       [False, False, False, False, False],
                                       [False, False, False, False, False],
                                       [False, False, False, False,  True],
                                       [False, False, False, False,  True],
                                       [False, False, False, False, False],
                                       [False, False, False, False, False]])
    assert torch.all(edges_mask == correct_edges_mask)
    assert torch.all(nodes_mask == correct_nodes_mask)

@pytest.mark.parametrize(
    "angle, exp_result",
    [
        (
            torch.pi / 2,
            torch.tensor(
                [[1., 0., 1.],                                                         
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
                [[1., 0., -1.],                                                         
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
    edges_mask, nodes_mask = get_rotation_masks(graph)
    graph.edata[constants.rotatable_edges_mask_name] = edges_mask
    graph.edata[constants.rotation_affected_nodes_mask_name] = nodes_mask
    rotations = torch.tensor([0., angle, 0.])
    result = apply_rotations(graph, rotations).ndata[constants.atom_position_name]
    assert torch.allclose(result, exp_result,  atol=1e-6)


@pytest.mark.parametrize(
    "angle, exp_result",
    [
        (
            torch.pi / 2,
            torch.tensor(
                [[1., 0., 1.],                                                         
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
                [[1., 0., -1.],                                                         
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
    edges_mask, nodes_mask = get_rotation_masks(graph)
    graph.edata[constants.rotatable_edges_mask_name] = edges_mask
    graph.edata[constants.rotation_affected_nodes_mask_name] = nodes_mask
    rotations = torch.tensor([2., angle, -1.])
    result = apply_rotations(graph, rotations).ndata[constants.atom_position_name]
    assert torch.allclose(result, exp_result,  atol=1e-6)

# def test_apply_rotation_alanine_dipeptide():
#     from rdkit import Chem

#     mol = Chem.MolFromSmiles(constants.ad_smiles)
#     mol = Chem.AddHs(mol)
#     AllChem.EmbedMolecule(rmol)
#     rconf = rmol.GetConformer()
#     start_pos = rconf.GetPositions()

#     featurizer = MolDGLFeaturizer(constants.ad_atom_types)

#     graph = featurizer.mol2dgl(mol)
#     graph.ndata[constants.atom_position_name] = torch.from_numpy(start_pos)

    
#     rmol = Chem.MolFromSmiles(constants.ad_smiles)
#     rmol = Chem.AddHs(rmol)
#     AllChem.EmbedMolecule(rmol)
#     rconf = rmol.GetConformer()
#     test_pos = rconf.GetPositions()
#     initial_tas = get_all_torsion_angles(rmol, rconf)

#     conf = RDKitConformer(
#         test_pos, constants.ad_smiles, constants.ad_atom_types, constants.ad_free_tas
#     )