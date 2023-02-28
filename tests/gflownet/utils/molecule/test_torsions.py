import pytest
import torch
import dgl

from gflownet.utils.molecule.torsions import get_rotation_masks

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