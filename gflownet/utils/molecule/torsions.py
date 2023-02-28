import torch
import networkx as nx
import numpy as np

from pytorch3d.transforms import axis_angle_to_matrix

from gflownet.utils.molecule import constants

def get_rotation_masks(dgl_graph):
    """
    :param dgl_graph: the dgl.Graph object with bidirected edges in the order: [e_1_fwd, e_1_bkw, e_2_fwd, e_2_bkw, ...] 
    """
    nx_graph = nx.DiGraph(dgl_graph.to_networkx())
    # bonds are indirected edges
    bonds = torch.stack(dgl_graph.edges()).numpy().T[::2]
    bonds_mask = np.zeros(bonds.shape[0], dtype=bool)
    nodes_mask = np.zeros((bonds.shape[0], dgl_graph.num_nodes()), dtype=bool)
    # fill in masks for bonds 
    for bond_idx, bond in enumerate(bonds):
        modified_graph = nx_graph.to_undirected()
        modified_graph.remove_edge(*bond)
        if not nx.is_connected(modified_graph):
            smallest_component_nodes = sorted(nx.connected_components(modified_graph), key=len)[0]
            if len(smallest_component_nodes) > 1:
                bonds_mask[bond_idx] = True
                affected_nodes = np.array(list(smallest_component_nodes - set(bond)))
                nodes_mask[bond_idx, affected_nodes] = np.ones_like(affected_nodes, dtype=bool)
    # broadcast bond masks to edges masks
    edges_mask = torch.from_numpy(bonds_mask.repeat(2))
    nodes_mask = torch.from_numpy(nodes_mask.repeat(2, axis=0))
    return edges_mask, nodes_mask

def apply_rotations(dgl_graph, rotations):
    """
    Apply rotations (torsion angles updates)
    :param dgl_graph: bidirectional dgl.Graph
    :param rotations: a sequence of torsion angle updates of length = number of bonds in the molecule.
    The order corresponds to the order of edges in the graph, such that action[i] is
    an update for the torsion angle corresponding to the edge[2i]
    """
    pos = graph.ndata[constants.atom_position_name]
    edge_mask = graph.edata[constants.rotatable_edges_mask_name]
    node_mask = graph.edata[constants.rotation_affected_nodes_mask_name]
    edges = torch.stack(graph.edges()).T
    # TODO check how slow it is and whether it's possible to vectorise this loop
    for idx_update, update in enumerate(rotations):
        idx_edge = idx_update * 2
        if edge_mask[idx_edge]:
            begin_pos = pos[edges[idx_edge][0]]
            end_pos = pos[edges[idx_edge][1]]
            rot_vector = end_pos - begin_pos
            rot_vector = rot_vector / torch.linalg.norm(rot_vector) * update
            rot_matrix = axis_angle_to_matrix(rot_vector)
            x = pos[node_mask[idx_edge]]
            pos[node_mask[idx_edge]] = torch.matmul((x - begin_pos), rot_matrix.T) + begin_pos
    dgl_graph.ndata[constants.atom_position_name] = pos
    return dgl_graph



