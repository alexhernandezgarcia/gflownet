import torch
import networkx as nx
import numpy as np

from pytorch3d.transforms import axis_angle_to_matrix

from gflownet.utils.molecule import constants

"""
CONVENTIONS:
- In a dgl graph, edges are always directed, so each bond in the molecule gives rise to two directed edges, one is a reversed copy of another.
    These edges go one after another in the edges of a dgl graph. We call first edge "forward edge" and 
    second edge "backward edge". In the forward edge, the first node of the edge (edge begining) has 
    a smaller index than the second node (edge end). In the backward edge, this order is reversed
- A bond in a molecule is rotatable if:
    - it is a bridge in the graph, i.e. if we romewe it, the molecule graph with be 
        separated into two disconnected components. This is done to exclude torsion angles in cycles
    - and it is not adjacent to a leaf node (the node wich has only one bond). That's because there're no torsion angles 
        corresponding to leaf-adjacent bonds
- Rotation vector corresponding to a bond is directed from the node with a smaller index to the node with a larger index
    (from beginning of the forward edge to tthe end of the forward edge)
- We use right-hand rule for rotations, i.e. rotation appears counterclockwise when the rotation vector points toward the observer.
    For more details: https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions 
"""

def get_rotation_masks(dgl_graph):
    """
    Creates masks to facilitate updates of the torsion angles in the molecular graph. 
    :param dgl_graph: the dgl.Graph object corresponding to a single molecular graph 
        with bidirected edges in the order: [e_1_fwd, e_1_bkw, e_2_fwd, e_2_bkw, ...]
    
    Returns three masks:
    - edges_mask: boolean torch.Tensor of shape [n_edges,], each of the two directed edges 
        makred as True if corresponding bond is rotatable (see convention above) and as False otherwise
    - nodes_mask: boolean torch.Tensor of shape [n_edges, n_atoms] which is True for the atoms affected by the rotation. 
        For each rotatable bond we select the smallest part of the graph to be affected (it can be either "beginning side" 
        which contains begining of the forward edge or "end side" which contains the end of the forward edge) 
    - rotation signs: float torch.Tensor of 0., 1., -1. defining the multiplicative factor 
        for applying rotational matrix to the atom positions. It is 
        - 0. if there's no rotation, 
        - 1. if rotation is applied to the end side
        - -1. if rotation is applied to the beginning side 
    """
    nx_graph = nx.DiGraph(dgl_graph.to_networkx())
    # bonds are undirected edges
    bonds = torch.stack(dgl_graph.edges()).numpy().T[::2]
    bonds_mask = np.zeros(bonds.shape[0], dtype=bool)
    nodes_mask = np.zeros((bonds.shape[0], dgl_graph.num_nodes()), dtype=bool)
    rotation_signs = np.zeros(bonds.shape[0], dtype=float)
    # fill in masks for bonds
    for bond_idx, bond in enumerate(bonds):
        modified_graph = nx_graph.to_undirected()
        modified_graph.remove_edge(*bond)
        if not nx.is_connected(modified_graph):
            smallest_component_nodes = sorted(
                nx.connected_components(modified_graph), key=len
            )[0]
            if len(smallest_component_nodes) > 1:
                bonds_mask[bond_idx] = True
                rotation_signs[bond_idx] = (
                    -1 if bond[0] in smallest_component_nodes else 1
                )
                affected_nodes = np.array(list(smallest_component_nodes - set(bond)))
                nodes_mask[bond_idx, affected_nodes] = np.ones_like(
                    affected_nodes, dtype=bool
                )

    # broadcast bond masks to edges masks
    edges_mask = torch.from_numpy(bonds_mask.repeat(2))
    rotation_signs = torch.from_numpy(rotation_signs.repeat(2))
    nodes_mask = torch.from_numpy(nodes_mask.repeat(2, axis=0))
    return edges_mask, nodes_mask, rotation_signs


def apply_rotations(graph, rotations):
    """
    Apply rotations (torsion angles updates)

    Args
    ----
    dgl_graph : bidirectional dgl.Graph
    rotations : a sequence of torsion angle updates of length = number of bonds in the molecule. 
        The order corresponds to the order of edges in the graph, such that action[i] is
        an update for the torsion angle corresponding to the edge[2i]

    Returns
    -------
    updated dgl graph
    """
    pos = graph.ndata[constants.atom_position_name]
    edge_mask = graph.edata[constants.rotatable_edges_mask_name]
    node_mask = graph.edata[constants.rotation_affected_nodes_mask_name]
    rot_signs = graph.edata[constants.rotation_signs_name]
    edges = torch.stack(graph.edges()).T
    # TODO check how slow it is and whether it's possible to vectorise this loop
    for idx_update, update in enumerate(rotations):
        idx_edge = idx_update * 2
        if edge_mask[idx_edge]:
            begin_pos = pos[edges[idx_edge][0]]
            end_pos = pos[edges[idx_edge][1]]
            rot_vector = end_pos - begin_pos
            rot_vector = (
                rot_vector
                / torch.linalg.norm(rot_vector)
                * update
                * rot_signs[idx_edge]
            )
            rot_matrix = axis_angle_to_matrix(rot_vector)
            x = pos[node_mask[idx_edge]]
            pos[node_mask[idx_edge]] = (
                torch.matmul((x - begin_pos), rot_matrix.T) + begin_pos
            )
    graph.ndata[constants.atom_position_name] = pos
    return graph
