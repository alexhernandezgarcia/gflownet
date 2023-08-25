import torch
import networkx as nx
import numpy as np

from rdkit import Chem
from pytorch3d.transforms import axis_angle_to_matrix
from torch.nn.functional import normalize

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

def get_rotation_masks(dgl_graph, mol=None):
    """
    Creates masks to facilitate updates of the torsion angles in the molecular graph. 
    
    Args
    ----
    dgl_graph : dgl.Graph 
        A single molecular graph with bidirected edges 
        in the order: [e_1_fwd, e_1_bkw, e_2_fwd, e_2_bkw, ...]
    mol (optional) : rdkit.Chem.rdchem.Mol 
        A molecule in the RDKit formal, used for masking out non-single bounds
    
    Returns
    -------
    edges_mask : boolean torch.Tensor of shape [n_edges,], each of the two directed edges 
        makred as True if corresponding bond is rotatable (see convention above) and as False otherwise
    nodes_mask : boolean torch.Tensor of shape [n_edges, n_atoms] which is True for the atoms affected by the rotation. 
        For each rotatable bond we select the smallest part of the graph to be affected (it can be either "beginning side" 
        which contains begining of the forward edge or "end side" which contains the end of the forward edge) 
    rotation signs : float torch.Tensor of 0., 1., -1. defining the multiplicative factor 
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
            if mol is not None:
                single_bond = mol.GetBondBetweenAtoms(int(bond[0]),int(bond[1])).GetBondType() == Chem.rdchem.BondType.SINGLE
            else:
                single_bond = True
            if len(smallest_component_nodes) > 1 and single_bond:
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

def mask_out_torsion_anlges(graph, torsion_indices):
    """
    Adjusts rotation masks in the graph, such that only angles with indecies 
    torsion_indices are kept rotatable.

    Args
    ----
    graph : dgl.Graph
        A graph with rotation masks in graph.edata
    torsion_indices : list of ints
        Indecies of the rotatable bonds which will be kept rotatable. 
    """
    rotatable_edges_indecies = graph.edata[constants.rotatable_edges_mask_name].nonzero().flatten()
    meta_indecies = set(range(len(rotatable_edges_indecies)))
    meta_indecies_to_keep =  set([item for x in torsion_indices for item in (2 * x, 2* x + 1)])
    meta_indecies_to_fix = torch.tensor(sorted(meta_indecies - meta_indecies_to_keep))
    if len(meta_indecies_to_fix) > 0:
        indecies_to_fix = rotatable_edges_indecies[meta_indecies_to_fix] 
        graph.edata[constants.rotatable_edges_mask_name][indecies_to_fix] = False
        n_atoms = graph.edata[constants.rotation_affected_nodes_mask_name].shape[1]
        graph.edata[constants.rotation_affected_nodes_mask_name][indecies_to_fix] = torch.zeros(n_atoms, 
                                                                                                 dtype=torch.bool)
        graph.edata[constants.rotation_signs_name][indecies_to_fix] = 0 
    return graph

def get_rotatable_bonds(graph):
    rot_idx = graph.edata[constants.rotatable_edges_mask_name].nonzero().flatten()[::2]
    edges = torch.stack(graph.edges()).T
    return edges[rot_idx]

def get_torsion_angles(graph):
    def get_smallest_neighbour(node, neighbour_to_exclude):
        neighbours = set(graph.predecessors(node).tolist()) - set([neighbour_to_exclude])
        return min(neighbours)
    edges = get_rotatable_bonds(graph)
    torsion_angles = []
    for edge in edges:
        begin = edge[0].item()
        end = edge[1].item()
        torsion_angles.append((get_smallest_neighbour(begin, end), begin, end, get_smallest_neighbour(end, begin)))
    return torch.tensor(torsion_angles)

def compute_torsion_angles(graph, torsion_angles, epsilon=1e-9):
    """
    torsion_angle : tuple of 4 integers
    """
    #import ipdb; ipdb.set_trace()
    pos = graph.ndata[constants.atom_position_name]
    b_1 = pos[torsion_angles[:,1]] - pos[torsion_angles[:,0]]
    b_2 = pos[torsion_angles[:,2]] - pos[torsion_angles[:,1]]
    b_3 = pos[torsion_angles[:,3]] - pos[torsion_angles[:,2]]
    n_1 = normalize(torch.cross(b_1, b_2, dim=1), dim=1)
    n_2 = normalize(torch.cross(b_2, b_3, dim=1), dim=1)
    m_1 = torch.cross(n_1, normalize(b_2, dim=1) , dim=1)
    x = torch.sum(n_1 * n_2, axis=1)
    y = torch.sum(m_1 * n_2, axis=1)
    return -torch.atan2(y, x) 

