import torch
import networkx as nx
import numpy as np

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
