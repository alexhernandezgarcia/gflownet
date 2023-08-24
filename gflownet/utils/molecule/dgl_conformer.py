import torch
from gflownet.utils.molecule import torsions, constants
from gflownet.utils.molecule.featurizer import MolDGLFeaturizer

class DGLConformer:
    def __init__(self, atom_positions, smiles, torsion_indices, atom_types=constants.ad_atom_types, add_hydrogens=True):
        featuriser = MolDGLFeaturizer(atom_types)
        dgl_graph = featuriser.smiles2dgl(smiles, add_hydrogens=add_hydrogens)
        dgl_graph.ndata[constants.atom_position_name] = torch.tensor(atom_positions)
        self.possibly_rotatable_bonds = torsions.get_rotatable_bonds(dgl_graph)
        self.graph = torsions.mask_out_torsion_anlges(dgl_graph, torsion_indices)
        self.rotatable_bonds = torsions.get_rotatable_bonds(self.graph)

    def apply_rotations(self, rotations):
        """
        Apply rotations (torsion angles updates)
        :param rotations: a sequence of torsion angle updates of length = number of bonds in the molecule.
        The order corresponds to the order of edges in self.graph, such that action[i] is
        an update for the torsion angle corresponding to the edge[2i]
        """
        self.graph = torsions.apply_rotations(self.graph, rotations)

    def randomise_torsion_angles(self):
        n_edges = self.graph.edges()[0].shape
        rotations = torch.rand(n_edges // 2) * 2 * torch.pi
        self.apply_rotations(rotations)
