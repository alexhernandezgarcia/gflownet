import torch

from gflownet.utils.common import tfloat, tfloat_graph
from gflownet.utils.molecule import torsions, constants
from gflownet.utils.molecule.featurizer import MolDGLFeaturizer

class DGLConformer:
    def __init__(self, atom_positions, smiles, torsion_indices=None, atom_types=constants.ad_atom_types, 
                 add_hydrogens=True, float_type=torch.float32, device=torch.device('cpu')):
        self.float_type = float_type
        self.device = device
        featuriser = MolDGLFeaturizer(atom_types)
        dgl_graph = featuriser.smiles2dgl(smiles, add_hydrogens=add_hydrogens)
        dgl_graph.ndata[constants.atom_position_name] = torch.tensor(atom_positions)
        self.possibly_rotatable_bonds = torsions.get_rotatable_bonds(dgl_graph)
        self.possibly_rotatatble_torsion_angles = torsions.get_rotatable_torsion_angles_names(dgl_graph)
        self.graph = tfloat_graph(dgl_graph, self.device, self.float_type) 
        if torsion_indices is not None:
            self.graph = tfloat_graph(torsions.mask_out_torsion_anlges(dgl_graph, torsion_indices), self.device, self.float_type)
        self.n_edges = len(self.graph.edges()[0])
        self.n_nodes = len(self.graph.nodes())
        self.rotatable_bonds = torsions.get_rotatable_bonds(self.graph)
        self.rotatable_torsion_angles = torsions.get_rotatable_torsion_angles_names(self.graph)

    def apply_rotations(self, rotations):
        """
        Applies rotations (torsion angles updates) to the rotatable torsion angles

        Args
        ----
        rotations : 1D torch tensor 
            A sequence of torsion angle updates of length = number of rotatable torsion angles in this conformer.
            The order corresponds to the order in self.rotatable_torsion_angles
        """
        all_rotations = torch.zeros(self.n_edges // 2, dtype=self.float_type, device=self.device)
        rot_idx = torch.arange(self.n_edges // 2, device=self.device)[self.graph.edata[constants.rotatable_edges_mask_name][::2]]
        all_rotations[rot_idx] = rotations
        self.graph = torsions.apply_rotations(self.graph, all_rotations)

    def randomise_torsion_angles(self):
        """
        Randomizes rotatable torsion angles such that they are sampled from uniform distribution over torus
        """
        rotations = torch.rand(len(self.rotatable_bonds), dtype=self.float_type, device=self.device) * 2 * torch.pi
        self.apply_rotations(rotations)

    def compute_rotatable_torsion_angles(self):
        """
        Computes rotatable torsion angles, values are in [-pi, pi]

        Returns
        -------
            A 1D torch tensor of float values of rotatable torsion angles in radians.
        """
        return tfloat(torsions.compute_torsion_angles(self.graph, self.rotatable_torsion_angles), 
                      device=self.device, float_type=self.float_type)
    
    def set_rotatable_torsion_angles(self, values):
        """
        Sets rotatable torsion angles to the specified values

        Args
        ----
        values : 1D torch float tensor
            Values in radians to assign to the rotatable torsion angles
        """
        current_values = self.compute_rotatable_torsion_angles()
        update = values - current_values
        self.apply_rotations(update)
    
    def get_atom_positions(self):
        return self.graph.ndata[constants.atom_position_name]