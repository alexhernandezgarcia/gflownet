import dgl
import numpy as np

from gflownet.utils.common import download_file_if_not_exists
from gflownet.utils.molecule import constants
from gflownet.utils.molecule.dgl_conformer import DGLConformer

class AtomPositionsDataset:
    def __init__(self, path_to_data, url_to_data):
        path_to_data = download_file_if_not_exists(path_to_data, url_to_data)
        self.positions = np.load(path_to_data)

    def __getitem__(self, i):
        return self.positions[i]

    def __len__(self):
        return self.positions.shape[0]

    def sample(self, size=None):
        idx = np.random.randint(0, len(self), size=size)
        return self.positions[idx]

class ConformersDataset:
    def __init__(self, path_to_data, url_to_data):
        # TODO create a new dataset if path_to_data or url_to_data doesn't exist
        path_to_data = download_file_if_not_exists(path_to_data, url_to_data)
        with open(path_to_data, 'rb') as inp:
            self.conformers = pickle.load(inp)
        
    def get_conformer(self):
        """
        Returns dgl graph with features stored in the dataset:
          - ndata: 
            - atom features
            - atomic numbers
            - atom position
          - edata:
            - edge features
            - rotatable bonds mask
        """
        smiles = np.random.choice(self.conformers.keys())
        edges = self.conformers[smiles]['edges']
        graph = dgl.graph(edges)
        graph.ndata[constants.atom_feature_name] = self.conformers[smiles][constants.atom_feature_name]
        graph.ndata[constants.atomic_numbers_name] = self.conformers[smiles][constants.atomic_numbers_name]
        graph.edata[constants.edge_feature_name] = self.conformers[smiles][constants.edge_feature_name]
        graph.edata[constants.rotatable_bonds_mask] = self.conformers[smiles][constants.rotatable_bonds_mask]
        conf_idx = np.random.randint(0, self.conformers[smiles][constants.atom_position_name].shape[0])
        graph.ndata[constants.atom_position_name] = self.conformers[smiles][constants.atom_position_name][conf_idx]
        conformer = DGLConformer(graph)
        return smiles, conformer