import numpy as np

from gflownet.utils.common import download_file_if_not_exists


class AtomPositionsDataset:
    def __init__(self, smiles: str, path_to_data: str, url_to_data: str):
        path_to_data = download_file_if_not_exists(path_to_data, url_to_data)
        conformers = np.load(path_to_data, allow_pickle=True).item()

        self.positions = conformers[smiles]["conformers"]
        self.torsion_angles = conformers[smiles]["torsion_angles"]

    def __getitem__(self, i):
        return self.positions[i]

    def __len__(self):
        return self.positions.shape[0]

    def sample(self, size=None):
        idx = np.random.randint(0, len(self), size=size)
        return self.positions[idx]

    def first(self):
        return self[0]
