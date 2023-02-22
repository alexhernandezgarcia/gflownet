import numpy as np

from gflownet.utils.common import download_file_if_not_exists

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
