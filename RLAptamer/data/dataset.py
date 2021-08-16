import os

from torch.utils.data import Dataset

path = "/datasets/aptamer"


class Aptamer_Dataset(Dataset):
    def __init__(
        self, mode: str, islabelled: bool, data_path: str = ""
    ):
        self.islabelled = islabelled
        self.root = data_path + path
        self.mode = mode
        self.sequences = make_dataset(self.root)
        if len(self.imgs) == 0:
            raise RuntimeError("Found 0 sequences, please check the data set")

    #TODO Write a make_dataset method to
    def make_dataset(self):
        return self.sequences

    def add_datapoint(self)

    def __getitem__(self, index):
        return self.sequences[index]

    def __len__(self):
        return len(self.sequences)
