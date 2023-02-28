import torch

class DGLConformer:
    def __init__(self, dgl_graph):
        self.graph = dgl_graph

    def increment_torsion_angles(self, increments):
        raise NotImplementedError

    def randomise_torsion_angles(self):
        raise NotImplementedError

    