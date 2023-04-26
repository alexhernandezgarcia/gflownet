import torch


class DGLConformer:
    def __init__(self, dgl_graph):
        self.graph = dgl_graph

    def apply_rotations(self, rotations):
        """
        Apply rotations (torsion angles updates)
        :param rotations: a sequence of torsion angle updates of length = number of bonds in the molecule.
        The order corresponds to the order of edges in self.graph, such that action[i] is
        an update for the torsion angle corresponding to the edge[2i]
        """
        raise NotImplementedError

    def randomise_torsion_angles(self):
        raise NotImplementedError
