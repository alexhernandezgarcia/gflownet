import dgl
import torch

from gflownet.utils.molecule import constants


class MolDGLFeaturizer:
    """A class for converting RDKit molecule into DGL graph with featurizing chemical properties into node and edge features"""

    def __init__(self, atom_types):
        self.atom_types = atom_types
        self.atom_degrees = constants.atom_degrees
        self.atom_hybridizations = constants.atom_hybridizations
        self.bond_types = constants.bond_types

    def one_hot_encode(self, value, choices):
        """
        Creates a one-hot encoding vector.
        :param value: The value for which the encoding should be one
        :param choices: A list of possible values
        :return: A one-hot encoding of the value as a 1d torch float tensor of length len(choices)
        """
        encoding = [0] * len(choices)
        index = choices.index(value)
        encoding[index] = 1

        return torch.tensor(encoding, dtype=torch.float)

    def get_node_features(self, mol):
        """
        Simple atom featurization, considered features:
        - one-hot of the atom type index from self.atom_types tuple
        - atomic number (as a float number)
        - one-hot of the atom degree
        - one-hot of the atom hybritization

        Other atom features to add in the fututre:
            .GetIsAromatic(), .GetImplicitValence(), .GetFormalCharge(),
            .IsAtomInRingOfSize(...) with various sizes, some other info about rings (see torsinal diff code)
            and maybe some others from Chenghao's code

        :param mol: The rdkit.Chem.rdchem.Mol object
        :returns: a torch.Tensor of node features of shape [number of atoms, node feature size]
        (ordering of the nodes is gived by rdkit atoms order in mol)
        """
        features = []
        for i, atom in enumerate(mol.GetAtoms()):
            atom_features = []
            # one-hot atom index
            atom_features.append(self.one_hot_encode(atom.GetSymbol(), self.atom_types))
            # atomic number (as it is, not one-hot)
            atom_features.append(torch.tensor([atom.GetAtomicNum()], dtype=torch.float))
            # one-hot atom degree (number of adjacent edges)
            atom_features.append(
                self.one_hot_encode(atom.GetDegree(), self.atom_degrees)
            )
            # one-hot hybridization
            atom_features.append(
                self.one_hot_encode(atom.GetHybridization(), self.atom_hybridizations)
            )

            features.append(torch.cat(atom_features, dim=0))
        return torch.stack(features)

    def get_edges_and_edge_features(self, mol):
        """
        Simple edge extraction and featurisation, considered features:
        - one-hot od the bound type
        Edges are directional (because of the dgl framework), each bond in the mol gives rise to two directional edges,
        which goes one after another in the output
        :param mol: The rdkit.Chem.rdchem.Mol object
        :returns: edges and edge_features, where
            - edges is a tuple of two lists (source nodes and destination nodes) of length 2 * number of bonds in mol
            - edge_features is a torch.Tensor of considered edge features (shape [2 * number of bonds, edge feature size])
        """
        sources = []
        destinations = []
        edge_features = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            # add both the edge and its reversed (revirsed after the forward edge, that's important for the adgent's code!)
            sources.extend([start, end])
            destinations.extend([end, start])
            # one-hot bond type (two identical vectors, one for forward and one for reversed edge)
            edge_features.extend(
                [self.one_hot_encode(bond.GetBondType(), self.bond_types)] * 2
            )
        edge_features = torch.stack(edge_features, dim=0)
        return (sources, destinations), edge_features

    def get_atomic_numbers(self, mol):
        atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        return torch.tensor(atomic_numbers)

    def mol2dgl(self, mol):
        """
        Converts rdkit.Chem.rdchem.Mol to dgl.heterograph.DGLHeteroGraph. Takes into
        account chemical properties of atoms and bonds, without considering their 3D positions
        (conformers of the molecule are not used here

        :param mol: The rdkit.Chem.rdchem.Mol object
        :returns: dgl.heterograph.DGLHeteroGraph with .ndata and .edata containing atom and bond features
        """
        node_features = self.get_node_features(mol)
        edges, edge_features = self.get_edges_and_edge_features(mol)
        graph = dgl.graph(edges)
        graph.ndata[constants.atom_feature_name] = node_features
        graph.ndata[constants.atomic_numbers_name] = self.get_atomic_numbers(mol)
        graph.edata[constants.edge_feature_name] = edge_features
        return graph


if __name__ == "__main__":
    # simple test for MolDGLFeaturizer
    from rdkit import Chem

    mol = Chem.MolFromSmiles(constants.ad_smiles)
    mol = Chem.AddHs(mol)

    featurizer = MolDGLFeaturizer(constants.ad_atom_types)

    graph = featurizer.mol2dgl(mol)
    print("node features shape:", graph.ndata[constants.atom_feature_name].shape)
    print("edge features shape:", graph.edata[constants.edge_feature_name].shape)
    print("edges:", *graph.edges(), sep="\n")
    assert graph.ndata[constants.atom_feature_name].shape[0] == mol.GetNumAtoms()
    assert graph.edata[constants.edge_feature_name].shape[0] == 2 * mol.GetNumBonds()
