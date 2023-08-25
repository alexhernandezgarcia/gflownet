import torch

from torch import nn as nn
from gflownet.policy.mol_crystals.model_components import (MLP)
from gflownet.policy.mol_crystals.molecule_graph_model import molecule_graph_model


class GraphConditionedPolicy(nn.Module):
    def __init__(self, device, n_node_feats, n_graph_feats, max_mol_radius, output_dim, num_crystal_features, seed=0):
        super(GraphConditionedPolicy, self).__init__()

        self.device = device
        torch.manual_seed(seed)

        self.num_crystal_features = num_crystal_features
        self.model = molecule_graph_model(
            dataDims=None,
            atom_embedding_dims=5,
            seed=seed,
            num_atom_feats=n_node_feats + 3,  # we will add directly the normed coordinates to the node features
            num_mol_feats=n_graph_feats,
            output_dimension=output_dim,
            activation='gelu',
            num_fc_layers=4,
            fc_depth=256,
            fc_dropout_probability=0,
            fc_norm_mode=None,
            graph_filters=128,
            graph_convolutional_layers=4,
            concat_mol_to_atom_features=True,
            pooling='max',
            graph_norm='graph layer',
            num_spherical=6,
            num_radial=32,
            graph_convolution='TransformerConv',
            num_attention_heads=1,
            add_spherical_basis=False,
            add_torsional_basis=False,
            graph_embedding_size=256,
            radial_function='gaussian',
            max_num_neighbors=100,
            convolution_cutoff=6,
            positional_embedding=None,
            max_molecule_size=max_mol_radius,
            crystal_mode=False,
            crystal_convolution_type=None,
        )

    def forward(self, conditions):  # combine state & conditions for input
        '''
        :param conditions:
        :return:
        conditions include atom & mol-wise features, and normed atom coordinates, point Net style
        convolutions are done with radial basis functions
        graph convolution is TransformerConv conditioned on edge embeddings
        graph -> gnn -> mlp -> output
        '''
        normed_coords = conditions.pos / self.conditioner.max_molecule_size  # norm coords by maximum molecule radius
        conditions.x = torch.cat((conditions.x[:, :-self.num_crystal_features], normed_coords), dim=-1)  # concatenate to input features, leaving out crystal info from conditioner

        return self.model(conditions)


