import torch

from torch import nn as nn
from gflownet.policy.mol_crystals.model_components import (MLP)
from gflownet.policy.mol_crystals.molecule_graph_model import molecule_graph_model


class GraphConditionedPolicy(nn.Module):
    def __init__(self, device, hidden_depth, n_layers, state_dim, n_node_feats, n_graph_feats, max_mol_radius, output_dim, n_crystal_features, seed=0):
        super(GraphConditionedPolicy, self).__init__()

        self.device = device
        self.max_mol_radius = max_mol_radius
        torch.manual_seed(seed)

        self.n_node_features = n_node_feats
        self.n_graph_features = n_graph_feats
        self.n_crystal_features = n_crystal_features
        graph_dimension = 256
        self.conditioner = molecule_graph_model(
            dataDims=None,
            atom_embedding_dims=5,
            seed=seed,
            num_atom_feats=n_node_feats + 3 - n_crystal_features,  # we will add directly the normed coordinates to the node features
            num_mol_feats=n_graph_feats,
            output_dimension=graph_dimension,
            activation='gelu',
            num_fc_layers=0,
            fc_depth=0,
            fc_dropout_probability=0,
            fc_norm_mode=None,
            graph_filters=graph_dimension // 2,
            graph_convolutional_layers=2,
            concat_mol_to_atom_features=True,
            pooling='max',
            graph_norm=None,
            num_spherical=6,
            num_radial=32,
            graph_convolution='TransformerConv',
            num_attention_heads=1,
            add_spherical_basis=False,
            add_torsional_basis=False,
            graph_embedding_size=graph_dimension,
            radial_function='gaussian',
            max_num_neighbors=100,
            convolution_cutoff=6,
            positional_embedding=None,
            max_molecule_size=max_mol_radius,
            crystal_mode=False,
            crystal_convolution_type=None,
        )
        self.model = MLP(
            input_dim=state_dim,
            conditioning_dim = 0, #graph_dimension + 1,
            layers=n_layers,
            filters=hidden_depth,
            output_dim = output_dim,
            norm=None,
            dropout=0,
            activation='gelu'
        )

    def forward(self, states, conditions):  # combine state & conditions for input
        """
        :param states:
        :param conditions:
        :return:
        conditions include atom & mol-wise features, and normed atom coordinates, point Net style
        convolutions are done with radial basis functions
        graph convolution is TransformerConv conditioned on edge embeddings
        graph -> gnn -> mlp -> output
        """
        # conditions = conditions.clone()  # avoid contaminating the source data
        # normed_coords = conditions.pos / self.max_mol_radius  # norm coords by maximum molecule radius
        # conditions.x = torch.cat((conditions.x[:, :-self.n_crystal_features], normed_coords,), dim=-1)  # concatenate states and coordinates to input features, leaving out crystal info

        #return self.model(states, conditions=torch.cat((conditions.y[:,None].float(),self.conditioner(conditions)),dim=-1))  # conditional generation

        return self.model(states)  # unconditional generation

