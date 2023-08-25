import torch

from torch import nn as nn
from mol_crystals.model_components import (molecule_graph_model, MLP)


class crystal_generator(nn.Module):
    def __init__(self, device, n_node_feats, n_graph_feats, max_mol_radius, output_action_space_size, seed=0):
        super(crystal_generator, self).__init__()

        self.device = device
        torch.manual_seed(seed)

        graph_embedding_size = 256
        self.conditioner = molecule_graph_model(
            dataDims=None,
            atom_embedding_dims=5,
            seed=seed,
            num_atom_feats=n_node_feats,  # we will add directly the normed coordinates to the node features
            num_mol_feats=n_graph_feats,
            output_dimension=graph_embedding_size,
            activation='gelu',
            num_fc_layers=4,
            fc_depth=256,
            output_dimension = output_action_space_size,
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

        '''
        generator model
        '''


    def forward(self, conditions):
        normed_coords = conditions.pos / self.conditioner.max_molecule_size  # norm coords by maximum molecule radius
        crystal_information = conditions.x[:, -self.num_crystal_features:]

        if self.skinny_inputs:
            conditions.x = torch.cat((conditions.x[:, 0, None], normed_coords), dim=-1)  # take only the atomic number for atomwise features
        else:
            conditions.x = torch.cat((conditions.x[:, :-self.num_crystal_features], normed_coords), dim=-1)  # concatenate to input features, leaving out crystal info from conditioner

        conditions_encoding = self.conditioner(conditions)
        conditions_encoding = torch.cat((conditions_encoding, crystal_information[conditions.ptr[:-1]]), dim=-1)

        return self.model(conditions=conditions_encoding)


