import torch
from torch import nn

from gflownet.policy.mol_crystals.MikesGraphNet import MikesGraphNet
from gflownet.policy.mol_crystals.model_components import MLP


class molecule_graph_model(nn.Module):
    def __init__(self, dataDims, seed,
                 num_atom_feats,
                 num_mol_feats,
                 output_dimension,
                 activation,
                 num_fc_layers,
                 fc_depth,
                 fc_dropout_probability,
                 fc_norm_mode,
                 graph_filters,
                 graph_convolutional_layers,
                 concat_mol_to_atom_features,
                 pooling,
                 graph_norm,
                 num_spherical,
                 num_radial,
                 graph_convolution,
                 num_attention_heads,
                 add_spherical_basis,
                 add_torsional_basis,
                 graph_embedding_size,
                 radial_function,
                 max_num_neighbors,
                 convolution_cutoff,
                 max_molecule_size,
                 return_latent=False,
                 crystal_mode=False,
                 crystal_convolution_type=None,
                 positional_embedding='sph',
                 atom_embedding_dims=5,
                 device='cuda'):

        super(molecule_graph_model, self).__init__()
        # initialize constants and layers
        self.device = device
        self.return_latent = return_latent
        self.activation = activation
        self.num_fc_layers = num_fc_layers
        self.fc_depth = fc_depth
        self.fc_dropout_probability = fc_dropout_probability
        self.fc_norm_mode = fc_norm_mode
        self.graph_convolution = graph_convolution
        self.output_dimension = output_dimension
        self.graph_convolution_layers = graph_convolutional_layers
        self.graph_filters = graph_filters
        self.graph_norm = graph_norm
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.num_attention_heads = num_attention_heads
        self.add_spherical_basis = add_spherical_basis
        self.add_torsional_basis = add_torsional_basis
        self.n_mol_feats = num_mol_feats  # dataDims['num mol features']
        self.n_atom_feats = num_atom_feats  # dataDims['num atom features']
        self.radial_function = radial_function
        self.max_num_neighbors = max_num_neighbors
        self.graph_convolution_cutoff = convolution_cutoff
        if not concat_mol_to_atom_features:  # if we are not adding molwise feats to atoms, subtract the dimension
            self.n_atom_feats -= self.n_mol_feats
        self.pooling = pooling
        self.fc_norm_mode = fc_norm_mode
        self.graph_embedding_size = graph_embedding_size
        self.crystal_mode = crystal_mode
        self.crystal_convolution_type = crystal_convolution_type
        self.max_molecule_size = max_molecule_size
        self.atom_embedding_dims = atom_embedding_dims  # todo clean this up

        if dataDims is None:
            self.num_atom_types = 101
        else:
            self.num_atom_types = list(dataDims['atom embedding dict sizes'].values())[0] + 1

        torch.manual_seed(seed)

        self.graph_net = MikesGraphNet(
            crystal_mode=crystal_mode,
            crystal_convolution_type=self.crystal_convolution_type,
            graph_convolution_filters=self.graph_filters,
            graph_convolution=self.graph_convolution,
            out_channels=self.fc_depth,
            hidden_channels=self.graph_embedding_size,
            num_blocks=self.graph_convolution_layers,
            num_radial=self.num_radial,
            num_spherical=self.num_spherical,
            max_num_neighbors=self.max_num_neighbors,
            cutoff=self.graph_convolution_cutoff,
            activation='gelu',
            embedding_hidden_dimension=self.atom_embedding_dims,
            num_atom_features=self.n_atom_feats,
            norm=self.graph_norm,
            dropout=self.fc_dropout_probability,
            spherical_embedding=self.add_spherical_basis,
            torsional_embedding=self.add_torsional_basis,
            radial_embedding=self.radial_function,
            num_atom_types=self.num_atom_types,
            attention_heads=self.num_attention_heads,
        )

        # initialize global pooling operation
        self.global_pool = global_aggregation(self.pooling, self.fc_depth,
                                              geometric_embedding=positional_embedding,
                                              num_radial=num_radial,
                                              spherical_order=num_spherical,
                                              radial_embedding=radial_function,
                                              max_molecule_size=max_molecule_size)

        # molecule features FC layer
        if self.n_mol_feats != 0:
            self.mol_fc = nn.Linear(self.n_mol_feats, self.n_mol_feats)

        # FC model to post-process graph fingerprint
        if self.num_fc_layers > 0:
            self.gnn_mlp = MLP(layers=self.num_fc_layers,
                               filters=self.fc_depth,
                               norm=self.fc_norm_mode,
                               dropout=self.fc_dropout_probability,
                               input_dim=self.fc_depth,
                               output_dim=self.fc_depth,
                               conditioning_dim=self.n_mol_feats,
                               seed=seed
                               )
        else:
            self.gnn_mlp = nn.Identity()

        if self.fc_depth != self.output_dimension:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(self.fc_depth, self.output_dimension, bias=False)
        else:
            self.output_fc = nn.Identity()

    def forward(self, data=None, x=None, pos=None, batch=None, ptr=None, aux_ind=None, num_graphs=None, return_latent=False, return_dists=False):
        if data is not None:
            x = data.x
            pos = data.pos
            batch = data.batch
            aux_ind = data.aux_ind
            ptr = data.ptr
            num_graphs = data.num_graphs

        extra_outputs = {}
        if self.n_mol_feats > 0:
            mol_feats = self.mol_fc(x[ptr[:-1], -self.n_mol_feats:])  # molecule features are repeated, only need one per molecule (hence data.ptr)
        else:
            mol_feats = None

        x, dists_dict = self.graph_net(x[:, :self.n_atom_feats], pos, batch, ptr=ptr, ref_mol_inds=aux_ind, return_dists=return_dists)  # get atoms encoding

        if self.crystal_mode:  # model only outputs ref mol atoms - many fewer
            x = self.global_pool(x, pos, batch[torch.where(aux_ind == 0)[0]], output_dim=num_graphs)
        else:
            x = self.global_pool(x, pos, batch, output_dim=num_graphs)  # aggregate atoms to molecule

        if self.num_fc_layers > 0:
            x = self.gnn_mlp(x, conditions=mol_feats)  # mix graph fingerprint with molecule-scale features

        output = self.output_fc(x)

        if return_dists:
            extra_outputs['dists dict'] = dists_dict
        if return_latent:
            extra_outputs['latent'] = x.cpu().detach().numpy()

        if len(extra_outputs) > 0:
            return output, extra_outputs
        else:
            return output
