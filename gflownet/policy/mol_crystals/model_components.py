import sys
import torch
from torch import nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from MikesGraphNet import MikesGraphNet


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
        self.output_classes = output_dimension
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

        if self.fc_depth != self.output_classes:  # only want this if we have to change the dimension
            self.output_fc = nn.Linear(self.fc_depth, self.output_classes, bias=False)
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


class MLP(nn.Module):
    def __init__(self, layers, filters, input_dim, output_dim,
                 activation='gelu', seed=0, dropout=0, conditioning_dim=0,
                 norm=None, bias=True, norm_after_linear=False):
        super(MLP, self).__init__()
        # initialize constants and layers

        self.n_layers = layers

        if isinstance(filters, list):
            self.n_filters = filters
        else:
            self.n_filters = [filters for n in range(layers + 1)]
            self.same_depth = True

        if self.n_filters.count(self.n_filters[0]) != len(self.n_filters):  # if they are not all the same, we need residue adjustments
            self.same_depth = False
            self.residue_adjust = torch.nn.ModuleList([
                nn.Linear(self.n_filters[i], self.n_filters[i + 1], bias=False)
                for i in range(self.n_layers)
            ])
        else:
            self.same_depth = True

        self.input_filters = self.n_filters
        self.output_filters = self.n_filters

        self.conditioning_dim = conditioning_dim
        self.output_dim = output_dim
        self.input_dim = input_dim + conditioning_dim
        self.norm_mode = norm
        self.dropout_p = dropout
        self.activation = activation
        self.norm_after_linear = norm_after_linear

        torch.manual_seed(seed)

        self.fc_layers = torch.nn.ModuleList([
            nn.Linear(self.n_filters[i], self.n_filters[i + 1], bias=bias)
            for i in range(self.n_layers)
        ])
        if norm_after_linear:
            self.fc_norms = torch.nn.ModuleList([
                Normalization(self.norm_mode, self.n_filters[i + 1])
                for i in range(self.n_layers)
            ])
            self.fc_activations = torch.nn.ModuleList([
                Activation(activation, self.n_filters[i + 1])
                for i in range(self.n_layers)
            ])

        else:
            self.fc_norms = torch.nn.ModuleList([
                Normalization(self.norm_mode, self.n_filters[i])
                for i in range(self.n_layers)
            ])
            self.fc_activations = torch.nn.ModuleList([
                Activation(activation, self.n_filters[i])
                for i in range(self.n_layers)
            ])

        self.fc_dropouts = torch.nn.ModuleList([
            nn.Dropout(p=self.dropout_p)
            for _ in range(self.n_layers)
        ])

        if self.input_dim != self.n_filters[0]:
            self.init_layer = nn.Linear(self.input_dim, self.n_filters[0])  # set appropriate sizing
        else:
            self.init_layer = nn.Identity()

        if self.output_dim != self.n_filters[-1]:
            self.output_layer = nn.Linear(self.n_filters[-1], self.output_dim, bias=False)
        else:
            self.output_layer = nn.Identity()

    def forward(self, x, conditions=None, return_latent=False, batch=None):
        if 'geometric' in str(type(x)):  # extract conditions from trailing atomic features
            # todo fix above
            # if x.num_graphs == 1:
            #     x = x.x[:, -self.input_dim:]
            # else:
            x = gnn.global_max_pool(x.x, x.batch)[:, -self.input_dim:]  # x.x[x.ptr[:-1]][:, -self.input_dim:]

        if conditions is not None:
            # if type(conditions) == torch_geometric.data.batch.DataBatch: # extract conditions from trailing atomic features
            #     if len(x) == 1:
            #         conditions = conditions.x[:,-self.conditioning_dim:]
            #     else:
            #         conditions = conditions.x[conditions.ptr[:-1]][:,-self.conditioning_dim:]

            x = torch.cat((x, conditions), dim=1)

        x = self.init_layer(x)  # get the right feature depth

        for i, (norm, linear, activation, dropout) in enumerate(zip(self.fc_norms, self.fc_layers, self.fc_activations, self.fc_dropouts)):
            if self.same_depth:
                res = x.clone()
            else:
                res = self.residue_adjust[i](x)
            if self.norm_after_linear:
                x = res + dropout(activation(norm(linear(x), batch=batch)))  # residue
            else:
                x = res + dropout(activation(linear(norm(x, batch=batch))))  # residue

        if return_latent:
            return self.output_layer(x), x
        else:
            return self.output_layer(x)


class Normalization(nn.Module):
    def __init__(self, norm, filters, *args, **kwargs):
        super().__init__()
        self.norm_type = norm
        if norm == 'batch':
            self.norm = gnn.BatchNorm(filters)
        elif norm == 'graph layer':
            self.norm = gnn.LayerNorm(filters)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(filters)
        elif norm == 'instance':
            self.norm = gnn.InstanceNorm(filters)  # not tested
        elif norm == 'graph':
            self.norm = gnn.GraphNorm(filters)
        elif norm is None:
            self.norm = nn.Identity()
        else:
            print(norm + " is not a valid normalization")
            sys.exit()

    def forward(self, input, batch=None):
        if batch is not None and self.norm_type != 'batch' and self.norm_type is not None:
            return self.norm(input, batch)

        return self.norm(input)


class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func.lower() == 'relu':
            self.activation = F.relu
        elif activation_func.lower() == 'gelu':
            self.activation = F.gelu
        elif activation_func.lower() == 'kernel':
            self.activation = kernelActivation(n_basis=10, span=4, channels=filters)
        elif activation_func.lower() == 'leaky relu':
            self.activation = F.leaky_relu

    def forward(self, input):
        return self.activation(input)


class kernelActivation(nn.Module):  # a better (pytorch-friendly) implementation of activation as a linear combination of basis functions
    def __init__(self, n_basis, span, channels, *args, **kwargs):
        super(kernelActivation, self).__init__(*args, **kwargs)

        self.channels, self.n_basis = channels, n_basis
        # define the space of basis functions
        self.register_buffer('dict', torch.linspace(-span, span, n_basis))  # positive and negative values for Dirichlet Kernel
        gamma = 1 / (6 * (self.dict[-1] - self.dict[-2]) ** 2)  # optimum gaussian spacing parameter should be equal to 1/(6*spacing^2) according to KAFnet paper
        self.register_buffer('gamma', torch.ones(1) * gamma)  #

        # self.register_buffer('dict', torch.linspace(0, n_basis-1, n_basis)) # positive values for ReLU kernel

        # define module to learn parameters
        # 1d convolutions allow for grouping of terms, unlike nn.linear which is always fully-connected.
        # #This way should be fast and efficient, and play nice with pytorch optim
        self.linear = nn.Conv1d(channels * n_basis, channels, kernel_size=1, groups=int(channels), bias=False)

        # nn.init.normal(self.linear.weight.data, std=0.1)

    def kernel(self, x):
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(2)
        if len(x) == 2:
            x = x.reshape(2, self.channels, 1)

        return torch.exp(-self.gamma * (x - self.dict) ** 2)

    def forward(self, x):
        x = self.kernel(x).unsqueeze(-1)  # run activation, output shape batch, features, y, x, basis
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])  # concatenate basis functions with filters
        x = self.linear(x).squeeze(-1).squeeze(-1)  # apply linear coefficients and sum

        return x
