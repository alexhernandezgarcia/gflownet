from math import pi as PI

from models.basis_functions import TorsionalEmbedding, SphericalBasisLayer, GaussianEmbedding, BesselBasisLayer
from models.components import Normalization, Activation
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter
from torch_sparse import SparseTensor
import torch_geometric.nn as gnn
from models.asymmetric_radius_graph import asymmetric_radius_graph
from models.components import MLP
from old.positional_encodings import PosEncoding3D


class MikesGraphNet(torch.nn.Module):
    def __init__(self, hidden_channels: int,
                 graph_convolution_filters: int,
                 graph_convolution: str,
                 out_channels: int,
                 num_blocks: int,
                 num_spherical: int,
                 num_radial: int,
                 num_atom_types=101,
                 cutoff: float = 5.0,
                 max_num_neighbors: int = 32,
                 envelope_exponent: int = 5,
                 activation='gelu',
                 embedding_hidden_dimension=5,
                 norm=None,
                 dropout=0,
                 radial_embedding='bessel',
                 spherical_embedding=True,
                 torsional_embedding=True,
                 num_atom_features=1,
                 attention_heads=1,
                 crystal_mode=False,
                 crystal_convolution_type=1,
                 positional_embedding=False,
                 ):
        super(MikesGraphNet, self).__init__()

        self.num_blocks = num_blocks
        self.spherical_embedding = spherical_embedding
        self.torsional_embedding = torsional_embedding
        self.max_num_neighbors = max_num_neighbors
        self.cutoff = cutoff
        self.crystal_mode = crystal_mode
        self.convolution_mode = graph_convolution
        self.crystal_convolution_type = crystal_convolution_type
        self.positional_embedding = positional_embedding

        if radial_embedding == 'bessel':
            self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        elif radial_embedding == 'gaussian':
            self.rbf = GaussianEmbedding(start=0.0, stop=cutoff, num_gaussians=num_radial)
        if spherical_embedding:
            self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)
        if torsional_embedding:
            self.tbf = TorsionalEmbedding(num_spherical, num_radial, cutoff, bessel_forms=self.sbf.bessel_forms)
        if positional_embedding:
            self.pos_embedding = PosEncoding3D(hidden_channels // 3, cutoff=10)

        self.atom_embeddings = EmbeddingBlock(hidden_channels, num_atom_types, num_atom_features, embedding_hidden_dimension,
                                              activation)

        self.interaction_blocks = torch.nn.ModuleList([
            GCBlock(graph_convolution_filters,
                    hidden_channels,
                    radial_dim=num_radial,
                    spherical_dim=num_spherical,
                    spherical=spherical_embedding,
                    torsional=torsional_embedding,
                    convolution_mode=graph_convolution,
                    norm=norm,
                    dropout=dropout,
                    heads=attention_heads,
                    )
            for _ in range(num_blocks)
        ])
        self.convolution_mode = graph_convolution

        self.fc_blocks = torch.nn.ModuleList([
            MLP(
                layers=1,
                filters=hidden_channels,
                input_dim=hidden_channels,
                output_dim=hidden_channels,
                activation=activation,
                norm=norm,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        if hidden_channels != out_channels:
            self.output_layer = nn.Linear(hidden_channels, out_channels)
        else:
            self.output_layer = nn.Identity()

    def get_geom_embedding(self, edge_index, pos, num_nodes):
        '''
        compute elements for radial & spherical embeddings
        '''
        i, j = edge_index  # i->j source-to-target
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        sbf, tbf, idx_kj, idx_ji = None, None, None, None

        if torch.sum(dist == 0) > 0:
            zeros_at = torch.where(dist == 0)  # add a little jitter, we absolutely cannot have zero distances
            pos[i[zeros_at]] += (torch.ones_like(pos[i[zeros_at]]) / 5)
            dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        assert torch.sum(dist == 0) == 0

        if self.spherical_embedding:
            i, j = edge_index

            # value = torch.arange(j.size(0), device=j.device)
            adj_t = SparseTensor(row=i, col=j, value=torch.arange(j.size(0), device=j.device),
                                 sparse_sizes=(num_nodes, num_nodes))
            adj_t_row = adj_t[j]
            num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

            # Node indices (k->j->i) for triplets.
            idx_i = i.repeat_interleave(num_triplets)
            idx_j = j.repeat_interleave(num_triplets)
            idx_k = adj_t_row.storage.col()
            mask = idx_i != idx_k  # Remove i == k triplets.
            idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

            # Edge indices (k-j, j->i) for triplets.
            idx_kj = adj_t_row.storage.value()[mask]
            idx_ji = adj_t_row.storage.row()[mask]

            # Calculate angles. 0 to pi
            pos_ji = pos[idx_i] - pos[idx_j]
            pos_jk = pos[idx_k] - pos[idx_j]
            # a = (pos_ji * pos_jk).sum(dim=-1)  # cos_angle * |pos_ji| * |pos_jk|
            # b = torch.cross(pos_ji, pos_jk).norm(dim=-1)  # sin_angle * |pos_ji| * |pos_jk|
            angle = torch.atan2(torch.cross(pos_ji, pos_jk).norm(dim=-1), (pos_ji * pos_jk).sum(dim=-1))

            sbf = self.sbf(dist, angle, idx_kj)

            if self.torsional_embedding:
                idx_batch = torch.arange(len(idx_i), device=angle.device)
                idx_k_n = adj_t[idx_j].storage.col()
                repeat = num_triplets
                num_triplets_t = num_triplets.repeat_interleave(repeat)[mask]
                idx_i_t = idx_i.repeat_interleave(num_triplets_t)
                idx_j_t = idx_j.repeat_interleave(num_triplets_t)
                idx_k_t = idx_k.repeat_interleave(num_triplets_t)
                idx_batch_t = idx_batch.repeat_interleave(num_triplets_t)
                mask = idx_i_t != idx_k_n
                idx_i_t, idx_j_t, idx_k_t, idx_k_n, idx_batch_t = idx_i_t[mask], idx_j_t[mask], idx_k_t[mask], idx_k_n[mask], idx_batch_t[mask]

                pos_j0 = pos[idx_k_t] - pos[idx_j_t]
                pos_ji = pos[idx_i_t] - pos[idx_j_t]
                pos_jk = pos[idx_k_n] - pos[idx_j_t]
                # dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
                plane1 = torch.cross(pos_ji, pos_j0)
                plane2 = torch.cross(pos_ji, pos_jk)
                a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
                b = torch.cross(plane1, plane2).norm(dim=-1)  # sin_angle * |plane1| * |plane2|
                torsion1 = torch.atan2(b, a)  # -pi to pi
                torsion1[torsion1 <= 0] += 2 * PI  # 0 to 2pi
                torsion = scatter(torsion1, idx_batch_t, reduce='min')
                tbf = self.tbf(dist, angle, torsion, idx_kj)
            else:
                tbf = None

        return dist, self.rbf(dist), sbf, tbf, idx_kj, idx_ji

    def forward(self, z, pos, batch, ptr, ref_mol_inds=None, return_dists=False, n_repeats=None):
        # graph model starts here
        x = self.atom_embeddings(z)  # embed atomic numbers & compute initial atom-wise feature vector #

        if self.crystal_mode:  # assumes input with inside-outside structure, and enforces periodicity after each convolution
            inside_inds = torch.where(ref_mol_inds == 0)[0]
            outside_inds = torch.where(ref_mol_inds == 1)[0]  # atoms which are not in the asymmetric unit but which we will convolve - pre-excluding many from outside the cutoff
            inside_batch = batch[inside_inds]  # get the feature vectors we want to repeat
            n_repeats = [int(torch.sum(batch == ii) / torch.sum(inside_batch == ii)) for ii in range(len(ptr) - 1)]  # number of molecules in convolution region

            # intramolecular edges
            edge_index = asymmetric_radius_graph(pos, batch=batch, r=self.cutoff,  # intramolecular interactions - stack over range 3 convolutions
                                                 max_num_neighbors=self.max_num_neighbors, flow='source_to_target',
                                                 inside_inds=inside_inds, convolve_inds=inside_inds)

            # intermolecular edges
            edge_index_inter = asymmetric_radius_graph(pos, batch=batch, r=self.cutoff,  # extra radius for intermolecular graph convolution
                                                       max_num_neighbors=self.max_num_neighbors, flow='source_to_target',
                                                       inside_inds=inside_inds, convolve_inds=outside_inds)

            if self.crystal_convolution_type == 1:  # convolve with inter and intramolecular edges
                edge_index = torch.cat((edge_index, edge_index_inter), dim=1)

            dist, rbf, sbf, tbf, idx_kj, idx_ji = self.get_geom_embedding(edge_index, pos, num_nodes=len(z))

            for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):
                if n < (self.num_blocks - 1):  # to do this molecule-wise, we need to multiply n_repeats by Z for each crystal
                    x = x + convolution(x, rbf, dist, edge_index, batch, sbf=sbf, tbf=tbf, idx_kj=idx_kj, idx_ji=idx_ji)  # graph convolution

                    x[inside_inds] = x[inside_inds] + fc(x[inside_inds], batch=batch[inside_inds])  # feature-wise 1D convolution on only relevant atoms, FC includes residual

                    for ii in range(len(ptr) - 1):  # for each crystal
                        x[ptr[ii]:ptr[ii + 1], :] = x[inside_inds[inside_batch == ii]].repeat(n_repeats[ii], 1)  # copy the first unit cell to all periodic images

                else:  # on the final convolutional block, do not broadcast the reference cell, and include intermolecular interactions in conv_type 2
                    dist_inter, rbf_inter, sbf_inter, tbf_inter, idx_kj_inter, idx_ji_inter = \
                        self.get_geom_embedding(torch.cat((edge_index, edge_index_inter), dim=1), pos, num_nodes=len(z))  # compute no matter what for tracking purposes

                    if self.crystal_convolution_type == 2:
                        x = convolution(x, rbf_inter, dist_inter, torch.cat((edge_index, edge_index_inter), dim=1), batch,
                                        sbf=sbf_inter, tbf=tbf_inter, idx_kj=idx_kj_inter, idx_ji=idx_ji_inter)  # return only the results of the intermolecular convolution, omitting intramolecular features

                    elif self.crystal_convolution_type == 1:
                        x = x + convolution(x, rbf, dist, edge_index, batch, sbf=sbf, tbf=tbf, idx_kj=idx_kj, idx_ji=idx_ji)  # standard graph convolution

                    x = x[inside_inds] + fc(x[inside_inds], batch=batch[inside_inds])  # feature-wise 1D convolution on only relevant atoms, and return only those atoms, FC includes residual

        else:  # isolated molecule
            for n, (convolution, fc) in enumerate(zip(self.interaction_blocks, self.fc_blocks)):

                edge_index = gnn.radius_graph(pos, r=self.cutoff, batch=batch,
                                              max_num_neighbors=self.max_num_neighbors, flow='source_to_target')  # note - requires batch be monotonically increasing
                dist, rbf, sbf, tbf, idx_kj, idx_ji = self.get_geom_embedding(edge_index, pos, num_nodes=len(z))

                # x = self.inside_norm1[n](x)
                if self.convolution_mode != 'none':
                    x = x + convolution(x, rbf, dist, edge_index, batch, sbf=sbf, tbf=tbf, idx_kj=idx_kj, idx_ji=idx_ji)  # graph convolution - residual is already inside the conv operator

                x = fc(x, batch=batch)  # feature-wise 1D convolution, FC includes residual and norm

        if return_dists:  # return dists, batch #, and inside/outside identifier, and atomic number
            dist_output = {}
            dist_output['intramolecular dist'] = dist
            dist_output['intramolecular dist batch'] = batch[edge_index[0]]
            dist_output['intramolecular dist atoms'] = [z[edge_index[0], 0].long(), z[edge_index[1], 0].long()]
            dist_output['intramolecular dist inds'] = edge_index
            if self.crystal_mode:
                dist_output['intermolecular dist'] = (pos[edge_index_inter[0]] - pos[edge_index_inter[1]]).pow(2).sum(dim=-1).sqrt()
                dist_output['intermolecular dist batch'] = batch[edge_index_inter[0]]
                dist_output['intermolecular dist atoms'] = [z[edge_index_inter[0], 0].long(), z[edge_index_inter[1], 0].long()]
                dist_output['intermolecular dist inds'] = edge_index_inter

        # out = self.output_layer(x)
        # assert torch.sum(torch.isnan(out)) == 0

        return self.output_layer(x), dist_output if return_dists else None

        '''
        import networkx as nx
        import matplotlib.pyplot as plt
        intra_edges = (edge_index[:, edge_index[0, :] < ptr[1]].cpu().detach().numpy().T)
        inter_edges = (edge_index_inter[:, edge_index_inter[0, :] < ptr[1]].cpu().detach().numpy().T)
        plt.clf()
        G = nx.Graph()
        G = G.to_directed()
        G.add_weighted_edges_from(np.concatenate((intra_edges, np.ones(len(intra_edges))[:, None] * 2), axis=1))
        G.add_weighted_edges_from(np.concatenate((inter_edges, np.ones(len(inter_edges))[:, None] * 0.25), axis=1))
        edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
        node_weights = np.concatenate((np.ones(9)*2, np.ones(len(G.nodes)-9)))
        nx.draw_kamada_kawai(G, arrows=True, node_size=node_weights * 100, edge_color=weights, linewidths = 1, width=weights, 
        edge_cmap=plt.cm.RdYlGn, node_color = node_weights, cmap=plt.cm.RdYlGn)
        '''


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_atom_types, num_atom_features, atom_type_embedding_dimension, activation='gelu'):
        super(EmbeddingBlock, self).__init__()
        self.num_embeddings = 1
        self.embeddings = nn.Embedding(num_atom_types + 1, atom_type_embedding_dimension)
        self.linear = nn.Linear(atom_type_embedding_dimension + num_atom_features - self.num_embeddings, hidden_channels)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)  # make dim 1 explicit

        embedding = self.embeddings(x[:, 0].long())
        concat_vec = torch.cat([embedding, x[:, self.num_embeddings:]], dim=-1)

        return self.linear(concat_vec)


class GCBlock(torch.nn.Module):
    def __init__(self, graph_convolution_filters, hidden_channels, radial_dim, convolution_mode, spherical_dim=None, spherical=False, torsional=False, norm=None, dropout=0, heads=1):
        super(GCBlock, self).__init__()
        if norm == 'graph':
            message_norm = 'layer'
        else:
            message_norm = norm

        self.norm1 = Normalization(message_norm, graph_convolution_filters)
        self.norm2 = Normalization(message_norm, graph_convolution_filters)
        self.node_to_message = nn.Linear(hidden_channels, graph_convolution_filters, bias=False)
        self.message_to_node = nn.Linear(graph_convolution_filters, hidden_channels, bias=False)  # don't want to send spurious messages, though it probably doesn't matter anyway
        self.radial_to_message = nn.Linear(radial_dim, graph_convolution_filters, bias=False)
        self.convolution_mode = convolution_mode

        if spherical:  # need more linear layers to aggregate angular information to radial
            assert spherical_dim is not None, "Spherical information must have a dimension != 0 for spherical message aggregation"
            self.spherical_to_message = nn.Linear(radial_dim * spherical_dim, graph_convolution_filters, bias=False)
            if not torsional:
                self.radial_spherical_aggregation = nn.Linear(graph_convolution_filters * 2, graph_convolution_filters, bias=False)  # torch.add  # could also do dot
        if torsional:
            assert spherical
            self.torsional_to_message = nn.Linear(spherical_dim * spherical_dim * radial_dim, graph_convolution_filters, bias=False)
            self.radial_torsional_aggregation = nn.Linear(graph_convolution_filters * 2, graph_convolution_filters, bias=False)  # torch.add  # could also do dot

        if convolution_mode == 'GATv2':
            self.GConv = gnn.GATv2Conv(
                in_channels=graph_convolution_filters,
                out_channels=graph_convolution_filters // heads,
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=graph_convolution_filters,
            )
        elif convolution_mode == 'TransformerConv':
            self.GConv = gnn.TransformerConv(
                in_channels=graph_convolution_filters,
                out_channels=graph_convolution_filters // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=graph_convolution_filters,
                beta=True,
            )
        elif convolution_mode == 'full message passing':
            self.GConv = MPConv(
                in_channels=graph_convolution_filters,
                out_channels=graph_convolution_filters,
                edge_dim=graph_convolution_filters,
                dropout=dropout,
                norm=None,  # can't do graph norm here
            )
        elif convolution_mode.lower() == 'schnet':  #
            assert not spherical, 'schnet currently only works with pure radial bases'
            self.GConv = CFConv(
                in_channels=graph_convolution_filters,
                out_channels=graph_convolution_filters,
                num_filters=graph_convolution_filters,
                cutoff=5
                , )
        elif convolution_mode == 'none':
            self.GConv = nn.Identity()

    def compute_edge_attributes(self, edge_index, rbf, idx_ji, sbf=None, tbf=None, idx_kj=None):
        if tbf is not None:
            # aggregate spherical and torsional messages to radial
            edge_attr = (self.radial_torsional_aggregation(
                torch.cat((self.radial_to_message(rbf)[idx_kj], self.torsional_to_message(tbf)), dim=1)))  # combine radial and torsional info in triplet space
            # torch.sum(torch.stack((self.radial_to_message(rbf)[idx_kj], self.spherical_to_message(sbf), self.torsional_to_message(tbf))),dim=0)
            edge_attr = scatter(edge_attr, idx_ji, dim=0)  # collect triplets back down to pair space

        elif sbf is not None and tbf is None:
            # aggregate spherical messages to radial
            # rbf = self.radial_to_message(rbf)
            # sbf = self.spherical_message(sbf)
            edge_attr = (self.radial_spherical_aggregation(torch.cat((self.radial_to_message(rbf)[idx_kj], self.spherical_to_message(sbf)), dim=1)))  # combine radial and spherical info in triplet space
            edge_attr = scatter(edge_attr, idx_ji, dim=0)  # collect triplets back down to pair space

        else:  # no angular information
            edge_attr = (self.radial_to_message(rbf))

        # sometimes, there are different numbers edges according to spherical and radial bases (usually, trailing zeros, I think), so we force them to align
        if len(edge_attr) != edge_index.shape[1]:
            edge_index = edge_index[:, :len(edge_attr)]

        return edge_attr, edge_index

    def forward(self, x, rbf, dists, edge_index, batch, sbf=None, tbf=None, idx_kj=None, idx_ji=None):
        # convert local information into edge weights
        x = self.norm1(self.node_to_message(x), batch)
        edge_attr, edge_index = self.compute_edge_attributes(edge_index, rbf, idx_ji, sbf, tbf, idx_kj)
        edge_attr = self.norm2(edge_attr, batch[edge_index[0]])

        # convolve # todo only update nodes which will actually pass messages on this round
        if self.convolution_mode.lower() == 'schnet':
            x = self.GConv(x, edge_index, dists, edge_attr)
        elif self.convolution_mode.lower() == 'none':
            pass
        else:
            x = self.GConv(x, edge_index, edge_attr)

        x = self.message_to_node(x)

        return x


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CFConv(gnn.MessagePassing):
    '''
    ~~the graph convolution used in the popular SchNet~~
    '''

    def __init__(self, in_channels, out_channels, num_filters, cutoff):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.cutoff = cutoff

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)  # de-weight distant nodes
        # W = self.nn(edge_attr) * C.view(-1, 1)
        W = edge_attr * C.view(-1, 1)  # in my method, edge_attr are pre-featurized

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)

        return x

    def message(self, x_j, W):
        return x_j * W


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim, dropout=0, norm=None, activation='leaky relu'):
        super(MPConv, self).__init__()

        self.MLP = MLP(layers=4,
                       filters=out_channels,
                       input_dim=in_channels * 2 + edge_dim,
                       dropout=dropout,
                       norm=norm,
                       output_dim=out_channels,
                       activation=activation,
                       )

    def forward(self, x, edge_index, edge_attr):
        m = self.MLP(torch.cat((x[edge_index[0]], x[edge_index[1]], edge_attr), dim=-1))

        return scatter(m, edge_index[1], dim=0, dim_size=len(x))  # send directional messages from i to j, enforcing the size of the output dimension


class FCBlock(torch.nn.Module):
    '''
    fully-connected block, following the original transformer architecture with norm first
    '''

    def __init__(self, hidden_channels, norm, dropout, activation):
        super(FCBlock, self).__init__()
        self.norm = Normalization(norm, hidden_channels)
        self.activation = Activation(activation, hidden_channels)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(self.norm(x)))))
        return x


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
        self.linear = nn.Conv1d(channels * n_basis, channels, kernel_size=(1, 1), groups=int(channels), bias=False)

        # nn.init.normal(self.linear.weight.data, std=0.1)

    def kernel(self, x):
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(2)
        if len(x) == 2:
            x = x.reshape(2, self.channels, 1)

        return torch.exp(-self.gamma * (x - self.dict) ** 2)

    def forward(self, x):
        x = self.kernel(x).unsqueeze(-1).unsqueeze(-1)  # run activation, output shape batch, features, y, x, basis
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])  # concatenate basis functions with filters
        x = self.linear(x).squeeze(-1).squeeze(-1)  # apply linear coefficients and sum

        return x


def triplets(edge_index, num_nodes):
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji
