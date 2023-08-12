from copy import deepcopy
from typing import Optional

import dgl
import torch
from dgl.nn.pytorch.conv import EGNNConv
from dgl.nn.pytorch.glob import SumPooling
from torch import nn

from gflownet.policy.base import Policy


class EGNNModel(nn.Module):
    def __init__(
        self,
        out_dim: int,
        node_feat_dim: int,
        edge_feat_dim: int = 0,
        n_gnn_layers: int = 7,
        n_node_mlp_layers: int = 2,
        n_pool_mlp_layers: int = 2,
        egnn_hidden_dim: int = 128,
        node_mlp_hidden_dim: int = 128,
        pool_mlp_hidden_dim: int = 128,
    ):
        super().__init__()

        egnn_layers = []
        for i in range(n_gnn_layers):
            egnn_layers.append(
                EGNNConv(
                    node_feat_dim if i == 0 else egnn_hidden_dim,
                    egnn_hidden_dim,
                    egnn_hidden_dim,
                    edge_feat_dim,
                )
            )
        self.egnn_layers = nn.ModuleList(egnn_layers)

        node_mlp_layers = []
        for i in range(n_node_mlp_layers):
            node_mlp_layers.append(
                (
                    nn.Linear(
                        egnn_hidden_dim if i == 0 else node_mlp_hidden_dim,
                        node_mlp_hidden_dim,
                    )
                )
            )
            if i < n_node_mlp_layers - 1:
                node_mlp_layers.append(nn.SiLU())
        self.node_mlp = nn.Sequential(*node_mlp_layers)

        self.pool = SumPooling()

        pool_mlp_layers = []
        for i in range(n_pool_mlp_layers):
            pool_mlp_layers.append(
                (
                    nn.Linear(
                        node_mlp_hidden_dim if i == 0 else pool_mlp_hidden_dim,
                        pool_mlp_hidden_dim if i < n_pool_mlp_layers - 1 else out_dim,
                    )
                )
            )
            if i < n_pool_mlp_layers - 1:
                pool_mlp_layers.append(nn.SiLU())
        self.pool_mlp = nn.Sequential(*pool_mlp_layers)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feat: torch.Tensor,
        coord_feat: torch.Tensor,
        edge_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h, x = node_feat, coord_feat
        for egnn_layer in self.egnn_layers:
            h, x = egnn_layer(g, h, x, edge_feat)
        h = self.node_mlp(h)
        h = self.pool(g, h)
        h = self.pool_mlp(h)

        return h


class EGNNPolicy(Policy):
    def __init__(self, config, env, device, float_precision, base=None):
        self.model = None
        self.config = None
        self.graph = env.graph
        # We increase the node feature size by 1 to anticipate including current
        # timestamp as one of the features.
        self.node_feat_dim = env.graph.ndata["atom_features"].shape[1] + 1
        self.edge_feat_dim = env.graph.edata["edge_features"].shape[1]
        self.is_model = True

        super().__init__(
            config=config,
            env=env,
            device=device,
            float_precision=float_precision,
            base=base,
        )

    def parse_config(self, config):
        self.config = {} if config is None else config

    def instantiate(self):
        self.model = EGNNModel(
            self.output_dim, self.node_feat_dim, self.edge_feat_dim, **self.config
        ).to(self.device)

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        graphs = []
        for state in states:
            graph = deepcopy(self.graph).to(self.device)
            graph.ndata["atom_features"] = torch.cat(
                [
                    graph.ndata["atom_features"],
                    state[:, -1].unsqueeze(-1),
                ],
                dim=1,
            )
            graph.ndata["coordinates"] = state[:, :-1]
            graphs.append(graph)
        batch = dgl.batch(graphs)
        output = self.model(
            batch,
            batch.ndata["atom_features"],
            batch.ndata["coordinates"],
            batch.edata["edge_features"],
        )
        return output
