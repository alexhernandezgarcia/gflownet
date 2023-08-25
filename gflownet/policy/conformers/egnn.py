from copy import deepcopy

import dgl
import torch
from dgl.nn.pytorch.conv import EGNNConv
from torch import nn

from gflownet.policy.base import Policy


class EGNNModel(nn.Module):
    def __init__(
        self,
        out_dim: int,
        node_feat_dim: int,
        edge_feat_dim: int = 0,
        n_gnn_layers: int = 7,
        n_mlp_layers: int = 2,
        egnn_hidden_dim: int = 128,
        mlp_hidden_dim: int = 128,
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

        mlp_layers = []
        for i in range(n_mlp_layers):
            mlp_layers.append(
                (
                    nn.Linear(
                        2 * egnn_hidden_dim if i == 0 else mlp_hidden_dim,
                        mlp_hidden_dim if i < n_mlp_layers - 1 else out_dim,
                    )
                )
            )
            if i < n_mlp_layers - 1:
                mlp_layers.append(nn.SiLU())
        self.mlp = nn.Sequential(*mlp_layers)

    @staticmethod
    def concat_message_function(edges):
        return {
            "edge_features": torch.cat(
                [edges.src["atom_features"], edges.dst["atom_features"]], dim=1
            )
        }

    def forward(
        self,
        g: dgl.DGLGraph,
    ) -> torch.Tensor:
        for egnn_layer in self.egnn_layers:
            g.ndata["atom_features"], g.ndata["coordinates"] = egnn_layer(
                g,
                g.ndata["atom_features"],
                g.ndata["coordinates"],
                g.edata["edge_features"],
            )

        g.apply_edges(EGNNModel.concat_message_function)

        graphs = dgl.unbatch(g)
        graphs = [dgl.edge_subgraph(g, g.edata["rotatable_edges"]) for g in graphs]
        g = dgl.batch(graphs)

        g.edata["edge_features"] = self.mlp(g.edata["edge_features"])
        graphs = dgl.unbatch(g)

        output = torch.stack([g.edata["edge_features"].flatten() for g in graphs])

        return output


class EGNNPolicy(Policy):
    def __init__(self, config, env, device, float_precision, base=None):
        self.model = None
        self.config = None
        self.graph = env.graph
        self.n_components = env.n_comp
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
            self.n_components * 3, self.node_feat_dim, self.edge_feat_dim, **self.config
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
        output = self.model(batch)
        return output
