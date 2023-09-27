from copy import deepcopy

import dgl
import torch
from dgl.nn.pytorch.conv import EGNNConv
from torch import nn

from gflownet.policy.base import Policy


def _build_mlp(
    input_dim: int, hidden_dim: int, output_dim: int, n_layers: int
) -> nn.Module:
    mlp_layers = []
    for i in range(n_layers):
        mlp_layers.append(
            (
                nn.Linear(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim if i < n_layers - 1 else output_dim,
                )
            )
        )
        if i < n_layers - 1:
            mlp_layers.append(nn.SiLU())
    return nn.Sequential(*mlp_layers)


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
        n_torsion_angles: int = 2,
        separate_mlp_per_torsion: bool = False,
    ):
        super().__init__()

        self.out_dim = out_dim

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

        if separate_mlp_per_torsion:
            self.mlp = nn.ModuleList(
                [
                    _build_mlp(
                        2 * egnn_hidden_dim, mlp_hidden_dim, out_dim, n_mlp_layers
                    )
                    for _ in range(n_torsion_angles)
                ]
            )
        else:
            self.mlp = _build_mlp(
                2 * egnn_hidden_dim, mlp_hidden_dim, out_dim, n_mlp_layers
            )

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
        edge_features = g.edata["edge_features"][g.edata["rotatable_edges"]]

        if isinstance(self.mlp, nn.ModuleList):
            edge_features = edge_features.reshape(
                g.batch_size, len(self.mlp), -1
            ).permute(1, 0, 2)
            output = torch.hstack(
                [mlp(edge_features[i]) for i, mlp in enumerate(self.mlp)]
            )
        else:
            output = self.mlp(edge_features)
            output = output.reshape(g.batch_size, -1)

        return output


class EGNNPolicy(Policy):
    def __init__(self, config, env, device, float_precision, base=None):
        self.model = None
        self.config = None
        self.use_fake_edges = None
        self.fake_edge_radius = None
        self.graph = EGNNPolicy.create_core_graph(env.graph).to(device)
        self.n_torsion_angles = env.n_dim
        self.n_components = env.n_comp
        # We increase the node feature size by 1 to anticipate including current
        # timestamp as one of the features.
        self.node_feat_dim = env.graph.ndata["atom_features"].shape[1] + 1
        # We increase the edge feature size by 1 to anticipate including one-hot
        # encoding for fake edges.
        self.edge_feat_dim = env.graph.edata["edge_features"].shape[1] + 1
        self.is_model = True

        super().__init__(
            config=config,
            env=env,
            device=device,
            float_precision=float_precision,
            base=base,
        )

    @staticmethod
    def create_core_graph(graph: dgl.DGLGraph) -> dgl.DGLGraph:
        output = dgl.graph(graph.edges())
        output.ndata["atom_features"] = graph.ndata["atom_features"].clone().detach()
        output.edata["edge_features"] = torch.cat(
            [
                graph.edata["edge_features"].clone().detach(),
                torch.zeros(graph.num_edges(), 1),
            ],
            dim=1,
        )
        output.edata["rotatable_edges"] = (
            graph.edata["rotatable_edges"].clone().detach()
        )

        return output

    def add_fake_edges(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        in_proximity = (
            torch.cdist(graph.ndata["coordinates"], graph.ndata["coordinates"])
            <= self.fake_edge_radius
        )
        adjacent = graph.adj().to_dense().bool()
        new_edges = torch.nonzero(
            in_proximity
            & ~adjacent.to(self.device)
            & ~torch.eye(graph.num_nodes(), device=self.device).bool()
        )
        edge_features = torch.zeros(
            (new_edges.shape[0], graph.edata["edge_features"].shape[1]),
            device=self.device,
        )
        edge_features[:, -1] = 1
        graph = dgl.add_edges(
            graph,
            new_edges[:, 0],
            new_edges[:, 1],
            {
                "edge_features": edge_features,
                "rotatable_edges": torch.zeros(
                    new_edges.shape[0], dtype=torch.bool, device=self.device
                ),
            },
        )

        return graph

    def parse_config(self, config):
        self.config = {} if config is None else config

        self.use_fake_edges = self.config.pop("use_fake_edges", False)
        self.fake_edge_radius = self.config.pop("fake_edge_radius", 2.0)

    def instantiate(self):
        self.model = EGNNModel(
            self.n_components * 3,
            self.node_feat_dim,
            self.edge_feat_dim,
            n_torsion_angles=self.n_torsion_angles,
            **self.config
        ).to(self.device)

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        graphs = []
        for state in states:
            graph = deepcopy(self.graph)
            graph.ndata["atom_features"] = torch.cat(
                [
                    graph.ndata["atom_features"],
                    state[:, -1].unsqueeze(-1),
                ],
                dim=1,
            )
            graph.ndata["coordinates"] = state[:, :-1]
            if self.use_fake_edges:
                graph = self.add_fake_edges(graph)
            graphs.append(graph)
        batch = dgl.batch(graphs).to(self.device)
        output = self.model(batch)
        return output
