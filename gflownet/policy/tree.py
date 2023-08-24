from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from omegaconf import OmegaConf
from torch_geometric.nn import global_mean_pool


class Backbone(torch.nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        hidden_dim: int = 64,
        input_dim: int = 5,
        layer: str = "GCNConv",
        activation: str = "LeakyReLU",
        dropout: float = 0.5,
    ):
        super().__init__()

        layer = getattr(torch_geometric.nn, layer)
        activation = getattr(torch.nn, activation)

        layers = []
        for i in range(n_layers):
            layers.append(
                (
                    layer(input_dim if i == 0 else hidden_dim, hidden_dim),
                    "x, edge_index -> x",
                )
            )
            layers.append(activation())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))

        self.model = torch_geometric.nn.Sequential("x, edge_index, batch", layers)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        return self.model(x, edge_index, batch)


class LeafSelectionHead(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        n_layers: int = 2,
        hidden_dim: int = 64,
        layer: str = "GCNConv",
        activation: str = "LeakyReLU",
        dropout: float = 0.5,
    ):
        super().__init__()

        layer = getattr(torch_geometric.nn, layer)
        activation = getattr(torch.nn, activation)

        layers = []
        for i in range(n_layers):
            layers.append(
                (
                    layer(hidden_dim, 1 if i == n_layers - 1 else hidden_dim),
                    "x, edge_index -> x",
                )
            )
            if i < n_layers - 1:
                layers.append(activation())
                if dropout > 0:
                    layers.append(torch.nn.Dropout(p=dropout))

        self.backbone = backbone
        self.model = torch_geometric.nn.Sequential("x, edge_index, batch", layers)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        mask = x[:, 0] == 0
        x = self.backbone(data)
        x = self.model(x, edge_index, batch)
        x = x.squeeze(-1)
        x = x.masked_fill(mask, -np.inf)

        return F.softmax(x, dim=0)


def _construct_node_head(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_layers: int,
    activation: str,
    dropout: float,
) -> torch.nn.Module:
    activation = getattr(torch.nn, activation)

    layers = []
    for i in range(n_layers):
        layers.append(
            torch.nn.Linear(
                input_dim if i == 0 else hidden_dim,
                output_dim if i == n_layers - 1 else hidden_dim,
            ),
        )
        if i < n_layers - 1:
            layers.append(activation())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
    layers.append(torch.nn.ReLU())

    return torch.nn.Sequential(*layers)


class FeatureSelectionHead(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        output_dim: int,
        n_layers: int = 2,
        input_dim: int = 128,
        hidden_dim: int = 64,
        activation: str = "LeakyReLU",
        dropout: float = 0.5,
    ):
        super().__init__()

        self.backbone = backbone
        self.model = _construct_node_head(
            input_dim, hidden_dim, output_dim, n_layers, activation, dropout
        )

    def forward(
        self, data: torch_geometric.data.Data, node_index: torch.Tensor
    ) -> torch.Tensor:
        x, edge_index, batch = (data.x, data.edge_index, data.batch)
        x = self.backbone(data)
        x_pool = global_mean_pool(x, batch)
        x_node = x[node_index, :]
        x = torch.cat([x_pool, x_node], dim=1)
        x = self.model(x)

        return x


class ThresholdSelectionHead(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        input_dim: int,
        output_dim: int,
        n_layers: int = 2,
        hidden_dim: int = 64,
        activation: str = "LeakyReLU",
        dropout: float = 0.5,
    ):
        super().__init__()

        self.backbone = backbone
        self.model = _construct_node_head(
            input_dim, hidden_dim, output_dim, n_layers, activation, dropout
        )

    def forward(
        self,
        data: torch_geometric.data.Data,
        node_index: torch.Tensor,
        feature_index: torch.Tensor,
    ) -> torch.Tensor:
        x, edge_index, batch = (data.x, data.edge_index, data.batch)
        x = self.backbone(data)
        x_pool = global_mean_pool(x, batch)
        x_node = x[node_index, :]
        x = torch.cat([x_pool, x_node, feature_index], dim=1)
        x = self.model(x)

        return x


class OperatorSelectionHead(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        input_dim: int,
        n_layers: int = 2,
        hidden_dim: int = 64,
        activation: str = "LeakyReLU",
        dropout: float = 0.5,
    ):
        super().__init__()

        self.backbone = backbone
        self.model = _construct_node_head(
            input_dim, hidden_dim, 1, n_layers, activation, dropout
        )

    def forward(
        self,
        data: torch_geometric.data.Data,
        node_index: torch.Tensor,
        feature_index: torch.Tensor,
        threshold: torch.Tensor,
    ) -> torch.Tensor:
        x, edge_index, batch = (data.x, data.edge_index, data.batch)
        x = self.backbone(data)
        x_pool = global_mean_pool(x, batch)
        x_node = x[node_index, :]
        x = torch.cat([x_pool, x_node, feature_index, threshold], dim=1)
        x = self.model(x)

        return x


class TreeModel(torch.nn.Module):
    def __init__(
        self,
        backbone: Optional[torch.nn.Module] = None,
        backbone_args: Optional[dict] = None,
        leaf_head_args: Optional[dict] = None,
        feature_head_args: Optional[dict] = None,
        threshold_head_args: Optional[dict] = None,
        operator_head_args: Optional[dict] = None,
    ):
        super().__init__()

        if backbone is None:
            self.backbone = Backbone(**backbone_args)
        else:
            self.backbone = backbone

        self.leaf_head = LeafSelectionHead(backbone=backbone, **leaf_head_args)
        self.feature_head = FeatureSelectionHead(backbone=backbone, **feature_head_args)
        self.threshold_head = ThresholdSelectionHead(
            backbone=backbone, **threshold_head_args
        )
        self.operator_head = OperatorSelectionHead(
            backbone=backbone, **operator_head_args
        )

    def forward(self, x):
        pass


class TreePolicy:
    def __init__(self, config, env, device, float_precision, base=None):
        # Device and float precision
        self.device = device
        self.float = float_precision
        # Optional base model
        self.base = base

        self.parse_config(config)
        self.instantiate()
        self.is_model = True

    def parse_config(self, config):
        if config is None:
            config = OmegaConf.create()

        self.backbone_args = config.get("backbone_args", {})
        self.leaf_head_args = config.get("leaf_head_args", {})
        self.feature_head_args = config.get("feature_head_args", {})
        self.threshold_head_args = config.get("threshold_head_args", {})
        self.operator_head_args = config.get("operator_head_args", {})

    def instantiate(self):
        self.model = TreeModel(
            backbone=self.base,
            backbone_args=self.backbone_args,
            leaf_head_args=self.leaf_head_args,
            feature_head_args=self.feature_head_args,
            threshold_head_args=self.threshold_head_args,
            operator_head_args=self.operator_head_args,
        ).to(self.device)

    def __call__(self, states):
        return self.model(states)
