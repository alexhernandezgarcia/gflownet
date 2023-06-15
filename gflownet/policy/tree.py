import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric


class Backbone(torch.nn.Module):
    def __init__(
        self,
        n_layers: int = 5,
        hidden_dim: int = 64,
        input_dim: int = 4,
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
