from typing import Optional

import torch
import torch_geometric
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

from gflownet.envs.tree import Tree
from gflownet.policy.base import Policy


class Backbone(torch.nn.Module):
    def __init__(
        self,
        n_layers: int = 3,
        hidden_dim: int = 128,
        input_dim: int = 5,
        layer: str = "GINEConv",
        activation: str = "LeakyReLU",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        layer = getattr(torch_geometric.nn, layer)
        activation = getattr(torch.nn, activation)

        layers = []
        for i in range(n_layers):
            layers.append(
                (
                    layer(
                        torch.nn.Linear(
                            input_dim if i == 0 else hidden_dim, hidden_dim
                        ),
                        edge_dim=2,
                    ),
                    "x, edge_index, edge_attr -> x",
                )
            )
            layers.append(activation())

        self.model = torch_geometric.nn.Sequential(
            "x, edge_index, edge_attr, batch", layers
        )

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        return self.model(x, edge_index, edge_attr, batch)


class Head(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        out_dim: int,
        n_layers: int = 2,
        hidden_dim: int = 256,
        activation: str = "LeakyReLU",
    ):
        super().__init__()

        self.backbone = backbone

        activation = getattr(torch.nn, activation)

        layers = []
        for i in range(n_layers):
            layers.append(
                torch.nn.Linear(
                    backbone.hidden_dim if i == 0 else hidden_dim,
                    out_dim if i == n_layers - 1 else hidden_dim,
                )
            )
            if i < n_layers - 1:
                layers.append(activation())

        self.head = torch.nn.Sequential(*layers)

    def forward(self, data: torch_geometric.data.Data) -> (torch.Tensor, torch.Tensor):
        x, edge_index, edge_attr, k, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.k,
            data.batch,
        )
        x = self.backbone(data)
        x = global_add_pool(x, batch)
        x = self.head(x)

        return x


class SimpleTreeModel(torch.nn.Module):
    def __init__(
        self,
        policy_output_dim: int,
        base: Optional["TreePolicy"] = None,
        backbone_args: Optional[dict] = None,
        head_args: Optional[dict] = None,
    ):
        super().__init__()

        self.policy_output_dim = policy_output_dim

        if base is None:
            self.backbone = Backbone(**backbone_args)
        else:
            self.backbone = base.model.backbone

        self.model = Head(
            backbone=self.backbone, out_dim=policy_output_dim, **head_args
        )

    def forward(self, x):
        batch = Batch.from_data_list([Tree.to_pyg(state) for state in x])
        return self.model(batch)


class SimpleTreePolicy(Policy):
    def __init__(self, config, env, device, float_precision, base=None):
        self.backbone_args = {}
        self.head_args = {}
        self.policy_output_dim = env.policy_output_dim

        super().__init__(
            config=config,
            env=env,
            device=device,
            float_precision=float_precision,
            base=base,
        )

        self.is_model = True

    def parse_config(self, config):
        if config is not None:
            self.backbone_args.update(config.get("backbone_args", {}))
            self.head_args.update(config.get("head_args", {}))

    def instantiate(self):
        self.model = SimpleTreeModel(
            policy_output_dim=self.policy_output_dim,
            base=self.base,
            backbone_args=self.backbone_args,
            head_args=self.head_args,
        ).to(self.device)

    def __call__(self, states):
        return self.model(states)
