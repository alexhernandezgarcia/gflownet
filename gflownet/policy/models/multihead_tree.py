from typing import Optional

import torch
import torch_geometric
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import unbatch

from gflownet.envs.tree import Attribute, Stage, Tree
from gflownet.policy.base import Policy


class Backbone(torch.nn.Module):
    """
    GNN backbone: a stack of GNN layers that can be used for processing graphs.
    """

    def __init__(
        self,
        input_dim: int,
        n_layers: int = 3,
        hidden_dim: int = 64,
        layer: str = "GCNConv",
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
                    layer(input_dim if i == 0 else hidden_dim, hidden_dim),
                    "x, edge_index -> x",
                )
            )
            layers.append(activation())

        self.model = torch_geometric.nn.Sequential("x, edge_index, batch", layers)

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        return self.model(x, edge_index, batch)


class LeafSelectionHead(torch.nn.Module):
    """
    Node-level prediction head. Consists of a stack of GNN layers, and if ``model_eos``
    is True, a separate linear layer for modeling the exit action.

    Note that in the forward function a conversion from the node-level predictions to an
    expected vector policy output is being done. Because of that, the output is a
    regular tensor (with logits at correct positions, regardless of the graph shape).
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        max_nodes: int,
        model_eos: bool = True,
        n_layers: int = 2,
        hidden_dim: int = 64,
        layer: str = "GCNConv",
        activation: str = "LeakyReLU",
    ):
        super().__init__()

        self.max_nodes = max_nodes
        self.model_eos = model_eos

        layer = getattr(torch_geometric.nn, layer)
        activation = getattr(torch.nn, activation)

        body_layers = []
        for i in range(n_layers - 1):
            body_layers.append(
                (
                    layer(backbone.hidden_dim if i == 0 else hidden_dim, hidden_dim),
                    "x, edge_index -> x",
                )
            )
            body_layers.append(activation())

        self.backbone = backbone
        self.body = torch_geometric.nn.Sequential("x, edge_index, batch", body_layers)
        self.leaf_head_layer = layer(hidden_dim, 2)

        if model_eos:
            self.eos_head_layers = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data: torch_geometric.data.Data) -> (torch.Tensor, torch.Tensor):
        x, edge_index, k, batch = (
            data.x,
            data.edge_index,
            data.k,
            data.batch,
        )
        x = self.backbone(data)
        x = self.body(x, edge_index, batch)

        leaf_logits = self.leaf_head_layer(x, edge_index)
        if batch is None:
            y_leaf = torch.full((1, self.max_nodes, 2), torch.nan)
            y_leaf[0, k] = leaf_logits
            # TODO: quadruple-check that flattening is done in a right order
            y_leaf = y_leaf.flatten()
        else:
            logits_batch = unbatch(leaf_logits, batch)
            k_batch = unbatch(k, batch)
            y_leaf = torch.full((len(logits_batch), self.max_nodes, 2), torch.nan)
            for i, (logits_i, k_i) in enumerate(zip(logits_batch, k_batch)):
                y_leaf[i, k_i] = logits_i
            # TODO: quadruple-check that flattening is done in a right order
            y_leaf = y_leaf.reshape((len(logits_batch), -1))

        if not self.model_eos:
            return y_leaf

        x_pool = global_add_pool(x, batch)
        y_eos = self.eos_head_layers(x_pool)[:, 0]

        return y_leaf, y_eos


def _construct_node_head(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_layers: int,
    activation: str,
) -> torch.nn.Module:
    """
    A helper for constructing an MLP.
    """
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

    return torch.nn.Sequential(*layers)


class FeatureSelectionHead(torch.nn.Module):
    """
    A graph-level prediction head that pools the representations from the
    backbone, and passes them through an MLP.

    Expected to have the output dimensionality equal to the number of
    available features.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        input_dim: int,
        output_dim: int,
        n_layers: int = 2,
        hidden_dim: int = 64,
        activation: str = "LeakyReLU",
    ):
        super().__init__()

        self.backbone = backbone
        self.model = _construct_node_head(
            input_dim,
            hidden_dim,
            output_dim,
            n_layers,
            activation,
        )

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, batch = (data.x, data.edge_index, data.batch)
        x = self.backbone(data)
        x = global_add_pool(x, batch)
        x = self.model(x)

        return x


class ThresholdSelectionHead(torch.nn.Module):
    """
    A graph-level prediction head that pools the representations from the
    backbone, and passes them through an MLP.

    Expected to have output dimensionality equal to the number of available
    features plus one, with the last element being the features that were
    selected in the previous stage (which are concatenated with the pooled
    graph representation).
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        input_dim: int,
        output_dim: int,
        n_layers: int = 2,
        hidden_dim: int = 64,
        activation: str = "LeakyReLU",
    ):
        super().__init__()

        self.backbone = backbone
        self.model = _construct_node_head(
            input_dim,
            hidden_dim,
            output_dim,
            n_layers,
            activation,
        )

    def forward(
        self,
        data: torch_geometric.data.Data,
        feature_index: torch.Tensor,
    ) -> torch.Tensor:
        x, edge_index, batch = (data.x, data.edge_index, data.batch)
        x = self.backbone(data)
        x_pool = global_add_pool(x, batch)
        x = torch.cat([x_pool, feature_index.unsqueeze(-1)], dim=1)
        x = self.model(x)

        return x


class OperatorSelectionHead(torch.nn.Module):
    """
    A graph-level prediction head that pools the representations from the
    backbone, and passes them through an MLP.

    Expected to have output dimensionality equal to the number of available
    features plus two, with the last two elements being the features and the
    thresholds that were selected in the previous stage (which are
    concatenated with the pooled graph representation).
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        input_dim: int,
        n_layers: int = 2,
        hidden_dim: int = 64,
        activation: str = "LeakyReLU",
    ):
        super().__init__()

        self.backbone = backbone
        self.model = _construct_node_head(
            input_dim,
            hidden_dim,
            2,
            n_layers,
            activation,
        )

    def forward(
        self,
        data: torch_geometric.data.Data,
        feature_index: torch.Tensor,
        threshold: torch.Tensor,
    ) -> torch.Tensor:
        x, edge_index, batch = (data.x, data.edge_index, data.batch)
        x = self.backbone(data)
        x_pool = global_add_pool(x, batch)
        x = torch.cat(
            [x_pool, feature_index.unsqueeze(-1), threshold.unsqueeze(-1)],
            dim=1,
        )
        x = self.model(x)

        return x


class ForwardTreeModel(torch.nn.Module):
    """
    A model that combines the backbone and several output heads, which
    will be used depending on the current stage of the passed state.
    """

    def __init__(
        self,
        continuous: bool,
        n_features: int,
        policy_output_dim: int,
        leaf_index: int,
        feature_index: int,
        threshold_index: int,
        operator_index: int,
        eos_index: int,
        base: Optional["MultiheadTreePolicy"] = None,
        backbone_args: Optional[dict] = None,
        leaf_head_args: Optional[dict] = None,
        feature_head_args: Optional[dict] = None,
        threshold_head_args: Optional[dict] = None,
        operator_head_args: Optional[dict] = None,
    ):
        super().__init__()

        self.continuous = continuous
        self.n_features = n_features
        self.policy_output_dim = policy_output_dim
        self.leaf_index = leaf_index
        self.feature_index = feature_index
        self.threshold_index = threshold_index
        self.operator_index = operator_index
        self.eos_index = eos_index

        if base is None:
            self.backbone = Backbone(**backbone_args)
        else:
            self.backbone = base.model.backbone

        self.leaf_head = LeafSelectionHead(backbone=self.backbone, **leaf_head_args)
        self.feature_head = FeatureSelectionHead(
            backbone=self.backbone,
            input_dim=self.backbone.hidden_dim,
            **feature_head_args,
        )
        self.threshold_head = ThresholdSelectionHead(
            backbone=self.backbone,
            input_dim=(self.backbone.hidden_dim + 1),
            **threshold_head_args,
        )
        self.operator_head = OperatorSelectionHead(
            backbone=self.backbone,
            input_dim=(self.backbone.hidden_dim + 2),
            **operator_head_args,
        )

    def forward(self, x):
        logits = torch.full((x.shape[0], self.policy_output_dim), torch.nan)
        stages = x[:, -1, 0]

        for stage in stages.unique():
            indices = stages == stage
            states = x[indices]

            batch = Batch.from_data_list(
                [Tree.state2pyg(state, self.n_features) for state in states]
            )

            if stage == Stage.COMPLETE:
                y_leaf, y_eos = self.leaf_head(batch)
                logits[indices, self.leaf_index : self.feature_index] = y_leaf
                logits[indices, self.eos_index] = y_eos
            elif stage == Stage.LEAF:
                logits[indices, self.feature_index : self.threshold_index] = (
                    self.feature_head(batch)
                )
            else:
                ks = [Tree.find_active(state) for state in states]
                feature_index = torch.Tensor(
                    [states[i, k_i, Attribute.FEATURE] for i, k_i in enumerate(ks)]
                )

                if stage == Stage.FEATURE:
                    head_output = self.threshold_head(
                        batch,
                        feature_index,
                    )
                    if self.continuous:
                        logits[indices, (self.eos_index + 1) :] = head_output
                    else:
                        logits[indices, self.threshold_index : self.operator_index] = (
                            head_output
                        )
                elif stage == Stage.THRESHOLD:
                    threshold = torch.Tensor(
                        [
                            states[i, k_i, Attribute.THRESHOLD]
                            for i, k_i in enumerate(ks)
                        ]
                    )
                    head_output = self.operator_head(
                        batch,
                        feature_index,
                        threshold,
                    )

                    indices = torch.nonzero(indices).squeeze(-1)

                    for i, k_i in enumerate(ks):
                        logits[
                            indices[i],
                            (self.operator_index + 2 * k_i) : (
                                self.operator_index + 2 * (k_i + 1)
                            ),
                        ] = head_output[i]
                else:
                    raise ValueError(f"Unrecognized stage = {stage}.")

        return logits


class BackwardTreeModel(torch.nn.Module):
    """
    A model that combines the backbone and several output heads, which
    will be used depending on the current stage of the passed state.

    In contrast to the ForwardTreeModel has less output heads, as some
    of the backward transitions are deterministic.
    """

    def __init__(
        self,
        continuous: bool,
        n_features: int,
        policy_output_dim: int,
        leaf_index: int,
        feature_index: int,
        threshold_index: int,
        operator_index: int,
        eos_index: int,
        base: Optional["MultiheadTreePolicy"] = None,
        backbone_args: Optional[dict] = None,
        leaf_head_args: Optional[dict] = None,
    ):
        super().__init__()

        self.continuous = continuous
        self.n_features = n_features
        self.policy_output_dim = policy_output_dim
        self.leaf_index = leaf_index
        self.feature_index = feature_index
        self.threshold_index = threshold_index
        self.operator_index = operator_index
        self.eos_index = eos_index

        if base is None:
            self.backbone = Backbone(**backbone_args)
        else:
            self.backbone = base.model.backbone

        self.complete_stage_head = LeafSelectionHead(
            backbone=self.backbone, model_eos=False, **leaf_head_args
        )
        self.leaf_stage_head = LeafSelectionHead(
            backbone=self.backbone, model_eos=False, **leaf_head_args
        )

    def forward(self, x):
        logits = torch.full((x.shape[0], self.policy_output_dim), torch.nan)
        stages = x[:, -1, 0]

        for stage in stages.unique():
            indices = stages == stage
            states = x[indices]

            batch = Batch.from_data_list(
                [Tree.state2pyg(state, self.n_features) for state in states]
            )

            if stage == Stage.COMPLETE:
                logits[indices, self.operator_index : self.eos_index] = (
                    self.complete_stage_head(batch)
                )
                logits[indices, self.eos_index] = 1.0
            elif stage == Stage.LEAF:
                logits[indices, self.leaf_index : self.feature_index] = (
                    self.leaf_stage_head(batch)
                )
            elif stage == Stage.FEATURE:
                logits[indices, self.feature_index : self.threshold_index] = 1.0
            elif stage == Stage.THRESHOLD:
                if self.continuous:
                    logits[indices, (self.eos_index + 1) :] = 1.0
                else:
                    logits[indices, self.threshold_index : self.operator_index] = 1.0
            else:
                raise ValueError(f"Unrecognized stage = {stage}.")

        return logits


class MultiheadTreePolicy(Policy):
    """
    Policy wrapper using ForwardTreeModel and BackwardTreeModel as the policy models.
    """

    def __init__(self, config, env, device, float_precision, base=None):
        self.backbone_args = {"input_dim": env.get_pyg_input_dim()}
        self.leaf_head_args = {"max_nodes": env.n_nodes}
        self.feature_head_args = {"output_dim": env.X_train.shape[1]}
        if env.continuous:
            self.threshold_head_args = {"output_dim": env.components * 3}
        else:
            self.threshold_head_args = {"output_dim": len(env.thresholds)}
        self.operator_head_args = {}
        self.continuous = env.continuous
        self.n_features = env.n_features
        self.policy_output_dim = env.policy_output_dim
        self.leaf_index = env._action_index_pick_leaf
        self.feature_index = env._action_index_pick_feature
        self.threshold_index = env._action_index_pick_threshold
        self.operator_index = env._action_index_pick_operator
        self.eos_index = env._action_index_eos

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
            self.leaf_head_args.update(config.get("leaf_head_args", {}))
            self.feature_head_args.update(config.get("feature_head_args", {}))
            self.threshold_head_args.update(config.get("threshold_head_args", {}))
            self.operator_head_args.update(config.get("operator_head_args", {}))

    def instantiate(self):
        if self.base is None:
            self.model = ForwardTreeModel(
                continuous=self.continuous,
                n_features=self.n_features,
                policy_output_dim=self.policy_output_dim,
                leaf_index=self.leaf_index,
                feature_index=self.feature_index,
                threshold_index=self.threshold_index,
                operator_index=self.operator_index,
                eos_index=self.eos_index,
                base=self.base,
                backbone_args=self.backbone_args,
                leaf_head_args=self.leaf_head_args,
                feature_head_args=self.feature_head_args,
                threshold_head_args=self.threshold_head_args,
                operator_head_args=self.operator_head_args,
            ).to(self.device)
        else:
            self.model = BackwardTreeModel(
                continuous=self.continuous,
                n_features=self.n_features,
                policy_output_dim=self.policy_output_dim,
                leaf_index=self.leaf_index,
                feature_index=self.feature_index,
                threshold_index=self.threshold_index,
                operator_index=self.operator_index,
                eos_index=self.eos_index,
                base=self.base,
                backbone_args=self.backbone_args,
                leaf_head_args=self.leaf_head_args,
            ).to(self.device)

    def __call__(self, states):
        return self.model(states)
