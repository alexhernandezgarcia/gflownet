"""
Transformer policy for the composite Tree environment.

The composite Tree environment (gflownet.envs.tree.tree.Tree) encodes states
for the policy as a flat vector with a per-node block structure:

    [ idle_flag | node_0_block | node_1_block | ... | node_{max_nodes-1}_block ]

where each node block is [exists, done, active, node_env_policy_encoding] and
node indices follow a heap layout (children of node k are at 2k + 1 and
2k + 2). This module treats each node block as a token, augments it with
learned positional and structural (depth, left/right side) embeddings, and
processes the resulting sequence with a standard pre-LN Transformer encoder.
The policy output is read from a learned CLS token, into which the global
idle flag is injected.

Conceptually inspired by the Set Transformer policy of the legacy DT-GFN
codebase, but implemented with standard PyTorch Transformer modules and
position-aware encoding.
"""

from typing import Optional

import torch
from torch import nn

from gflownet.policy.base import Policy


# Helper function for positional embeddings (return 1D torch tensor of len max_nodes)
def _heap_depths(max_nodes: int) -> torch.Tensor:
    """Depth of each node in the heap layout: depth(k) = floor(log2(k + 1))."""
    return torch.tensor(
        [(k + 1).bit_length() - 1 for k in range(max_nodes)], dtype=torch.long
    )


def _heap_sides(max_nodes: int) -> torch.Tensor:
    """Side of each node: 0 = root, 1 = left child (odd k), 2 = right child."""
    return torch.tensor(
        [0 if k == 0 else (1 if k % 2 == 1 else 2) for k in range(max_nodes)],
        dtype=torch.long,
    )


class TreeTransformerTrunk(nn.Module):
    """
    Encodes the flat Tree policy input into a single CLS representation.

    Tokenizes the input into one token per node position, adds learned
    positional and structural embeddings, prepends a CLS token carrying the
    idle flag, and runs a pre-LN Transformer encoder.
    """

    def __init__(
        self,
        per_node_dim: int,
        max_nodes: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int,
        dropout: float,
    ):
        super().__init__()

        self.per_node_dim = per_node_dim
        self.max_nodes = max_nodes
        self.d_model = d_model

        self.node_proj = nn.Linear(per_node_dim, d_model)
        self.pos_emb = nn.Embedding(max_nodes, d_model)
        n_depths = max_nodes.bit_length()  # depths 0 .. log2(max_nodes + 1) - 1
        self.depth_emb = nn.Embedding(n_depths, d_model)
        self.side_emb = nn.Embedding(3, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.idle_proj = nn.Linear(1, d_model)

        self.register_buffer("depth_index", _heap_depths(max_nodes))
        self.register_buffer("side_index", _heap_sides(max_nodes))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : tensor of shape [batch, 1 + max_nodes * per_node_dim]
            Output of Tree.states2policy.

        Returns
        -------
        tensor of shape [batch, d_model]
            CLS representation of each state.
        """
        batch_size = x.shape[0]
        idle_flag = x[:, :1]  # For whole batch, take the first element
        tokens = x[:, 1:].reshape(batch_size, self.max_nodes, self.per_node_dim)

        h = (
            self.node_proj(tokens)
            + self.pos_emb.weight.unsqueeze(0)
            + self.depth_emb(self.depth_index).unsqueeze(0)
            + self.side_emb(self.side_index).unsqueeze(0)
        )
        cls = self.cls_token.expand(batch_size, -1, -1) + self.idle_proj(
            idle_flag
        ).unsqueeze(1)
        h = torch.cat([cls, h], dim=1)
        h = self.encoder(h)
        return h[:, 0]  # For whole batch, take cls token (at position 0)


class TreeTransformerModel(nn.Module):
    """
    Trunk + linear head. Maps the CLS representation to the output_dim
    expected by the composite Tree environment. Separated from TreeTransformerTrunk
    to allow weight sharing of the trunk by the forward and backward policy (if wanted)
    by just having different linear heads on top of the shared trunk.
    """

    def __init__(self, trunk: TreeTransformerTrunk, output_dim: int):
        super().__init__()
        self.trunk = trunk
        self.head = nn.Linear(trunk.d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:  # Handle empty batches
            return torch.zeros(
                (0, self.head.out_features), dtype=x.dtype, device=x.device
            )
        return self.head(self.trunk(x))


class TreeTransformerPolicy(Policy):
    """
    Adapter between the TreeTransformerModel policy model and the GFlowNet framework. Handles
    loading hyperparams from the .yaml config, loading input/output sizes
    from the gflownet-env etc. And exposes the interface the gflownet trainer
    expects: policy.model, policy.is_model and policy(states).

    If ``shared_weights`` is True (only valid for the backward policy, which
    receives the forward policy as ``base``), the trunk is shared with the
    forward policy and only the output head is trained separately.
    """

    def __init__(self, config, env, device, float_precision, base=None):
        self.max_nodes = env.max_nodes
        self.per_node_dim = 3 + env.node_env.policy_input_dim
        assert env.policy_input_dim == 1 + self.max_nodes * self.per_node_dim, (
            "Tree policy input layout does not match the expected "
            "[idle | per-node blocks] structure."
        )

        super().__init__(
            config=config,
            env=env,
            device=device,
            float_precision=float_precision,
            base=base,
        )

        self.is_model = True

    def parse_config(self, config):
        if config is None:
            config = {}
        self.shared_weights = config.get("shared_weights", False)
        self.d_model = config.get("d_model", 128)
        self.n_layers = config.get("n_layers", 3)
        self.n_heads = config.get("n_heads", 4)
        self.ff_mult = config.get("ff_mult", 4)
        self.dropout = config.get("dropout", 0.0)

    def instantiate(self):
        if self.shared_weights:
            if self.base is None:
                raise ValueError(
                    "Base policy must be provided when shared_weights is True."
                )
            trunk = self.base.model.trunk
        else:
            trunk = TreeTransformerTrunk(
                per_node_dim=self.per_node_dim,
                max_nodes=self.max_nodes,
                d_model=self.d_model,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                ff_mult=self.ff_mult,
                dropout=self.dropout,
            )
        self.model = TreeTransformerModel(trunk, self.output_dim).to(
            device=self.device, dtype=self.float
        )

    def n_parameters(self, trainable_only: bool = True) -> int:
        """
        Returns the number of parameters of the policy model.

        Note that if the trunk is shared with the forward policy
        (``shared_weights: True``), its parameters are included in the count
        of both policies, so summing the counts of the forward and backward
        policies would count the trunk twice.

        Parameters
        ----------
        trainable_only : bool
            If True (default), count only parameters with requires_grad=True.
        """
        return sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad or not trainable_only
        )

    def __call__(self, states):
        return self.model(states)
