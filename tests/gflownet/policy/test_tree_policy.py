import numpy as np
import pytest
import torch

from gflownet.envs.tree import Operator, Tree
from gflownet.policy.tree import (
    Backbone,
    FeatureSelectionHead,
    LeafSelectionHead,
    OperatorSelectionHead,
    ThresholdSelectionHead,
)

N_OBSERVATIONS = 17
N_FEATURES = 5
N_NODES = 11
BACKBONE_HIDDEN_DIM = 64


@pytest.fixture()
def tree(
    n_observations: int = N_OBSERVATIONS,
    n_features: int = N_FEATURES,
    n_nodes: int = N_NODES,
):
    X = np.random.rand(n_observations, n_features)
    y = np.random.randint(2, size=n_observations)

    _tree = Tree(X_train=X, y_train=y)

    for _ in range(n_nodes):
        node = np.random.choice(list(Tree._find_leaves(_tree.state)))
        operator = np.random.choice([Operator.LT, Operator.GTE])
        feature = np.random.randint(n_features)
        threshold = np.random.rand()

        _tree.step((0, node))
        _tree.step((1, feature))
        _tree.step((2, threshold))
        _tree.step((3, operator))

    return _tree


@pytest.fixture()
def data(tree):
    return tree._to_pyg()


@pytest.fixture()
def backbone():
    return Backbone(hidden_dim=BACKBONE_HIDDEN_DIM)


def test__backbone__output_has_correct_shape(data, backbone):
    assert backbone(data).shape == (data.x.shape[0], BACKBONE_HIDDEN_DIM)


def test__leaf_selection__output_has_correct_shape(data, backbone):
    head = LeafSelectionHead(backbone)
    output = head(data)

    assert len(output.shape) == 1
    assert output.shape[0] == data.x.shape[0]


def test__feature_selection__output_has_correct_shape(data, backbone):
    head = FeatureSelectionHead(backbone, output_dim=N_FEATURES)
    output = head.forward(data, node_index=torch.Tensor([0]).long())

    assert len(output.shape) == 2
    assert output.shape[0] == 1
    assert output.shape[1] == N_FEATURES


def test__threshold_selection__output_has_correct_shape(data, backbone):
    input_dim = BACKBONE_HIDDEN_DIM * 2 + 1
    output_dim = 4
    head = ThresholdSelectionHead(backbone, input_dim=input_dim, output_dim=output_dim)
    output = head.forward(
        data, node_index=torch.Tensor([0]).long(), feature_index=torch.Tensor([[0]])
    )

    assert len(output.shape) == 2
    assert output.shape[0] == 1
    assert output.shape[1] == output_dim


def test__operator_selection__output_has_correct_shape(data, backbone):
    input_dim = BACKBONE_HIDDEN_DIM * 2 + 2
    head = OperatorSelectionHead(backbone, input_dim=input_dim)
    output = head.forward(
        data,
        node_index=torch.Tensor([0]).long(),
        feature_index=torch.Tensor([[0]]),
        threshold=torch.Tensor([[0.5]]),
    )

    assert len(output.shape) == 2
    assert output.shape[0] == 1
    assert output.shape[1] == 1
