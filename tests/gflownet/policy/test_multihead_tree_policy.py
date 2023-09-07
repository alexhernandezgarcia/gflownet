import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from gflownet.envs.tree import Attribute, Operator, Tree
from gflownet.policy.multihead_tree import (
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

        _tree.step((0, node, _tree.state[node, Attribute.CLASS]))
        _tree.step((1, -1, feature))
        _tree.step((2, -1, threshold))
        _tree.step((3, node, operator))

    return _tree


@pytest.fixture()
def data(tree):
    return tree._state2pyg()


@pytest.fixture()
def batch(data):
    return Batch.from_data_list([data, data])


@pytest.fixture()
def backbone(tree):
    return Backbone(input_dim=tree.get_pyg_input_dim(), hidden_dim=BACKBONE_HIDDEN_DIM)


def test__backbone__output_has_correct_shape(data, backbone):
    assert backbone(data).shape == (data.x.shape[0], BACKBONE_HIDDEN_DIM)


def test__leaf_selection__output_has_correct_shape_for_graph(tree, data, backbone):
    head = LeafSelectionHead(backbone=backbone, max_nodes=tree.n_nodes)
    node_output, eos_output = head(data)

    assert node_output.shape == (tree.n_nodes * 2,)
    assert eos_output.shape == (1,)


def test__leaf_selection__output_has_correct_shape_for_batch(tree, batch, backbone):
    head = LeafSelectionHead(backbone=backbone, max_nodes=tree.n_nodes)
    node_output, eos_output = head(batch)

    assert node_output.shape == (2, tree.n_nodes * 2)
    assert eos_output.shape == (2,)


def test__feature_selection__output_has_correct_shape(data, backbone):
    head = FeatureSelectionHead(
        backbone, input_dim=BACKBONE_HIDDEN_DIM, output_dim=N_FEATURES
    )
    output = head.forward(data)

    assert len(output.shape) == 2
    assert output.shape[0] == 1
    assert output.shape[1] == N_FEATURES


def test__threshold_selection__output_has_correct_shape(data, backbone):
    input_dim = BACKBONE_HIDDEN_DIM + 1
    output_dim = 4
    head = ThresholdSelectionHead(backbone, input_dim=input_dim, output_dim=output_dim)
    output = head.forward(data, feature_index=torch.Tensor([0]))

    assert len(output.shape) == 2
    assert output.shape[0] == 1
    assert output.shape[1] == output_dim


def test__operator_selection__output_has_correct_shape(data, backbone):
    input_dim = BACKBONE_HIDDEN_DIM + 2
    head = OperatorSelectionHead(backbone, input_dim=input_dim)
    output = head.forward(
        data,
        feature_index=torch.Tensor([0]),
        threshold=torch.Tensor([0.5]),
    )

    assert len(output.shape) == 2
    assert output.shape[0] == 1
    assert output.shape[1] == 2
