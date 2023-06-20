import numpy as np
import pytest

from gflownet.envs.random_forest import Operator, Node


@pytest.mark.repeat(10)
@pytest.mark.parametrize("output", [0, 1])
def test__basic_node_tree__always_predicts_output(output):
    node_tree = Node(output=output)
    x = np.random.random(5)

    assert node_tree.predict(x) == output


@pytest.fixture
def node_tree():
    node_tree = Node(1)
    node_tree.split(0, 0.5, Operator.LT)
    node_tree.right.split(1, 0.3, Operator.GTE)
    node_tree.right.right.split(2, 0.7, Operator.LT)

    return node_tree


@pytest.mark.parametrize(
    "x, output",
    [
        ([0, 0, 0], 0),
        ([0.4, 1, 1], 0),
        ([0.7, 0.1, 0.7], 1),
        ([0.7, 0.8, 0.3], 0),
        ([0.7, 0.8, 0.9], 1),
    ],
)
def test__node_tree__has_expected_output(node_tree, x, output):
    assert node_tree.predict(x) == output


def test__node_tree__has_expected_node_attributes(node_tree):
    assert np.allclose(node_tree.attributes(), [1, 0, 0.5, -1])
    assert np.allclose(node_tree.left.attributes(), [0, -1, -1, 0])
    assert np.allclose(node_tree.right.attributes(), [1, 1, 0.3, -1])
    assert np.allclose(node_tree.right.left.attributes(), [0, -1, -1, 1])
    assert np.allclose(node_tree.right.right.attributes(), [1, 2, 0.7, -1])
    assert np.allclose(node_tree.right.right.left.attributes(), [0, -1, -1, 0])
    assert np.allclose(node_tree.right.right.right.attributes(), [0, -1, -1, 1])
