import numpy as np
import pytest

from gflownet.envs.random_forest import Operator, Node


@pytest.mark.repeat(10)
@pytest.mark.parametrize("output", [0, 1])
def test__basic_tree__always_predicts_output(output):
    tree = Node(output=output)
    x = np.random.random(5)

    assert tree.predict(x) == output


@pytest.fixture
def tree():
    tree = Node(1)
    tree.split(0, 0.5, Operator.LT)
    tree.right.split(1, 0.3, Operator.GTE)
    tree.right.right.split(2, 0.7, Operator.LT)

    return tree


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
def test__tree__has_expected_output(tree, x, output):
    assert tree.predict(x) == output


def test__tree__has_expected_node_attributes(tree):
    assert np.array_equal(tree.attributes(), [1, 0, 0.5, -1])
    assert np.array_equal(tree.left.attributes(), [0, -1, -1, 0])
    assert np.array_equal(tree.right.attributes(), [1, 1, 0.3, -1])
    assert np.array_equal(tree.right.left.attributes(), [0, -1, -1, 1])
    assert np.array_equal(tree.right.right.attributes(), [1, 2, 0.7, -1])
    assert np.array_equal(tree.right.right.left.attributes(), [0, -1, -1, 0])
    assert np.array_equal(tree.right.right.right.attributes(), [0, -1, -1, 1])
