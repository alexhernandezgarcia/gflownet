from collections import Counter

import common
import numpy as np
import pytest
import torch

from gflownet.envs.tree import NodeType, Operator, Tree


@pytest.fixture
def X():
    return np.random.rand(17, 5)


@pytest.fixture
def y():
    return np.random.randint(2, size=17)


@pytest.mark.repeat(10)
def test__starting_tree__always_predicts_most_common_class(X, y):
    tree = Tree(X, y)
    x = np.random.random(5)

    assert tree.predict(x) == Counter(y).most_common()[0][0]


@pytest.fixture
def tree(X, y):
    _tree = Tree(X, y)

    # split node k = 0 (root) on feature = 0, with threshold = 0.5 and < operator
    _tree.step((0, 0))
    _tree.step((1, 0))
    _tree.step((2, 0.5))
    _tree.step((3, Operator.LT))

    # split node k = 2 (right child of root) on feature = 1, with threshold = 0.3 and >= operator
    _tree.step((0, 2))
    _tree.step((1, 1))
    _tree.step((2, 0.3))
    _tree.step((3, Operator.GTE))

    # split node k = 6 (right child of right child of root) on feature = 2, with threshold = 0.7 and < operator
    _tree.step((0, 6))
    _tree.step((1, 2))
    _tree.step((2, 0.7))
    _tree.step((3, Operator.LT))

    return _tree


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
def test__predict__has_expected_output(tree, x, output):
    assert tree.predict(x) == output


def test__node_tree__has_expected_node_attributes(tree):
    assert np.allclose(tree._get_attributes(0), [NodeType.CONDITION, 0, 0.5, -1, 0])
    assert np.allclose(tree._get_attributes(1), [NodeType.CLASSIFIER, -1, -1, 0, 0])
    assert np.allclose(tree._get_attributes(2), [NodeType.CONDITION, 1, 0.3, -1, 0])
    assert torch.all(torch.isnan(tree._get_attributes(3)))
    assert torch.all(torch.isnan(tree._get_attributes(4)))
    assert np.allclose(tree._get_attributes(5), [NodeType.CLASSIFIER, -1, -1, 1, 0])
    assert np.allclose(tree._get_attributes(6), [NodeType.CONDITION, 2, 0.7, -1, 0])
    assert np.allclose(tree._get_attributes(13), [NodeType.CLASSIFIER, -1, -1, 0, 0])
    assert np.allclose(tree._get_attributes(14), [NodeType.CLASSIFIER, -1, -1, 1, 0])


@pytest.fixture
def env(X, y):
    return Tree(X, y)


def test__get_parents__returns_no_parents_in_initial_state(env):
    common.test__get_parents__returns_no_parents_in_initial_state(env)


def test__get_parents__returns_same_state_and_eos_if_done(env):
    common.test__get_parents__returns_same_state_and_eos_if_done(env)


def test__step__returns_same_state_action_and_invalid_if_done(env):
    common.test__step__returns_same_state_action_and_invalid_if_done(env)


def test__actions2indices__returns_expected_tensor(env):
    common.test__actions2indices__returns_expected_tensor(env)
