from collections import Counter

import common
import numpy as np
import pytest
import torch

from gflownet.envs.tree import ActionType, NodeType, Operator, Stage, Status, Tree
from gflownet.utils.common import tfloat

NAN = float("NaN")


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
def tree_d1(X, y):
    return Tree(
        X=np.array([[1, 2], [3, 4], [5, 6]]), y=np.array([0, 0, 1]), max_depth=2
    )


def step_return_expected(env, action, state_expected, valid_expected=True):
    state_expected = tfloat(state_expected, float_type=env.float, device=env.device)
    state_next, action_done, valid = env.step(action)
    assert valid == valid_expected
    if valid is True:
        assert action_done == action
        assert env.equal(state_next, env.state)
        assert env.equal(state_next, state_expected)


def state_action_are_in_parents(env, state, action):
    state = tfloat(state, float_type=env.float, device=env.device)
    parents, parents_a = env.get_parents()
    assert any([env.equal(p, state) for p in parents])
    assert env.action2representative(action) in parents_a


@pytest.mark.parametrize(
    "source, a0, s1, a1, s2, a2, s3, a3, s4, a4, s5",
    [
        (
            [
                Stage.COMPLETE,
                NodeType.CLASSIFIER,
                -1,
                -1,
                0,
                Status.INACTIVE,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
            ],
            (ActionType.PICK_LEAF, 0),
            [
                Stage.LEAF,
                NodeType.CONDITION,
                -1,
                -1,
                -1,
                Status.ACTIVE,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
            ],
            (ActionType.PICK_FEATURE, 1),
            [
                Stage.FEATURE,
                NodeType.CONDITION,
                1,
                -1,
                -1,
                Status.ACTIVE,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
            ],
            (ActionType.PICK_THRESHOLD, 0.2),
            [
                Stage.THRESHOLD,
                NodeType.CONDITION,
                1,
                0.2,
                -1,
                Status.ACTIVE,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
                NAN,
            ],
            (ActionType.PICK_OPERATOR, Operator.LT),
            [
                Stage.COMPLETE,
                NodeType.CONDITION,
                1,
                0.2,
                -1,
                Status.INACTIVE,
                NodeType.CLASSIFIER,
                -1,
                -1,
                0,
                Status.INACTIVE,
                NodeType.CLASSIFIER,
                -1,
                -1,
                1,
                Status.INACTIVE,
            ],
            (-1, -1),
            [
                Stage.COMPLETE,
                NodeType.CONDITION,
                1,
                0.2,
                -1,
                Status.INACTIVE,
                NodeType.CLASSIFIER,
                -1,
                -1,
                0,
                Status.INACTIVE,
                NodeType.CLASSIFIER,
                -1,
                -1,
                1,
                Status.INACTIVE,
            ],
        ),
    ],
)
def test__steps__behaves_as_expected(
    tree_d1, source, a0, s1, a1, s2, a2, s3, a3, s4, a4, s5
):
    source = tfloat(source, float_type=tree_d1.float, device=tree_d1.device)
    assert tree_d1.equal(tree_d1.source, source)
    # Action 0, state 1
    step_return_expected(tree_d1, a0, s1, True)
    state_action_are_in_parents(tree_d1, source, a0)
    # Action 1, state 2
    step_return_expected(tree_d1, a1, s2, True)
    state_action_are_in_parents(tree_d1, s1, a1)
    # Action 2, state 3
    step_return_expected(tree_d1, a2, s3, True)
    state_action_are_in_parents(tree_d1, s2, a2)
    # Action 3, state 4
    step_return_expected(tree_d1, a3, s4, True)
    state_action_are_in_parents(tree_d1, s3, a3)
    # Action 4, state 5
    step_return_expected(tree_d1, a4, s5, True)
    state_action_are_in_parents(tree_d1, s4, a4)
    assert tree_d1.done is True


@pytest.fixture
def env(X, y):
    return Tree(X, y)


# def test__continuous_env_common(env):
#     return common.test__continuous_env_common(env)
#
#
# def test__all_env_common(env):
#     return common.test__all_env_common(env)