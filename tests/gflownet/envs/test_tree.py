from collections import Counter

import common
import numpy as np
import pytest
import torch

from gflownet.envs.tree import (
    ActionType,
    Attribute,
    NodeType,
    Operator,
    Stage,
    Status,
    Tree,
)
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

    assert tree._predict(x) == Counter(y).most_common()[0][0]


@pytest.fixture
def tree(X, y):
    _tree = Tree(X, y)

    # split node k = 0 (root) on feature = 0, with threshold = 0.5 and < operator
    _tree.step((0, 0, _tree.default_class))
    _tree.step((1, -1, 0))
    _tree.step((2, -1, 0.5))
    _tree.step((3, 0, Operator.LT))

    # split node k = 2 (right child of root) on feature = 1, with threshold = 0.3 and >= operator
    _tree.step((0, 2, _tree.state[2, Attribute.CLASS]))
    _tree.step((1, -1, 1))
    _tree.step((2, -1, 0.3))
    _tree.step((3, 2, Operator.GTE))

    # split node k = 6 (right child of right child of root) on feature = 2, with threshold = 0.7 and < operator
    _tree.step((0, 6, _tree.state[6, Attribute.CLASS]))
    _tree.step((1, -1, 2))
    _tree.step((2, -1, 0.7))
    _tree.step((3, 6, Operator.LT))

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
    assert tree._predict(x) == output


def test__node_tree__has_expected_node_attributes(tree):
    assert np.allclose(tree.state[0], [NodeType.CONDITION, 0, 0.5, -1, 0])
    assert np.allclose(tree.state[1], [NodeType.CLASSIFIER, -1, -1, 0, 0])
    assert np.allclose(tree.state[2], [NodeType.CONDITION, 1, 0.3, -1, 0])
    assert torch.all(torch.isnan(tree.state[3]))
    assert torch.all(torch.isnan(tree.state[4]))
    assert np.allclose(tree.state[5], [NodeType.CLASSIFIER, -1, -1, 1, 0])
    assert np.allclose(tree.state[6], [NodeType.CONDITION, 2, 0.7, -1, 0])
    assert np.allclose(tree.state[13], [NodeType.CLASSIFIER, -1, -1, 0, 0])
    assert np.allclose(tree.state[14], [NodeType.CLASSIFIER, -1, -1, 1, 0])


@pytest.fixture
def tree_d2(X, y):
    return Tree(
        X_train=np.array([[1, 2], [3, 4], [5, 6]]),
        y_train=np.array([0, 0, 1]),
        max_depth=2,
    )


@pytest.fixture
def tree_d3(X, y):
    return Tree(
        X_train=np.array([[1, 2], [3, 4], [5, 6]]),
        y_train=np.array([0, 0, 1]),
        max_depth=3,
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
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.COMPLETE, NAN, NAN, NAN, NAN],
            ],
            (ActionType.PICK_LEAF, 0, 0),
            [
                [NodeType.CONDITION, -1, -1, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.LEAF, NAN, NAN, NAN, NAN],
            ],
            (ActionType.PICK_FEATURE, -1, 1),
            [
                [NodeType.CONDITION, 1, -1, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.FEATURE, NAN, NAN, NAN, NAN],
            ],
            (ActionType.PICK_THRESHOLD, -1, 0.2),
            [
                [NodeType.CONDITION, 1, 0.2, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.THRESHOLD, NAN, NAN, NAN, NAN],
            ],
            (ActionType.PICK_OPERATOR, 0, Operator.LT),
            [
                [NodeType.CONDITION, 1, 0.2, -1, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 1, Status.INACTIVE],
                [Stage.COMPLETE, NAN, NAN, NAN, NAN],
            ],
            (-1, -1, -1),
            [
                [NodeType.CONDITION, 1, 0.2, -1, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 1, Status.INACTIVE],
                [Stage.COMPLETE, NAN, NAN, NAN, NAN],
            ],
        ),
    ],
)
def test__steps__behaves_as_expected_d2(
    tree_d2, source, a0, s1, a1, s2, a2, s3, a3, s4, a4, s5
):
    source = tfloat(source, float_type=tree_d2.float, device=tree_d2.device)
    assert tree_d2.equal(tree_d2.source, source)
    # Action 0, state 1
    step_return_expected(tree_d2, a0, s1, True)
    state_action_are_in_parents(tree_d2, source, a0)
    # Action 1, state 2
    step_return_expected(tree_d2, a1, s2, True)
    state_action_are_in_parents(tree_d2, s1, a1)
    # Action 2, state 3
    step_return_expected(tree_d2, a2, s3, True)
    state_action_are_in_parents(tree_d2, s2, a2)
    # Action 3, state 4
    step_return_expected(tree_d2, a3, s4, True)
    state_action_are_in_parents(tree_d2, s3, a3)
    # Action 4, state 5
    step_return_expected(tree_d2, a4, s5, True)
    state_action_are_in_parents(tree_d2, s4, a4)
    assert tree_d2.done is True


@pytest.mark.parametrize(
    "source, a0, s1, a1, s2, a2, s3, a3, s4, a4, s5",
    [
        (
            [
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.COMPLETE, NAN, NAN, NAN, NAN],
            ],
            (ActionType.PICK_LEAF, 0, 0),
            [
                [NodeType.CONDITION, -1, -1, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.LEAF, NAN, NAN, NAN, NAN],
            ],
            (ActionType.PICK_FEATURE, -1, 1),
            [
                [NodeType.CONDITION, 1, -1, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.FEATURE, NAN, NAN, NAN, NAN],
            ],
            (ActionType.PICK_THRESHOLD, -1, 0.2),
            [
                [NodeType.CONDITION, 1, 0.2, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.THRESHOLD, NAN, NAN, NAN, NAN],
            ],
            (ActionType.PICK_OPERATOR, 0, Operator.LT),
            [
                [NodeType.CONDITION, 1, 0.2, -1, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 1, Status.INACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.COMPLETE, NAN, NAN, NAN, NAN],
            ],
            (0, 2, 1),
            [
                [NodeType.CONDITION, 1, 0.2, -1, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [NodeType.CONDITION, -1, -1, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.LEAF, NAN, NAN, NAN, NAN],
            ],
        ),
    ],
)
def test__steps__behaves_as_expected_d3(
    tree_d3, source, a0, s1, a1, s2, a2, s3, a3, s4, a4, s5
):
    source = tfloat(source, float_type=tree_d3.float, device=tree_d3.device)
    assert tree_d3.equal(tree_d3.source, source)
    # Action 0, state 1
    step_return_expected(tree_d3, a0, s1, True)
    state_action_are_in_parents(tree_d3, source, a0)
    # Action 1, state 2
    step_return_expected(tree_d3, a1, s2, True)
    state_action_are_in_parents(tree_d3, s1, a1)
    # Action 2, state 3
    step_return_expected(tree_d3, a2, s3, True)
    state_action_are_in_parents(tree_d3, s2, a2)
    # Action 3, state 4
    step_return_expected(tree_d3, a3, s4, True)
    state_action_are_in_parents(tree_d3, s3, a3)
    # Action 4, state 5
    step_return_expected(tree_d3, a4, s5, True)
    state_action_are_in_parents(tree_d3, s4, a4)


# Action space
# [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
#  (1, 0), (1, 1),
#  (2, -1),
#  (3, 0), (3, 1),
#  (-1, -1)]
@pytest.mark.parametrize(
    "state, m_leaf_exp, m_feat_exp, m_th_exp, m_op_exp, m_eos_exp, m_cont",
    [
        (
            [
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.COMPLETE, NAN, NAN, NAN, NAN],
            ],
            [
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True, True],
            [True],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [False],
            [True, True, True],
        ),
        (
            [
                [NodeType.CONDITION, -1, -1, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.LEAF, NAN, NAN, NAN, NAN],
            ],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [False, False],
            [True],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True],
            [True, True, True],
        ),
        (
            [
                [NodeType.CONDITION, 1, -1, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.FEATURE, NAN, NAN, NAN, NAN],
            ],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True, True],
            [False],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True],
            [True, True, True],
        ),
        (
            [
                [NodeType.CONDITION, 1, 0.2, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.THRESHOLD, NAN, NAN, NAN, NAN],
            ],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True, True],
            [True],
            [
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True],
            [True, True, True],
        ),
        (
            [
                [NodeType.CONDITION, 1, 0.2, -1, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 1, Status.INACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.COMPLETE, NAN, NAN, NAN, NAN],
            ],
            [
                True,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True, True],
            [True],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [False],
            [True, True, True],
        ),
        (
            [
                [NodeType.CONDITION, 1, 0.2, -1, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [NodeType.CONDITION, -1, -1, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.LEAF, NAN, NAN, NAN, NAN],
            ],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [False, False],
            [True],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True],
            [True, True, True],
        ),
    ],
)
def test__get_masks_forward__returns_expected(
    tree_d3, state, m_leaf_exp, m_feat_exp, m_th_exp, m_op_exp, m_eos_exp, m_cont
):
    state = tfloat(state, float_type=tree_d3.float, device=tree_d3.device)
    mask = m_leaf_exp + m_feat_exp + m_th_exp + m_op_exp + m_eos_exp + m_cont
    tree_d3.set_state(state, done=False)
    assert tree_d3.get_mask_invalid_actions_forward() == mask


# Action space
# [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
#  (1, 0), (1, 1),
#  (2, -1),
#  (3, 0), (3, 1),
#  (-1, -1)]
@pytest.mark.parametrize(
    "state, m_leaf_exp, m_feat_exp, m_th_exp, m_op_exp, m_eos_exp, m_cont",
    [
        (
            [
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.COMPLETE, NAN, NAN, NAN, NAN],
            ],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True, True],
            [True],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True],
            [True, True, True],
        ),
        (
            [
                [NodeType.CONDITION, -1, -1, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.LEAF, NAN, NAN, NAN, NAN],
            ],
            [
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True, True],
            [True],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True],
            [True, True, True],
        ),
        (
            [
                [NodeType.CONDITION, 1, -1, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.FEATURE, NAN, NAN, NAN, NAN],
            ],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True, False],
            [True],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True],
            [True, True, True],
        ),
        (
            [
                [NodeType.CONDITION, 1, 0.2, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.THRESHOLD, NAN, NAN, NAN, NAN],
            ],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True, True],
            [False],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True],
            [True, True, True],
        ),
        (
            [
                [NodeType.CONDITION, 1, 0.2, -1, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 1, Status.INACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.COMPLETE, NAN, NAN, NAN, NAN],
            ],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True, True],
            [True],
            [
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True],
            [True, True, True],
        ),
        (
            [
                [NodeType.CONDITION, 1, 0.2, -1, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 1, Status.INACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.COMPLETE, NAN, NAN, NAN, NAN],
            ],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True, True],
            [True],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [False],
            [True, True, True],
        ),
        (
            [
                [NodeType.CONDITION, 1, 0.2, -1, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [NodeType.CONDITION, -1, -1, -1, Status.ACTIVE],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [NAN, NAN, NAN, NAN, NAN],
                [Stage.LEAF, NAN, NAN, NAN, NAN],
            ],
            [
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True, True],
            [True],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True],
            [True, True, True],
        ),
        # Both operators are compatible as backward actions
        (
            [
                [NodeType.CONDITION, 1, 0.2, -1, Status.INACTIVE],
                [NodeType.CONDITION, 0, 0.8, -1, Status.INACTIVE],
                [NodeType.CONDITION, 1, 0.3, -1, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 1, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 1, Status.INACTIVE],
                [NodeType.CLASSIFIER, -1, -1, 0, Status.INACTIVE],
                [Stage.COMPLETE, NAN, NAN, NAN, NAN],
            ],
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True, True],
            [True],
            [
                True,
                True,
                False,
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            [True],
            [True, True, True],
        ),
    ],
)
def test__get_masks_backward__returns_expected(
    tree_d3, state, m_leaf_exp, m_feat_exp, m_th_exp, m_op_exp, m_eos_exp, m_cont
):
    state = tfloat(state, float_type=tree_d3.float, device=tree_d3.device)
    mask = m_leaf_exp + m_feat_exp + m_th_exp + m_op_exp + m_eos_exp + m_cont
    tree_d3.set_state(state, done=not mask[tree_d3._action_index_eos])
    assert tree_d3.get_mask_invalid_actions_backward() == mask


@pytest.fixture
def env(X, y):
    return Tree(X, y)


class TestTreeDiscrete(common.BaseTestsDiscrete):
    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
        self.repeats = {
            "test__reset__state_is_source": 10,
        }
        self.n_states = {}  # TODO: Populate.


class TestTreeContinuous(common.BaseTestsContinuous):
    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
        self.repeats = {
            "test__reset__state_is_source": 10,
        }
        self.n_states = {}  # TODO: Populate.
