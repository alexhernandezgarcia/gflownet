import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score

from gflownet.envs.tree import ActionType, Attribute, Tree
from gflownet.proxy.tree import TreeProxy


@pytest.fixture
def X():
    return np.random.rand(17, 5)


@pytest.fixture
def y():
    return np.ones(17)


def test__tree_proxy__returns_expected_energies(X, y):
    empty_tree = Tree(X, y, max_depth=4, scale_data=False)
    full_tree = Tree(X, y, max_depth=4, scale_data=False)
    while True:
        splittable_leafs = [
            l
            for l in Tree._find_leaves(full_tree.state)
            if Tree._get_right_child(l) < full_tree.n_nodes
        ]

        if len(splittable_leafs) == 0:
            break

        def split_leaf(k: int):
            cls = full_tree.state[k, Attribute.CLASS].long().item()
            full_tree.step((ActionType.PICK_LEAF, k, cls))
            full_tree.step((ActionType.PICK_FEATURE, -1, np.random.randint(5)))
            full_tree.step((ActionType.PICK_THRESHOLD, -1, np.random.rand()))
            full_tree.step((ActionType.PICK_OPERATOR, k, np.random.randint(2)))

        split_leaf(np.random.choice(splittable_leafs))

    proxy = TreeProxy(use_prior=False, device="cpu", float_precision=32)
    proxy.setup(empty_tree)

    predictions = [Tree.predict(full_tree.state, x) for x in X]
    accuracy = accuracy_score(y, predictions)

    states = torch.stack([empty_tree.state, full_tree.state])
    energies = proxy(states)

    assert torch.equal(energies, torch.Tensor([-1.0, -accuracy]))
