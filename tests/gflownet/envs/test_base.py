from collections import Counter

import numpy as np
import pytest
import torch

from gflownet.envs.grid import Grid


@pytest.fixture
def grid():
    return Grid()


@pytest.fixture
def grid5x5():
    return Grid(n_dim=2, length=5)


@pytest.fixture
def grid10x10x10():
    return Grid(n_dim=3, length=10)


@pytest.mark.parametrize(
    "state_x, state_y, is_equal",
    [
        ### Integers
        (
            0,
            0,
            True,
        ),
        (
            17,
            17,
            True,
        ),
        (
            17,
            18,
            False,
        ),
        ### Floats
        (
            0.0,
            0.0,
            True,
        ),
        (
            17.8,
            17.8,
            True,
        ),
        (
            17.0,
            18.0,
            False,
        ),
        (
            17.0,
            18.0,
            False,
        ),
        ### Lists
        (
            [],
            [],
            True,
        ),
        (
            [],
            [0],
            False,
        ),
        (
            [0],
            [0],
            True,
        ),
        (
            [0],
            [1],
            False,
        ),
        (
            [0, 1, -1],
            [0, 1, -1],
            True,
        ),
        (
            [0, 1, 1],
            [0, 1, -1],
            False,
        ),
        (
            [0, 1],
            [0, 1, -1],
            False,
        ),
        (
            [0.0, 1.0, -1.0],
            [0.0, 1.0, -1.0],
            True,
        ),
        (
            [0.0, 1.0, -1.0],
            [0.0, 1.0, 1.0],
            False,
        ),
        (
            ["a", "b", -1, 1],
            ["a", "b", -1, 1],
            True,
        ),
        (
            ["a", "b", -1, 0],
            ["a", "b", -1, 1],
            False,
        ),
        ### Lists of lists
        (
            [[0, 1], ["a", "b", -1, 0]],
            [[0, 1], ["a", "b", -1, 0]],
            True,
        ),
        (
            [[0, 1], ["a", "b", -1, 1]],
            [[0, 1], ["a", "b", -1, 0]],
            False,
        ),
        (
            [[0, 1], ["a", "b", -1, 0], 0.5],
            [[0, 1], ["a", "b", -1, 0], 0.5],
            True,
        ),
        (
            [[0, 1], ["a", "b", -1, 0], 0.5],
            [[0, 1], ["a", "b", -1, 0], 1.5],
            False,
        ),
        ### Tensors
        (
            torch.tensor([0.0, 1.0, -1.0]),
            torch.tensor([0.0, 1.0, -1.0]),
            True,
        ),
        (
            torch.tensor([0.0, 1.0, -1.0]),
            torch.tensor([0.0, 1.0, 1.0]),
            False,
        ),
        (
            torch.tensor([0.0, 1.0, -1.0, torch.nan]),
            torch.tensor([0.0, 1.0, -1.0, torch.nan]),
            True,
        ),
        (
            torch.tensor([0.0, 1.0, -1.0, torch.nan]),
            torch.tensor([0.0, 1.0, 1.0, torch.nan]),
            False,
        ),
        (
            torch.tensor([[0.0, 1.0, -1.0], [1.0, 2.0, -1.0]]),
            torch.tensor([[0.0, 1.0, -1.0], [1.0, 2.0, -1.0]]),
            True,
        ),
        (
            torch.tensor([[0.0, 1.0, -1.0], [1.0, 2.0, -1.0]]),
            torch.tensor([[0.0, 1.0, -1.0], [1.0, 2.0, -1.00001]]),
            False,
        ),
        ### Numpy
        (
            np.array([0.0, 1.0, -1.0]),
            np.array([0.0, 1.0, -1.0]),
            True,
        ),
        (
            np.array([0.0, 1.0, -1.0]),
            np.array([0.0, 1.0, 1.0]),
            False,
        ),
        (
            np.array([0.0, 1.0, -1.0, np.nan]),
            np.array([0.0, 1.0, -1.0, np.nan]),
            True,
        ),
        (
            np.array([0.0, 1.0, -1.0, np.nan]),
            np.array([0.0, 1.0, 1.0, np.nan]),
            False,
        ),
        (
            np.array([[0.0, 1.0, -1.0], [1.0, 2.0, -1.0]]),
            np.array([[0.0, 1.0, -1.0], [1.0, 2.0, -1.0]]),
            True,
        ),
        (
            np.array([[0.0, 1.0, -1.0], [1.0, 2.0, -1.0]]),
            np.array([[0.0, 1.0, -1.0], [1.0, 2.0, -1.00001]]),
            False,
        ),
        ### Dictionaries
        (
            {0: [1, 2, 3], 1: ["a", "b"]},
            {0: [1, 2, 3], 1: ["a", "b"]},
            True,
        ),
        # Key is different
        (
            {0: [1, 2, 3], 1: ["a", "b"]},
            {0: [1, 2, 3], 2: ["a", "b"]},
            False,
        ),
        # Value is different
        (
            {0: [1, 2, 3], 1: ["a", "b"]},
            {0: [1, 2, 3], 1: ["a", "c"]},
            False,
        ),
        # Order of keys are different
        (
            {0: [1, 2, 3], 1: ["a", "b"]},
            {1: ["a", "b"], 0: [1, 2, 3]},
            True,
        ),
        # Different lengths
        (
            {0: [1, 2, 3], 1: ["a", "b"]},
            {0: [1, 2, 3], 1: ["a", "b"], 2: 0},
            False,
        ),
        (
            {0: [1, 2, 3], 1: ["a", "b"], 2: 0},
            {0: [1, 2, 3], 1: ["a", "b"]},
            False,
        ),
        ### Counters
        (
            Counter(),
            Counter(),
            True,
        ),
        (
            Counter({1: 1}),
            Counter({1: 1}),
            True,
        ),
        (
            Counter({1: 1}),
            Counter({1: 2}),
            False,
        ),
        (
            Counter({1: 1}),
            Counter({2: 1}),
            False,
        ),
        (
            Counter({1: 1}),
            Counter(),
            False,
        ),
        ### Tuples
        (
            (),
            (),
            True,
        ),
        (
            (1,),
            (1,),
            True,
        ),
        (
            (1,),
            (2,),
            False,
        ),
        (
            (1, 2),
            (1, 2, 3),
            False,
        ),
        (
            (1, [0, 1], "a", (2, 3)),
            (1, [0, 1], "a", (2, 3)),
            True,
        ),
        (
            (1, [0, 1], "a", (2, 3)),
            (1, [0, 1], "b", (2, 3)),
            False,
        ),
        (
            (1, [0, 1], "a", (2, 3)),
            (1, [0, 1], "a", (2, 3, 1)),
            False,
        ),
        ### List of Counter and tuple
        (
            [Counter(), ()],
            [Counter(), ()],
            True,
        ),
        (
            [Counter({(1, 1): 2}), ()],
            [Counter({(1, 1): 2}), ()],
            True,
        ),
        (
            [Counter({(1, 1): 2}), ()],
            [Counter({(1, 1): 3}), ()],
            False,
        ),
        (
            [Counter({(1, 1): 2}), ()],
            [Counter({(1, 1): 2}), (1, 2)],
            False,
        ),
        (
            [Counter({(1, 1): 2}), ()],
            [Counter(), ()],
            False,
        ),
    ],
)
def test__equal__behaves_as_expected(grid, state_x, state_y, is_equal):
    # The grid is use as a generic environment. Note that the values compared are not
    # grid states, but it does not matter for the purposes of this test.
    env = grid
    assert env.equal(state_x, state_y) == is_equal


@pytest.mark.parametrize(
    "env, n_states, max_attempts",
    [
        ("grid5x5", 10, 5),
        ("grid10x10x10", 20, 19),
    ],
)
def test__get_random_states__raises_value_error_if_n_states_larger_than_max_attempts(
    env, n_states, max_attempts, request
):
    env = request.getfixturevalue(env)
    with pytest.raises(ValueError):
        states = env.get_random_states(
            n_states, unique=True, exclude_source=False, max_attempts=max_attempts
        )
        assert True


@pytest.mark.parametrize(
    "env, n_states",
    [
        ("grid", 3),
        ("grid5x5", 10),
        ("grid10x10x10", 20),
    ],
)
def test__get_random_states__returns_unique_states_if_unique_is_true(
    env, n_states, request
):
    env = request.getfixturevalue(env)
    states = env.get_random_states(
        n_states, unique=True, exclude_source=False, max_attempts=10000
    )
    # Check that the number of states is the requested one
    assert len(states) == n_states
    # Check that all states are different
    states_unique = []
    for state in states:
        if not any([env.equal(state, s) for s in states_unique]):
            states_unique.append(state)
    assert len(states_unique) == len(states)


@pytest.mark.parametrize(
    "env, n_states",
    [
        ("grid", 3),
        ("grid5x5", 10),
        ("grid5x5", 20),
        ("grid10x10x10", 20),
    ],
)
def test__get_random_states__does_not_contain_the_source_if_exclude_source_is_true(
    env, n_states, request
):
    env = request.getfixturevalue(env)
    states = env.get_random_states(
        n_states, unique=False, exclude_source=True, max_attempts=10000
    )
    # Check that the number of states is the requested one
    assert len(states) == n_states
    # Check that the source state is not included
    assert all([not env.is_source(state) for state in states])


@pytest.mark.parametrize(
    "env",
    [
        ("grid"),
        ("grid5x5"),
        ("grid10x10x10"),
    ],
)
def test__eq__returns_true_for_copied_envs(env, request):
    env = request.getfixturevalue(env)
    env_copy = env.copy()
    assert env == env_copy
    # Changing id does not make environments not equal
    env.set_id("banana")
    assert env == env_copy


@pytest.mark.parametrize(
    "env",
    [
        ("grid"),
        ("grid5x5"),
        ("grid10x10x10"),
    ],
)
@pytest.mark.parametrize(
    "key, value",
    [
        ("continuous", True),
        ("n_dim", 1),
        ("state", [1]),
        ("eos", (1, 2)),
        ("source", []),
        ("source", {"a": 1, "b": 2}),
        ("state", torch.tensor([0.0, 1.0, -1.0])),
        ("state", np.array([0.0, 1.0, -1.0])),
    ],
)
def test__eq__returns_false_if_attribute_is_changed(env, key, value, request):
    env = request.getfixturevalue(env)
    env_copy = env.copy()
    env.__dict__[key] = value
    assert env != env_copy


@pytest.mark.parametrize(
    "env, subenv",
    [
        ("grid", "grid5x5"),
        ("grid5x5", "grid"),
        ("grid10x10x10", "grid5x5"),
    ],
)
def test__eq__returns_true_for_copied_envs_with_nested_subenv(env, subenv, request):
    env = request.getfixturevalue(env)
    subenv = request.getfixturevalue(subenv)
    env.__dict__["subenv_nested"] = subenv.copy().set_id(0)
    env.__dict__["subenv_nested_list"] = [
        subenv.copy().set_id(1),
        subenv.copy().set_id(2),
    ]
    env.__dict__["subenv_nested_dict"] = {
        11: subenv.copy().set_id(11),
        22: subenv.copy().set_id(22),
    }
    env_copy = env.copy()
    assert env == env_copy


@pytest.mark.parametrize(
    "env, subenv",
    [
        ("grid", "grid5x5"),
        ("grid5x5", "grid"),
        ("grid10x10x10", "grid5x5"),
    ],
)
@pytest.mark.parametrize(
    "key, value",
    [
        ("continuous", True),
        ("n_dim", 1),
        ("state", [1]),
        ("eos", (1, 2)),
        ("source", []),
        ("source", {"a": 1, "b": 2}),
        ("state", torch.tensor([0.0, 1.0, -1.0])),
        ("state", np.array([0.0, 1.0, -1.0])),
    ],
)
def test__eq__returns_false_if_attribute_is_changed_in_nested_subenv(
    env, subenv, key, value, request
):
    env = request.getfixturevalue(env)
    subenv = request.getfixturevalue(subenv)
    env.__dict__["subenv_nested"] = subenv
    env_copy = env.copy()
    env.subenv_nested.__dict__[key] = value
    assert env != env_copy


@pytest.mark.parametrize(
    "env, subenv",
    [
        ("grid", "grid5x5"),
        ("grid5x5", "grid"),
        ("grid10x10x10", "grid5x5"),
    ],
)
@pytest.mark.parametrize(
    "key, value",
    [
        ("continuous", True),
        ("n_dim", 1),
        ("state", [1]),
        ("eos", (1, 2)),
        ("source", []),
        ("source", {"a": 1, "b": 2}),
        ("state", torch.tensor([0.0, 1.0, -1.0])),
        ("state", np.array([0.0, 1.0, -1.0])),
    ],
)
def test__eq__returns_false_if_attribute_is_changed_in_nested_subenv_as_list(
    env, subenv, key, value, request
):
    env = request.getfixturevalue(env)
    subenv = request.getfixturevalue(subenv)
    env.__dict__["subenv_nested"] = [subenv.copy().set_id(1), subenv.copy().set_id(2)]
    env_copy = env.copy()
    env.subenv_nested[0].__dict__[key] = value
    assert env != env_copy


@pytest.mark.parametrize(
    "env, subenv",
    [
        ("grid", "grid5x5"),
        ("grid5x5", "grid"),
        ("grid10x10x10", "grid5x5"),
    ],
)
@pytest.mark.parametrize(
    "key, value",
    [
        ("continuous", True),
        ("n_dim", 1),
        ("state", [1]),
        ("eos", (1, 2)),
        ("source", []),
        ("source", {"a": 1, "b": 2}),
        ("state", torch.tensor([0.0, 1.0, -1.0])),
        ("state", np.array([0.0, 1.0, -1.0])),
    ],
)
def test__eq__returns_false_if_attribute_is_changed_in_nested_subenv_as_dict(
    env, subenv, key, value, request
):
    env = request.getfixturevalue(env)
    subenv = request.getfixturevalue(subenv)
    env.__dict__["subenv_nested"] = {
        1: subenv.copy().set_id(1),
        2: subenv.copy().set_id(2),
    }
    env_copy = env.copy()
    env.subenv_nested[1].__dict__[key] = value
    assert env != env_copy
