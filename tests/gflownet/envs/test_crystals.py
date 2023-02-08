import pytest
import torch

from gflownet.envs.crystals import Crystal


@pytest.fixture
def env():
    return Crystal(periodic_table=4, alphabet={0: "H", 1: "He", 2: "Li", 3: "Be"})


@pytest.mark.parametrize("periodic_table", [2, 5, 10, 84])
def test__environment__initializes_properly(periodic_table):
    env = Crystal(periodic_table=periodic_table)

    assert env.state == [0] * periodic_table


@pytest.mark.parametrize(
    "state, exp_tensor",
    [
        (
            [0, 0, 2, 0],
            [2, 2, 0, 0, 1, 0],
        ),
        (
            [3, 0, 0, 0],
            [0, 3, 1, 0, 0, 0],
        ),
        (
            [0, 1, 0, 1],
            [0, 2, 0, 0.5, 0, 0.5],
        ),
    ],
)
def test__state2oracle__returns_expected_tensor(env, state, exp_tensor):
    assert torch.equal(env.state2oracle(state), torch.Tensor(exp_tensor))


def test__state2readable(env):
    state = [2, 0, 1, 0]
    readable = {"H": 2, "Li": 1}

    env.state = state

    assert env.state2readable(state=state) == readable
    assert env.state2readable() == readable


def test__readable2state(env):
    state = [2, 0, 1, 0]
    short_readable = {"H": 2, "Li": 1}
    long_readable = {"H": 2, "He": 0, "Li": 1, "Be": 0}

    assert env.readable2state(readable=short_readable) == state
    assert env.readable2state(readable=long_readable) == state


def test__reset(env):
    env.step(0)
    env.step(1)
    env.step(0)

    assert env.state != [0] * env.periodic_table

    env.reset()

    assert env.state == [0] * env.periodic_table


@pytest.mark.parametrize(
    "periodic_table, min_atom_i, max_atom_i",
    [
        (4, 1, 5),
        (10, 1, 20),
        (84, 1, 8),
        (4, 3, 5),
        (10, 3, 20),
        (84, 3, 8),
    ],
)
def test__get_actions_space__returns_correct_number_of_actions(
    periodic_table, min_atom_i, max_atom_i
):
    environment = Crystal(
        periodic_table=periodic_table, min_atom_i=min_atom_i, max_atom_i=max_atom_i
    )

    assert len(environment.get_actions_space()) == periodic_table * (
        max_atom_i - min_atom_i + 1
    )


def test__get_parents__returns_no_parents_in_initial_state(env):
    parents, actions = env.get_parents()

    assert len(parents) == 0
    assert len(actions) == 0


def test__get_parents__returns_parents_after_step(env):
    env.step(0)

    parents, actions = env.get_parents()

    assert len(parents) != 0
    assert len(actions) != 0


@pytest.mark.parametrize("action_indices", [[], [0, 1, 0], [0], [6, 4, 2, 0]])
def test__get_parents__returns_same_number_of_parents_and_actions(env, action_indices):
    for action_idx in action_indices:
        env.step(action_idx=action_idx)

    parents, actions = env.get_parents()

    assert len(parents) == len(actions)


@pytest.mark.parametrize(
    "state, exp_parents, exp_actions",
    [
        (
            [0, 0, 2, 0],
            [[0, 0, 0, 0]],
            [(2, 2)],
        ),
        (
            [3, 0, 0, 0],
            [[0, 0, 0, 0]],
            [(0, 3)],
        ),
        (
            [0, 1, 0, 1],
            [[0, 1, 0, 0], [0, 0, 0, 1]],
            [(1, 1), (3, 1)],
        ),
        (
            [1, 2, 3, 4],
            [[0, 2, 3, 4], [1, 0, 3, 4], [1, 2, 0, 4], [1, 2, 3, 0]],
            [(0, 1), (1, 2), (2, 3), (3, 4)],
        ),
    ],
)
def test__get_parents__returns_expected_parents_and_actions(
    env, state, exp_parents, exp_actions
):
    env.state = state

    parents, actions = env.get_parents()

    assert set(tuple(x) for x in parents) == set(tuple(x) for x in exp_parents)
    assert set(env.action_space[x] for x in actions) == set(exp_actions)
