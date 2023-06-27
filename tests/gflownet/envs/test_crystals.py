import common
import pytest
import torch

from gflownet.envs.crystals import Crystal


@pytest.fixture
def env():
    return Crystal(
        elements=4,
        alphabet={1: "H", 2: "He", 3: "Li", 4: "Be"},
        oxidation_states={1: [-1, 0, 1], 2: [0], 3: [0, 1], 4: [0, 1, 2]},
    )


@pytest.mark.parametrize("elements", [2, 5, 10, 84])
def test__environment__initializes_properly(elements):
    env = Crystal(elements=elements)

    assert env.state == [0] * elements


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
    env.step((1, 1))
    env.step((2, 1))

    assert env.state != [0] * len(env.elements)

    env.reset()

    assert env.state == [0] * len(env.elements)


@pytest.mark.parametrize(
    "elements, min_atom_i, max_atom_i",
    [
        (4, 1, 5),
        (10, 1, 20),
        (84, 1, 8),
        (4, 3, 5),
        (10, 3, 20),
        (84, 3, 8),
    ],
)
def test__get_action_space__returns_correct_number_of_actions(
    elements, min_atom_i, max_atom_i
):
    environment = Crystal(
        elements=elements, min_atom_i=min_atom_i, max_atom_i=max_atom_i
    )
    exp_n_actions = elements * (max_atom_i - min_atom_i + 1) + 1

    assert len(environment.get_action_space()) == exp_n_actions


@pytest.mark.parametrize(
    "elements",
    [[1, 2, 3, 4], [1, 12, 84], [42]],
)
def test__get_action_space__returns_actions_for_each_element(elements):
    environment = Crystal(elements=elements)

    elements_in_action_space = set(e for e, n in environment.get_action_space())
    exp_elements_with_eos = set(elements + [-1])

    assert elements_in_action_space == exp_elements_with_eos


@pytest.mark.parametrize(
    "elements, min_atom_i, max_atom_i",
    [
        (4, 1, 5),
        (10, 1, 20),
        (84, 1, 8),
        (4, 3, 5),
        (10, 3, 20),
        (84, 3, 8),
    ],
)
def test__get_action_space__returns_actions_for_each_step_size(
    elements, min_atom_i, max_atom_i
):
    environment = Crystal(
        elements=elements, min_atom_i=min_atom_i, max_atom_i=max_atom_i
    )

    step_sizes_in_action_space = set(
        n for e, n in environment.get_action_space()[:-1]
    )  # skip eos
    exp_step_sizes = set(range(min_atom_i, max_atom_i + 1))

    assert step_sizes_in_action_space == exp_step_sizes


def test__get_mask_invalid_actions__all_false_but_eos_for_empty_state(env):
    assert not any(env.get_mask_invalid_actions()[:-1])
    assert env.get_mask_invalid_actions()[:-1]


@pytest.mark.parametrize(
    "state",
    [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]],
)
def test__get_mask_invalid_actions__already_set_elements_are_masked(env, state):
    mask = env.get_mask_invalid_actions(state)[:-1]
    action_space = env.action_space[:-1]

    nonzero_indices = [i for i, s_i in enumerate(state) if s_i > 0]

    for i in nonzero_indices:
        for a_j, m_j in zip(action_space, mask):
            if env.elem2idx[a_j[0]] == i:
                assert m_j


def test__get_parents__returns_no_parents_in_initial_state(env):
    return common.test__get_parents__returns_no_parents_in_initial_state(env)


def test__get_parents__returns_parents_after_step(env):
    env.step((1, 4))

    parents, actions = env.get_parents()

    assert len(parents) != 0
    assert len(actions) != 0


@pytest.mark.parametrize(
    "actions",
    [[], [(1, 2), (2, 3), (3, 4)], [(4, 2)], [(1, 3), (4, 2), (2, 3), (3, 2)]],
)
def test__get_parents__returns_same_number_of_parents_and_actions(env, actions):
    for action in actions:
        env.step(action=action)

    parents, actions = env.get_parents()

    assert len(parents) == len(actions)


@pytest.mark.parametrize(
    "state, exp_parents, exp_actions",
    [
        (
            [0, 0, 2, 0],
            [[0, 0, 0, 0]],
            [(3, 2)],
        ),
        (
            [3, 0, 0, 0],
            [[0, 0, 0, 0]],
            [(1, 3)],
        ),
        (
            [0, 1, 0, 1],
            [[0, 1, 0, 0], [0, 0, 0, 1]],
            [(2, 1), (4, 1)],
        ),
        (
            [1, 2, 3, 4],
            [[0, 2, 3, 4], [1, 0, 3, 4], [1, 2, 0, 4], [1, 2, 3, 0]],
            [(1, 1), (2, 2), (3, 3), (4, 4)],
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


@pytest.mark.parametrize(
    "actions, exp_state",
    [
        ([(1, 2), (2, 3), (3, 4)], [2, 3, 4, 0]),
        ([(4, 2)], [0, 0, 0, 2]),
        ([(1, 3), (4, 2), (2, 3), (3, 2)], [3, 3, 2, 2]),
    ],
)
def test__step__changes_state_as_expected(env, actions, exp_state):
    for action in actions:
        env.step(action=action)

    assert env.state == exp_state
    assert env.n_actions == len(actions)


@pytest.mark.parametrize(
    "valid_action, invalid_action",
    [
        ((1, 2), (1, 4)),
        ((3, 1), (3, 2)),
        ((4, 4), (4, 1)),
    ],
)
def test__step__does_not_change_state_if_element_already_set(
    env, valid_action, invalid_action
):
    initial_state = env.state

    state_after_valid, action, valid = env.step(valid_action)

    assert action == valid_action
    assert valid
    assert initial_state != state_after_valid


# TODO: uncomment when step can handle invalid actions
#     state_after_invalid, action, valid = env.step(invalid_action)

#     assert action == invalid_action
#     assert not valid
#     assert state_after_valid == state_after_invalid


@pytest.mark.parametrize(
    "state, exp_result",
    [
        (
            [0, 0, 0, 0],
            True,
        ),
        (
            [3, 0, 0, 0],
            True,
        ),
        (
            [0, 1, 0, 1],
            False,
        ),
        (
            [1, 2, 3, 4],
            False,
        ),
        (
            [5, 0, 0, 2],
            True,
        ),
    ],
)
def test__can_produce_neutral_charge__returns_expected_result(state, exp_result):
    environment = Crystal(
        elements=4,
        oxidation_states={1: [-1, 0, 1], 2: [0], 3: [1], 4: [2, 3]},
    )

    assert environment._can_produce_neutral_charge(state) == exp_result
