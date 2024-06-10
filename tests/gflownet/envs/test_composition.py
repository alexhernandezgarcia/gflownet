import common
import numpy as np
import pytest
import torch

from gflownet.envs.crystals.composition import Composition
from gflownet.utils.common import tfloat, tlong


@pytest.fixture
def env():
    return Composition(
        elements=4,
        alphabet={1: "H", 2: "He", 3: "Li", 4: "Be"},
        oxidation_states={1: [-1, 0, 1], 2: [0], 3: [0, 1], 4: [0, 1, 2]},
    )


@pytest.fixture
def env_restricted_elements():
    return Composition(
        elements=[1, 3],
        alphabet={1: "H", 2: "He", 3: "Li", 4: "Be"},
        oxidation_states={1: [-1, 0, 1], 2: [0], 3: [0, 1], 4: [0, 1, 2]},
    )


@pytest.fixture
def env_with_spacegroup():
    return Composition(
        elements=4,
        alphabet={1: "H", 2: "He", 3: "Li", 4: "Be"},
        oxidation_states={1: [-1, 0, 1], 2: [0], 3: [0, 1], 4: [0, 1, 2]},
        space_group=162,
        do_spacegroup_check=True,
    )


@pytest.mark.parametrize("elements", [2, 5, 10, 84])
def test__environment__initializes_properly(elements):
    env = Composition(elements=elements)

    assert env.state == {}


@pytest.mark.parametrize(
    "state, exp_tensor",
    [
        (
            {3: 2},
            [
                # fmt: off
                0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                # fmt: on
            ],
        ),
        (
            {1: 3},
            [
                # fmt: off
                0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                # fmt: on
            ],
        ),
        (
            {2: 1, 4: 1},
            [
                # fmt: off
                0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                # fmt: on
            ],
        ),
    ],
)
def test__state2proxy__returns_expected_tensor(env, state, exp_tensor):
    assert torch.equal(env.state2proxy(state)[0], tlong(exp_tensor, device=env.device))


@pytest.mark.parametrize(
    "batch, exp_tensor",
    [
        (
            [{3: 2}, {1: 3}, {2: 1, 4: 1}, {}],
            [
                # fmt: off
                [
                    0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                [
                    0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                [
                    0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ],
                # fmt: on
            ],
        ),
    ],
)
def test__states2proxy__returns_expected_tensor(env, batch, exp_tensor):
    assert torch.equal(
        env.states2proxy(batch),
        tfloat(exp_tensor, device=env.device, float_type=env.float),
    )


@pytest.mark.parametrize(
    "state, exp_tensor",
    [
        (
            {3: 2},
            [0, 0, 2, 0],
        ),
        (
            {1: 3},
            [3, 0, 0, 0],
        ),
        (
            {2: 1, 4: 1},
            [0, 1, 0, 1],
        ),
    ],
)
def test__state2policy__returns_expected_tensor(env, state, exp_tensor):
    assert torch.equal(
        env.state2policy(state),
        tfloat(exp_tensor, device=env.device, float_type=env.float),
    )


@pytest.mark.parametrize(
    "batch, exp_tensor",
    [
        (
            [{3: 2}, {1: 3}, {2: 1, 4: 1}, {}],
            [[0, 0, 2, 0], [3, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0]],
        ),
    ],
)
def test__states2policy__returns_expected_tensor(env, batch, exp_tensor):
    assert torch.equal(
        env.states2policy(batch),
        tfloat(exp_tensor, device=env.device, float_type=env.float),
    )


@pytest.mark.parametrize(
    "batch, exp_tensor",
    [
        (
            [{3: 2}, {1: 3}, {1: 3, 3: 2}, {}],
            [[0, 2], [3, 0], [3, 2], [0, 0]],
        ),
    ],
)
def test__states2policy__restricted_elements__returns_expected_tensor(
    env_restricted_elements, batch, exp_tensor
):
    env = env_restricted_elements
    assert torch.equal(
        env.states2policy(batch),
        tfloat(exp_tensor, device=env.device, float_type=env.float),
    )


@pytest.mark.parametrize(
    "state, readable",
    [
        ({1: 2, 3: 1}, "H2Li1"),
        ({1: 2}, "H2"),
        ({3: 2}, "Li2"),
        ({1: 2, 2: 5, 3: 1, 4: 16}, "H2Be16He5Li1"),
        ({2: 15, 3: 7, 4: 16}, "Be16He15Li7"),
    ],
)
def test__state2readable(env, state, readable):
    assert env.state2readable(state) == readable
    env.set_state(state)
    assert env.state2readable() == readable


@pytest.mark.parametrize(
    "state, readable",
    [
        ({1: 2, 3: 1}, "H2Li1"),
        ({1: 2}, "H2"),
        ({3: 2}, "Li2"),
        ({1: 2, 2: 5, 3: 1, 4: 16}, "H2Be16He5Li1"),
        ({2: 15, 3: 7, 4: 16}, "Be16He15Li7"),
    ],
)
def test__readable2state(env, state, readable):
    assert env.readable2state(readable) == state


def test__reset(env):
    env.step((1, 1))
    env.step((2, 1))

    assert env.state != {}

    env.reset()

    assert env.state == {}


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
    environment = Composition(
        elements=elements, min_atom_i=min_atom_i, max_atom_i=max_atom_i
    )
    exp_n_actions = elements * (max_atom_i - min_atom_i + 1) + 1

    assert len(environment.get_action_space()) == exp_n_actions


@pytest.mark.parametrize(
    "elements",
    [[1, 2, 3, 4], [1, 12, 84], [42]],
)
def test__get_action_space__returns_actions_for_each_element(elements):
    environment = Composition(elements=elements)

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
    environment = Composition(
        elements=elements, min_atom_i=min_atom_i, max_atom_i=max_atom_i
    )

    step_sizes_in_action_space = set(
        n for e, n in environment.get_action_space()[:-1]
    )  # skip eos
    exp_step_sizes = set(range(min_atom_i, max_atom_i + 1))

    assert step_sizes_in_action_space == exp_step_sizes


def test__get_mask_invalid_actions_forward__all_false_but_eos_for_empty_state(env):
    assert not any(env.get_mask_invalid_actions_forward()[:-1])
    assert env.get_mask_invalid_actions_forward()[:-1]


@pytest.mark.parametrize(
    "state",
    [{}, {2: 2}, {4: 1}, {1: 1, 3: 1}],
)
def test__get_mask_invalid_actions_forward__already_set_elements_are_masked(env, state):
    mask = env.get_mask_invalid_actions_forward(state)[:-1]
    action_space = env.action_space[:-1]

    for el in state:
        for action, is_invalid in zip(action_space, mask):
            if action[0] == el:
                assert is_invalid


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
            {3: 2},
            [{}],
            [(3, 2)],
        ),
        (
            {1: 3},
            [{}],
            [(1, 3)],
        ),
        (
            {2: 1, 4: 1},
            [{2: 1}, {4: 1}],
            [(2, 1), (4, 1)],
        ),
        (
            {1: 1, 2: 2, 3: 3, 4: 4},
            [
                {2: 2, 3: 3, 4: 4},
                {1: 1, 3: 3, 4: 4},
                {1: 1, 2: 2, 4: 4},
                {1: 1, 2: 2, 3: 3},
            ],
            [(1, 1), (2, 2), (3, 3), (4, 4)],
        ),
    ],
)
def test__get_parents__returns_expected_parents_and_actions(
    env, state, exp_parents, exp_actions
):
    env.state = state

    parents, actions = env.get_parents()

    assert set(tuple(sorted(p.items())) for p in parents) == set(
        tuple(sorted(p.items())) for p in exp_parents
    )
    assert set(actions) == set(exp_actions)


@pytest.mark.parametrize(
    "actions, exp_state",
    [
        ([(1, 2), (2, 3), (3, 4)], {1: 2, 2: 3, 3: 4}),
        ([(4, 2)], {4: 2}),
        ([(1, 3), (4, 2), (2, 3), (3, 2)], {1: 3, 2: 3, 3: 2, 4: 2}),
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
    # Copy because step will update the same dictionary
    initial_state = env.state.copy()

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
            {},
            True,
        ),
        (
            {1: 3},
            True,
        ),
        (
            {2: 1, 4: 1},
            False,
        ),
        (
            {1: 1, 2: 2, 3: 3, 4: 4},
            False,
        ),
        (
            {1: 5, 4: 2},
            True,
        ),
    ],
)
def test__can_produce_neutral_charge__returns_expected_result(state, exp_result):
    environment = Composition(
        elements=4,
        oxidation_states={1: [-1, 0, 1], 2: [0], 3: [1], 4: [2, 3]},
    )

    assert environment._can_produce_neutral_charge(state) == exp_result


def test__get_mask_invalid_actions_forward__accounts_for_required_elements():
    required_elements = [1, 2, 4, 5]
    env = Composition(
        elements=6, required_elements=required_elements, max_atom_i=1, max_diff_elem=4
    )
    mask = env.get_mask_invalid_actions_forward()

    for (element, n), masked in zip(env.action_space, mask):
        if element in required_elements:
            assert not masked
        else:
            assert masked


@pytest.mark.repeat(25)
def test__required_elements_does_not_cause_environment_to_get_stuck():
    required_elements = [1, 2, 3, 4, 5]
    env = Composition(
        elements=89, max_diff_elem=10, required_elements=required_elements
    )

    while not env.done:
        mask = env.get_mask_invalid_actions_forward()
        actions = [action for action, m in zip(env.action_space, mask) if not m]
        assert len(actions) > 0
        action = actions[np.random.choice(len(actions))]
        env.step(action)


@pytest.mark.repeat(25)
def test__required_atoms_does_not_cause_environment_to_get_stuck():
    required_elements = []
    env = Composition(
        elements=10,
        min_diff_elem=2,
        max_diff_elem=2,
        min_atoms=20,
        max_atoms=20,
    )

    while not env.done:
        mask = env.get_mask_invalid_actions_forward()
        actions = [action for action, m in zip(env.action_space, mask) if not m]
        assert len(actions) > 0
        action = actions[np.random.choice(len(actions))]
        env.step(action)


@pytest.mark.repeat(25)
def test__insufficient_elements_left_does_not_cause_environment_to_get_stuck():
    env = Composition(
        elements=10,
        min_diff_elem=5,
        max_diff_elem=5,
        max_atoms=25,
        min_atom_i=4,
        max_atom_i=10,
    )

    while not env.done:
        mask = env.get_mask_invalid_actions_forward()
        actions = [action for action, m in zip(env.action_space, mask) if not m]
        assert len(actions) > 0
        action = actions[np.random.choice(len(actions))]
        env.step(action)


class TestCompositionBasic(common.BaseTestsDiscrete):
    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
        self.repeats = {
            "test__get_logprobs__backward__returns_zero_if_done": 100,  # Overrides no repeat.
            "test__reset__state_is_source": 10,
        }
        self.n_states = {}  # TODO: Populate.


class TestCompositionWithSpaceGroup(common.BaseTestsDiscrete):
    @pytest.fixture(autouse=True)
    def setup(self, env_with_spacegroup):
        self.env = env_with_spacegroup
        self.repeats = {
            "test__get_logprobs__backward__returns_zero_if_done": 100,  # Overrides no repeat.
            "test__reset__state_is_source": 10,
        }
        self.n_states = {}  # TODO: Populate.
