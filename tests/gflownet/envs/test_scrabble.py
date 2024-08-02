import common
import pytest

from gflownet.envs.scrabble import Scrabble


@pytest.fixture
def env():
    return Scrabble(max_length=7, device="cpu")


def test__environment_initializes_properly():
    env = Scrabble(max_length=7, device="device")
    assert True


@pytest.mark.parametrize(
    "action_space",
    [
        [
            # fmt: off
            (1,), (2,), (3,), (4,), (5,),
            (6,), (7,), (8,), (9,), (10,),
            (11,), (12,), (13,), (14,), (15,),
            (16,), (17,), (18,), (19,), (20,),
            (21,), (22,), (23,), (24,), (25,),
            (26,), (-1,)
            # fmt: on
        ],
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "state, parents_expected, parents_a_expected",
    [
        (
            [0, 0, 0, 0, 0, 0, 0],
            [],
            [],
        ),
        (
            [1, 0, 0, 0, 0, 0, 0],
            [[0, 0, 0, 0, 0, 0, 0]],
            [(1,)],
        ),
        (
            [1, 3, 2, 4, 0, 0, 0],
            [[1, 3, 2, 0, 0, 0, 0]],
            [(4,)],
        ),
    ],
)
def test__get_parents__returns_expected(
    env, state, parents_expected, parents_a_expected
):
    parents, parents_a = env.get_parents(state)
    for p, p_e in zip(parents, parents_expected):
        assert p == p_e
    for p_a, p_a_e in zip(parents_a, parents_a_expected):
        assert p_a == p_a_e


@pytest.mark.parametrize(
    "state, action, next_state",
    [
        (
            [0, 0, 0, 0, 0, 0, 0],
            (1,),
            [1, 0, 0, 0, 0, 0, 0],
        ),
        (
            [1, 0, 0, 0, 0, 0, 0],
            (3,),
            [1, 3, 0, 0, 0, 0, 0],
        ),
        (
            [1, 3, 0, 0, 0, 0, 0],
            (2,),
            [1, 3, 2, 0, 0, 0, 0],
        ),
        (
            [1, 3, 2, 0, 0, 0, 0],
            (1,),
            [1, 3, 2, 1, 0, 0, 0],
        ),
        (
            [1, 3, 2, 1, 0, 0, 0],
            (4,),
            [1, 3, 2, 1, 4, 0, 0],
        ),
        (
            [1, 3, 2, 1, 4, 0, 0],
            (-1,),
            [1, 3, 2, 1, 4, 0, 0],
        ),
        (
            [1, 3, 2, 0, 0, 0, 0],
            (-1,),
            [1, 3, 2, 0, 0, 0, 0],
        ),
    ],
)
def test__step__returns_expected(env, state, action, next_state):
    env.set_state(state)
    env.step(action)
    assert env.state == next_state


@pytest.mark.parametrize(
    "state, readable",
    [
        ([0, 0, 0, 0, 0, 0, 0], ""),
        ([3, 1, 20, 0, 0, 0, 0], "C A T"),
        ([2, 9, 18, 4, 19, 0, 0], "B I R D S"),
        ([13, 1, 3, 8, 9, 14, 5], "M A C H I N E"),
    ],
)
def test__state2readable__returns_expected(env, state, readable):
    assert env.state2readable(state) == readable


@pytest.mark.parametrize(
    "state, readable",
    [
        ([0, 0, 0, 0, 0, 0, 0], ""),
        ([3, 1, 20, 0, 0, 0, 0], "C A T"),
        ([2, 9, 18, 4, 19, 0, 0], "B I R D S"),
        ([13, 1, 3, 8, 9, 14, 5], "M A C H I N E"),
    ],
)
def test__readable2state__returns_expected(env, state, readable):
    assert env.readable2state(readable) == state


class TestScrabbleCommon(common.BaseTestsDiscrete):
    """Common tests for Scrabble."""

    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__get_parents__all_parents_are_reached_with_different_actions": 10,
        }
        self.n_states = {}  # TODO: Populate.
