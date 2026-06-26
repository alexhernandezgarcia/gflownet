"""
Tests for the Sequence composite environment.

The common test battery (``common.BaseTestsDiscrete`` / ``common.BaseTestsContinuous``)
is wired in at the bottom of the file, mirroring ``test_setflex.py`` and
``test_stack.py``. ``test__gflownet_minimal_runs`` is disabled (set to 0) because, as
with the SetFlex, running it would require a proxy tailored to the composite states.
"""

import common
import numpy as np
import pytest
import torch

from gflownet.envs.composite.sequence import Sequence
from gflownet.envs.cube import ContinuousCube
from gflownet.envs.grid import Grid


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def env_grid():
    """Free-growth sequence of up to 3 grids (single unique type)."""
    return Sequence(
        envs_unique=(Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
        max_sequence_length=3,
    )


@pytest.fixture
def env_two_grids():
    """Free-growth sequence of up to 3 elements, two distinct grid types."""
    return Sequence(
        envs_unique=(
            Grid(n_dim=1, length=3, cell_min=-1.0, cell_max=1.0),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
        ),
        max_sequence_length=3,
    )


@pytest.fixture
def env_grid_front_only():
    return Sequence(
        envs_unique=(Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
        max_sequence_length=3,
        front_only=True,
    )


@pytest.fixture
def env_grid_end_only():
    return Sequence(
        envs_unique=(Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),),
        max_sequence_length=3,
        end_only=True,
    )


@pytest.fixture
def env_grid_fixed_bag():
    """Fixed-bag mode with a randomly sampled bag on every reset."""
    return Sequence(
        envs_unique=(
            Grid(n_dim=1, length=3, cell_min=-1.0, cell_max=1.0),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
        ),
        max_sequence_length=3,
        do_random_subenvs=True,
    )


@pytest.fixture
def env_cube():
    """Free-growth sequence of up to 2 continuous cubes."""
    return Sequence(
        envs_unique=(ContinuousCube(n_dim=2, n_comp=2, min_incr=0.1),),
        max_sequence_length=2,
    )


@pytest.fixture
def env_cube_grid():
    """Free-growth sequence of up to 2 elements, a cube and a grid."""
    return Sequence(
        envs_unique=(
            ContinuousCube(n_dim=2, n_comp=2, min_incr=0.1),
            Grid(n_dim=2, length=3, cell_min=-1.0, cell_max=1.0),
        ),
        max_sequence_length=2,
    )


@pytest.fixture
def env_cube_fixed_bag():
    return Sequence(
        envs_unique=(ContinuousCube(n_dim=2, n_comp=2, min_incr=0.1),),
        max_sequence_length=2,
        do_random_subenvs=True,
    )


# --------------------------------------------------------------------------- #
# Sequence-specific unit tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "env_fixture, n_unique",
    [("env_grid", 1), ("env_two_grids", 2), ("env_cube_grid", 2)],
)
def test__action_space_size_is_expected(env_fixture, n_unique, request):
    env = request.getfixturevalue(env_fixture)
    assert env.n_unique_envs == n_unique
    # 3 * U insert actions + 1 EOS + the actions of each unique env
    n_meta = 3 * n_unique + 1
    n_subenv_actions = sum(
        env_unique.action_space_dim for env_unique in env.envs_unique
    )
    assert env.n_meta_actions == n_meta
    assert env.action_space_dim == n_meta + n_subenv_actions


@pytest.mark.parametrize(
    "env_fixture", ["env_grid", "env_two_grids", "env_grid_front_only", "env_cube"]
)
def test__source_is_empty_sequence(env_fixture, request):
    env = request.getfixturevalue(env_fixture)
    env.reset()
    assert env.is_source(env.state)
    assert env.state["_active"] == -1
    assert env.state["_envs_unique"] == []
    assert env.state["_indices"] == []
    parents, actions = env.get_parents()
    assert parents == [] and actions == []


def test__empty_source_only_allows_first_inserts(env_two_grids):
    env = env_two_grids
    env.reset()
    valid = env.get_valid_actions()
    U = env.n_unique_envs
    expected = [env._pad_action((env._insert_id(0, t),), -1) for t in range(U)]
    assert set(valid) == set(expected)
    # The global EOS must not be valid at the empty source
    assert env.eos not in valid


def test__ccab_insertion_order_example():
    """Reproduce the documented C, C, A, B construction (insert first/end/front/front)."""
    A = Grid(n_dim=1, length=3)
    B = Grid(n_dim=2, length=3)
    C = Grid(n_dim=3, length=3)
    env = Sequence(envs_unique=(A, B, C), max_sequence_length=4)
    tA, tB, tC = (env._compute_unique_indices_of_subenvs([e])[0] for e in (A, B, C))

    def complete_active():
        guard = 0
        while env.state["_active"] != -1:
            key = env._seq_length() - 1
            sub = env.subenvs[key]
            eos = env._pad_action(sub.eos, env.state["_envs_unique"][key])
            va = env.get_valid_actions()
            _, _, valid = env.step(eos if eos in va else va[0])
            assert valid
            guard += 1
            assert guard < 50

    env.reset()
    for d, t in [
        (0, tA),
        (2, tB),
        (1, tC),
        (1, tC),
    ]:  # first A, end B, front C, front C
        _, _, valid = env.step(env._pad_action((env._insert_id(d, t),), -1))
        assert valid
        complete_active()

    assert env.state["_indices"] == [3, 2, 0, 1]
    assert [env.state["_envs_unique"][k] for k in env.state["_indices"]] == [
        tC,
        tC,
        tA,
        tB,
    ]


@pytest.mark.parametrize(
    "env_fixture", ["env_grid", "env_two_grids", "env_grid_front_only"]
)
def test__state2readable__readable2state__roundtrip(env_fixture, request):
    env = request.getfixturevalue(env_fixture)
    for _ in range(20):
        env.reset()
        # advance a few random forward steps
        for _ in range(np.random.randint(1, env.max_traj_length)):
            if env.done:
                break
            env.step_random()
        state = env.state
        readable = env.state2readable(state)
        recovered = env.readable2state(readable)
        assert env.equal(recovered, state), f"\n{readable}\n{recovered}\n{state}"


def test__front_only_and_end_only_cannot_be_combined():
    with pytest.raises(ValueError):
        Sequence(
            envs_unique=(Grid(n_dim=2, length=3),),
            max_sequence_length=3,
            front_only=True,
            end_only=True,
        )


# --------------------------------------------------------------------------- #
# Common test battery
# --------------------------------------------------------------------------- #
_REPEATS = {
    "test__reset__state_is_source": 10,
    "test__forward_actions_have_nonzero_backward_prob": 10,
    "test__backward_actions_have_nonzero_forward_prob": 10,
    "test__trajectories_are_reversible": 10,
    "test__step_random__does_not_sample_invalid_actions_forward": 10,
    "test__step_random__does_not_sample_invalid_actions_backward": 10,
    "test__get_mask__is_consistent_regardless_of_inputs": 10,
    "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
    "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
    "test__get_parents_step_get_mask__are_compatible": 10,
    "test__sample_backwards_reaches_source": 10,
    "test__state2readable__is_reversible": 20,
    "test__gflownet_minimal_runs": 0,
}
_N_STATES = {
    "test__backward_actions_have_nonzero_forward_prob": 3,
    "test__sample_backwards_reaches_source": 3,
    "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
    "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
}
_BATCH_SIZE = {
    "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
    "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
    "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
    "test__gflownet_minimal_runs": 10,
}


class TestSequenceGrid(common.BaseTestsDiscrete):
    """Common tests for a free-growth sequence of up to three grids."""

    @pytest.fixture(autouse=True)
    def setup(self, env_grid):
        self.env = env_grid
        self.repeats = dict(_REPEATS)
        self.n_states = dict(_N_STATES)
        self.batch_size = dict(_BATCH_SIZE)


class TestSequenceTwoGrids(common.BaseTestsDiscrete):
    """Common tests for a free-growth sequence of two distinct grid types."""

    @pytest.fixture(autouse=True)
    def setup(self, env_two_grids):
        self.env = env_two_grids
        self.repeats = dict(_REPEATS)
        self.n_states = dict(_N_STATES)
        self.batch_size = dict(_BATCH_SIZE)


class TestSequenceGridFrontOnly(common.BaseTestsDiscrete):
    @pytest.fixture(autouse=True)
    def setup(self, env_grid_front_only):
        self.env = env_grid_front_only
        self.repeats = dict(_REPEATS)
        self.n_states = dict(_N_STATES)
        self.batch_size = dict(_BATCH_SIZE)


class TestSequenceGridEndOnly(common.BaseTestsDiscrete):
    @pytest.fixture(autouse=True)
    def setup(self, env_grid_end_only):
        self.env = env_grid_end_only
        self.repeats = dict(_REPEATS)
        self.n_states = dict(_N_STATES)
        self.batch_size = dict(_BATCH_SIZE)


class TestSequenceGridFixedBag(common.BaseTestsDiscrete):
    """Common tests for the fixed-bag (random) mode with grids."""

    @pytest.fixture(autouse=True)
    def setup(self, env_grid_fixed_bag):
        self.env = env_grid_fixed_bag
        self.repeats = dict(_REPEATS)
        self.n_states = dict(_N_STATES)
        self.batch_size = dict(_BATCH_SIZE)


class TestSequenceCube(common.BaseTestsContinuous):
    """Common tests for a free-growth sequence of up to two continuous cubes."""

    @pytest.fixture(autouse=True)
    def setup(self, env_cube):
        self.env = env_cube
        self.repeats = dict(_REPEATS)
        self.n_states = dict(_N_STATES)
        self.batch_size = dict(_BATCH_SIZE)


class TestSequenceCubeGrid(common.BaseTestsContinuous):
    """Common tests for a free-growth sequence of a cube and a grid."""

    @pytest.fixture(autouse=True)
    def setup(self, env_cube_grid):
        self.env = env_cube_grid
        self.repeats = dict(_REPEATS)
        self.n_states = dict(_N_STATES)
        self.batch_size = dict(_BATCH_SIZE)


class TestSequenceCubeFixedBag(common.BaseTestsContinuous):
    @pytest.fixture(autouse=True)
    def setup(self, env_cube_fixed_bag):
        self.env = env_cube_fixed_bag
        self.repeats = dict(_REPEATS)
        self.n_states = dict(_N_STATES)
        self.batch_size = dict(_BATCH_SIZE)
