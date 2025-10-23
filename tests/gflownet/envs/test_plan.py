import common
import pytest

from gflownet.envs.iam.plan import Plan


@pytest.fixture
def env():
    return Plan()


def test__environment_initializes_properly():
    env = Plan()
    assert True


@pytest.mark.parametrize(
    "state, techs_expected",
    [
        (
            [
                0,
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
            ],
            set(),
        ),
        (
            [
                2,
                {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                {"SECTOR": 4, "TAG": 1, "TECH": 15, "AMOUNT": 3},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
            ],
            set([15, 28, 20]),
        ),
        (
            [
                2,
                {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                {"SECTOR": 4, "TAG": 1, "TECH": 15, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
            ],
            set([15, 28, 20]),
        ),
        (
            [
                2,
                {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                {"SECTOR": 4, "TAG": 1, "TECH": 0, "AMOUNT": 3},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
            ],
            set([28, 20]),
        ),
    ],
)
def test__get_techs_set__returns_expected(env, state, techs_expected):
    assert env._get_techs_set(state) == techs_expected


@pytest.mark.parametrize(
    "states, expected_lengths, expected_first_dicts",
    [
        # Single state at source (stage 0, first investment only)
        (
                [
                    [
                        0,  # Stage
                        {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 1},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    ],
                ],
                [29],  # Expected length for first state
                [{"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 1}],  # First dict of first state
        ),
        # Two states with different stages
        (
                [
                    [
                        0,  # Stage
                        {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 1},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    ],
                    [
                        2,  # Stage
                        {"SECTOR": 1, "TAG": 1, "TECH": 20, "AMOUNT": 4},
                        {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                        {"SECTOR": 4, "TAG": 1, "TECH": 15, "AMOUNT": 3},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    ],
                ],
                [29, 29],  # Expected lengths
                [
                    {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 1},
                    {"SECTOR": 1, "TAG": 1, "TECH": 20, "AMOUNT": 4},
                ],  # First dicts
        ),
    ],
)
def test__states2proxy__filters_to_dicts_only(env, states, expected_lengths, expected_first_dicts):
    """Test that states2proxy returns only dictionary substates, filtering out stage integers."""
    processed = env.states2proxy(states)

    # Check batch size is preserved
    assert len(processed) == len(states)

    # Check each processed state
    for i, (proc_state, expected_len, expected_first) in enumerate(
            zip(processed, expected_lengths, expected_first_dicts)
    ):
        # Check length (number of investment dicts)
        assert len(proc_state) == expected_len

        # Check all elements are dicts
        assert all(isinstance(x, dict) for x in proc_state)

        # Check first dict matches expected
        assert proc_state[0] == expected_first

class TestPlan(common.BaseTestsDiscrete):
    """Common tests for the Plan environment."""

    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
        self.repeats = {
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
            "test__gflownet_minimal_runs": 1,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 5,
        }
