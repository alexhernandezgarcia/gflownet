import common
import pytest
import torch

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


@pytest.mark.repeat(10)
def test__trajectory_random__does_not_crash_from_source(env):
    env.reset()
    env.trajectory_random()
    assert True


@pytest.mark.parametrize(
    "state1, state2, should_be_equal",
    [
        # Same investments, different order - should be equal
        (
                [
                    2,
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                    {"SECTOR": 4, "TAG": 1, "TECH": 0, "AMOUNT": 0},
                ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26,
                [
                    2,
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 4, "TAG": 1, "TECH": 0, "AMOUNT": 0},
                ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26,
                True,
        ),
        # Three complete investments in different orders - should be equal
        (
                [
                    3,
                    {"SECTOR": 1, "TAG": 2, "TECH": 5, "AMOUNT": 2},
                    {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 3},
                    {"SECTOR": 3, "TAG": 3, "TECH": 15, "AMOUNT": 1},
                    {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 25,
                [
                    3,
                    {"SECTOR": 3, "TAG": 3, "TECH": 15, "AMOUNT": 1},
                    {"SECTOR": 1, "TAG": 2, "TECH": 5, "AMOUNT": 2},
                    {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 3},
                    {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 25,
                True,
        ),
        # Different investments - should NOT be equal
        (
                [
                    2,
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                    {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26,
                [
                    2,
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 3, "TAG": 1, "TECH": 15, "AMOUNT": 1},  # Different TECH
                    {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26,
                False,
        ),
        # Different amounts for same tech - should NOT be equal
        (
                [
                    2,
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                    {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26,
                [
                    2,
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 2},  # Different AMOUNT
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                    {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26,
                False,
        ),
        # Same investments in reverse order with partial state - should be equal
        (
                [
                    2,
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                    {"SECTOR": 4, "TAG": 2, "TECH": 15, "AMOUNT": 0},  # Partial investment
                ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26,
                [
                    2,
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 4, "TAG": 2, "TECH": 15, "AMOUNT": 0},  # Same partial investment
                ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26,
                True,
        ),
    ],
)
def test__states2policy__order_invariance(state1, state2, should_be_equal):
    """
    Test that states2policy produces the same encoding for states with the same
    investments in different orders, and different encodings for different investments.
    """
    env = Plan()

    # Convert states to policy representations
    encoding1 = env.states2policy([state1])
    encoding2 = env.states2policy([state2])

    assert encoding1.sum() == (4 + 29)

    # Check if encodings are equal
    are_equal = torch.allclose(encoding1, encoding2)

    assert are_equal == should_be_equal, (
        f"Expected encodings to be {'equal' if should_be_equal else 'different'}, "
        f"but they were {'equal' if are_equal else 'different'}.\n"
        f"State 1: {state1[:4]}\n"
        f"State 2: {state2[:4]}\n"
        f"Encoding 1: {encoding1}\n"
        f"Encoding 2: {encoding2}"
    )


def test__states2policy__batch_order_invariance():
    """
    Test that states2policy maintains order invariance when processing batches.
    """
    env = Plan()

    # Create a batch where each pair should have the same encoding
    states_batch1 = [
        [
            2,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 2, "TAG": 2, "TECH": 10, "AMOUNT": 3},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26,
        [
            1,
            {"SECTOR": 3, "TAG": 1, "TECH": 15, "AMOUNT": 1},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 27,
    ]

    states_batch2 = [
        [
            2,
            {"SECTOR": 2, "TAG": 2, "TECH": 10, "AMOUNT": 3},
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26,
        [
            1,
            {"SECTOR": 3, "TAG": 1, "TECH": 15, "AMOUNT": 1},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 27,
    ]

    encodings1 = env.states2policy(states_batch1)
    encodings2 = env.states2policy(states_batch2)

    # First state pair should have same encoding (different order)
    assert torch.allclose(encodings1[0], encodings2[0]), (
        "Encodings for reordered states should be identical"
    )

    # Second state pair should have same encoding (identical states)
    assert torch.allclose(encodings1[1], encodings2[1]), (
        "Encodings for identical states should be identical"
    )

def test_debug_backward_constraints_and_masks(env):
    """
    Debug test: Print tech availability, forward mask, and log-probs after each backward step.
    """
    # Get a terminating state with multiple investments
    states = env.get_random_terminating_states(n_states=1)
    state = states[0]

    # Set environment to terminating state
    env.reset()
    env.set_state(state, done=True)

    print("\nInitial terminating state:", state)

    # Perform backward steps until source
    while not env.is_source():
        prev_state = env.state.copy()
        state_next, action, valid = env.step_random(backward=True)
        print("\nBackward step:")
        print("Action:", action)
        print("Previous state:", prev_state)
        print("Current state:", state_next)
        print("Current stage:", env._get_stage())

        # Print tech availability for all subenvs
        for idx, subenv in env.subenvs.items():
            print(f"Stage {idx} techs_available:", subenv.techs_available)

        # Check forward mask after backward step
        mask_fw = env.get_mask_invalid_actions_forward()
        print("Forward mask (invalid actions):", mask_fw)
        print("Number of valid actions:", sum(not m for m in mask_fw))

        # Compute forward log-probs for the last backward action
        policy_random = torch.unsqueeze(env.random_policy_output, 0)
        masks = torch.unsqueeze(torch.tensor(mask_fw, dtype=torch.bool), 0)
        actions_torch = torch.unsqueeze(torch.tensor(action, dtype=torch.float), 0)
        logprobs_fw = env.get_logprobs(
            policy_outputs=policy_random,
            actions=actions_torch,
            mask=masks,
            states_from=[env.state],
            is_backward=False,
        )
        print("Forward log-probs for last action:", logprobs_fw)

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
