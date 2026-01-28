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


def _make_plan_state(env, investments, active=-1, toggle=0):
    """
    Helper function to create a Plan state with the new SetFix structure.

    Parameters
    ----------
    env : Plan
        The Plan environment instance.
    investments : list of dict
        List of investment dictionaries with keys SECTOR, TAG, TECH, AMOUNT.
        Length must equal env.n_techs.
    active : int
        The active sub-environment index (-1 if none active).
    toggle : int
        The toggle flag (0 or 1).

    Returns
    -------
    dict
        A state in the new SetFix format.
    """
    assert len(investments) == env.n_techs

    # Determine which sub-environments are done
    dones = []
    for inv in investments:
        is_done = (inv["SECTOR"] != 0 and inv["TAG"] != 0 and
                   inv["TECH"] != 0 and inv["AMOUNT"] != 0)
        dones.append(1 if is_done else 0)

    state = {
        "_active": active,
        "_toggle": toggle,
        "_dones": dones,
        "_envs_unique": [0] * env.n_techs,  # All subenvs are of same unique type (index 0)
    }

    # Add substates
    for idx, inv in enumerate(investments):
        state[idx] = inv.copy()

    return state


def _empty_investments(n_techs):
    """Create a list of empty investment dictionaries."""
    return [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0} for _ in range(n_techs)]


@pytest.mark.parametrize(
    "investments, techs_expected",
    [
        # All empty investments
        (
                None,  # Will be replaced with empty investments
                set(),
        ),
        # Three investments with techs assigned
        (
                [
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                    {"SECTOR": 4, "TAG": 1, "TECH": 15, "AMOUNT": 3},
                ],  # Rest will be filled with empty
                {15, 28, 20},
        ),
        # Three investments, one without amount (still has tech)
        (
                [
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                    {"SECTOR": 4, "TAG": 1, "TECH": 15, "AMOUNT": 0},
                ],
                {15, 28, 20},
        ),
        # Two investments with tech, one with only sector/tag/amount
        (
                [
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                    {"SECTOR": 4, "TAG": 1, "TECH": 0, "AMOUNT": 3},
                ],
                {28, 20},
        ),
    ],
)
def test__get_techs_set__returns_expected(env, investments, techs_expected):
    if investments is None:
        investments = _empty_investments(env.n_techs)
    else:
        # Pad with empty investments
        while len(investments) < env.n_techs:
            investments.append({"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0})

    state = _make_plan_state(env, investments)
    assert env._get_techs_set(state) == techs_expected


@pytest.mark.repeat(10)
def test__trajectory_random__does_not_crash_from_source(env):
    env.reset()
    env.trajectory_random()
    assert True


@pytest.mark.repeat(100)
def test__trajectory_lengths_are_consistent(env):
    """
    Test that all trajectories from source to terminating state have the same length.

    In the Plan environment, each trajectory should:
    1. Toggle each sub-environment once to activate it
    2. Perform exactly 4 actions per sub-environment (SECTOR, TAG, TECH, AMOUNT)
    3. Perform EOS for each sub-environment
    4. Toggle each sub-environment once to deactivate it
    5. Perform global EOS

    Since can_alternate_subenvs=False, the expected pattern for each of n_techs
    sub-environments is: toggle_on, 4 actions, EOS, toggle_off
    Plus final global EOS.

    Total = n_techs * (1 + 4 + 1 + 1) + 1 = n_techs * 7 + 1
    But actually the sub-env EOS is one of the 4 actions boundary... let's just
    verify consistency across multiple trajectories.
    """
    trajectory_lengths = []

    for _ in range(100):
        env.reset()
        _, actions = env.trajectory_random()
        trajectory_lengths.append(len(actions))

    # All trajectories should have the same length
    assert len(set(trajectory_lengths)) == 1, (
        f"Trajectory lengths are inconsistent: {trajectory_lengths}"
    )


def test__trajectory_length_matches_expected(env):
    """
    Test that trajectory length matches the expected value based on the environment structure.

    For Plan with can_alternate_subenvs=False:
    - Each sub-environment requires: toggle_on (1) + subenv_actions + toggle_off (1)
    - Each InvestmentDiscrete sub-environment has max 5 actions (4 choices + EOS)
    - Final global EOS (1)

    The exact length depends on the trajectory taken through each sub-environment.
    """
    env.reset()
    _, actions = env.trajectory_random()

    # Count the different types of actions
    toggle_actions = sum(1 for a in actions if a[0] == -1 and a != env.eos)
    global_eos = sum(1 for a in actions if a == env.eos)
    subenv_actions = len(actions) - toggle_actions - global_eos

    # We should have exactly 2 * n_techs toggle actions (on and off for each)
    assert toggle_actions == 2 * env.n_techs, (
        f"Expected {2 * env.n_techs} toggle actions, got {toggle_actions}"
    )

    # We should have exactly 1 global EOS
    assert global_eos == 1, f"Expected 1 global EOS, got {global_eos}"

    # Each sub-environment should contribute its actions
    # InvestmentDiscrete has max_traj_length of 5 (4 choices + EOS)
    assert subenv_actions == env.n_techs * 5, (
        f"Expected {env.n_techs * 5} sub-environment actions, got {subenv_actions}"
    )


@pytest.mark.parametrize(
    "investments1, investments2, should_be_equal",
    [
        # Same investments, different order - should be equal
        (
                [
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                    {"SECTOR": 4, "TAG": 1, "TECH": 15, "AMOUNT": 3},
                ],
                [
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 4, "TAG": 1, "TECH": 15, "AMOUNT": 3},
                ],
                True,
        ),
        # Different investments - should NOT be equal
        (
                [
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                ],
                [
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 3, "TAG": 1, "TECH": 15, "AMOUNT": 1},  # Different TECH
                ],
                False,
        ),
        # Different amounts for same tech - should NOT be equal
        (
                [
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                ],
                [
                    {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 2},  # Different AMOUNT
                    {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                ],
                False,
        ),
    ],
)
def test__states2policy__order_invariance(investments1, investments2, should_be_equal):
    """
    Test that states2policy produces the same encoding for states with the same
    investments in different orders, and different encodings for different investments.
    """
    env = Plan()

    # Pad investments to full length
    while len(investments1) < env.n_techs:
        investments1.append({"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0})
    while len(investments2) < env.n_techs:
        investments2.append({"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0})

    state1 = _make_plan_state(env, investments1)
    state2 = _make_plan_state(env, investments2)

    # Convert states to policy representations
    encoding1 = env.states2policy([state1])
    encoding2 = env.states2policy([state2])

    # Check if encodings are equal
    are_equal = torch.allclose(encoding1, encoding2)

    assert are_equal == should_be_equal, (
        f"Expected encodings to be {'equal' if should_be_equal else 'different'}, "
        f"but they were {'equal' if are_equal else 'different'}.\n"
        f"Investments 1: {investments1[:4]}\n"
        f"Investments 2: {investments2[:4]}\n"
        f"Encoding 1 shape: {encoding1.shape}\n"
        f"Encoding 2 shape: {encoding2.shape}"
    )


def test__states2policy__batch_order_invariance():
    """
    Test that states2policy maintains order invariance when processing batches.
    """
    env = Plan()

    # Create investments for batch
    inv_batch1 = [
        [
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 2, "TAG": 2, "TECH": 10, "AMOUNT": 3},
        ],
        [
            {"SECTOR": 3, "TAG": 1, "TECH": 15, "AMOUNT": 1},
        ],
    ]

    inv_batch2 = [
        [
            {"SECTOR": 2, "TAG": 2, "TECH": 10, "AMOUNT": 3},
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
        ],
        [
            {"SECTOR": 3, "TAG": 1, "TECH": 15, "AMOUNT": 1},
        ],
    ]

    # Pad and create states
    states_batch1 = []
    states_batch2 = []

    for inv1, inv2 in zip(inv_batch1, inv_batch2):
        while len(inv1) < env.n_techs:
            inv1.append({"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0})
        while len(inv2) < env.n_techs:
            inv2.append({"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0})
        states_batch1.append(_make_plan_state(env, inv1))
        states_batch2.append(_make_plan_state(env, inv2))

    encodings1 = env.states2policy(states_batch1)
    encodings2 = env.states2policy(states_batch2)

    # First state pair should have same encoding (different order)
    assert torch.allclose(
        encodings1[0], encodings2[0]
    ), "Encodings for reordered states should be identical"

    # Second state pair should have same encoding (identical states)
    assert torch.allclose(
        encodings1[1], encodings2[1]
    ), "Encodings for identical states should be identical"


def test__source_state_structure(env):
    """Test that the source state has the correct structure."""
    env.reset()
    state = env.state

    # Check meta-data keys
    assert "_active" in state
    assert "_toggle" in state
    assert "_dones" in state
    assert "_envs_unique" in state

    # Check meta-data values for source
    assert state["_active"] == -1
    assert state["_toggle"] == 0
    assert state["_dones"] == [0] * env.n_techs
    assert len(state["_envs_unique"]) == env.n_techs

    # Check substates exist
    for idx in range(env.n_techs):
        assert idx in state
        substate = state[idx]
        assert substate == {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}


def test__terminating_state_structure(env):
    """Test that terminating states have the correct structure."""
    env.reset()
    env.trajectory_random()
    state = env.state

    # Check that all sub-environments are done
    assert all(d == 1 for d in state["_dones"]), "All sub-environments should be done"

    # Check that each substate has all fields filled
    for idx in range(env.n_techs):
        substate = state[idx]
        assert substate["SECTOR"] != 0, f"Sub-environment {idx} should have SECTOR set"
        assert substate["TAG"] != 0, f"Sub-environment {idx} should have TAG set"
        assert substate["TECH"] != 0, f"Sub-environment {idx} should have TECH set"
        assert substate["AMOUNT"] != 0, f"Sub-environment {idx} should have AMOUNT set"


def test__all_techs_used_in_terminating_state(env):
    """Test that all technologies are unique in a terminating state."""
    env.reset()
    env.trajectory_random()
    state = env.state

    techs = set()
    for idx in range(env.n_techs):
        tech = state[idx]["TECH"]
        assert tech not in techs, f"Technology {tech} is used more than once"
        techs.add(tech)

    assert len(techs) == env.n_techs, "All technologies should be unique"


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

    print("\nInitial terminating state:")
    print(f"Active: {state['_active']}, Toggle: {state['_toggle']}")
    print(f"Dones: {state['_dones']}")

    # Perform backward steps until source
    step_count = 0
    while not env.is_source():
        prev_state = env.state.copy()
        state_next, action, valid = env.step_random(backward=True)
        step_count += 1

        print(f"\nBackward step {step_count}:")
        print(f"Action: {action}")
        print(f"Active: {env.state['_active']}, Toggle: {env.state['_toggle']}")

        # Print tech availability for all subenvs
        for idx, subenv in enumerate(env.subenvs):
            print(f"  Subenv {idx} techs_available: {len(subenv.techs_available)} techs")

        # Check forward mask after backward step
        mask_fw = env.get_mask_invalid_actions_forward()
        n_valid = sum(not m for m in mask_fw)
        print(f"Number of valid forward actions: {n_valid}")


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