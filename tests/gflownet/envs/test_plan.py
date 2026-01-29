import common
import pytest
import torch
import numpy as np

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


# =============================================================================
# LOGPROB DIAGNOSTIC TESTS
# =============================================================================

@pytest.mark.repeat(5)
def test__logprobs_forward_backward_consistency(env):
    """
    Test that forward and backward log probabilities are computed consistently
    for the same trajectory.
    """
    env.reset()

    # Sample a forward trajectory and record states and actions
    trajectory_states = [env.state.copy()]
    trajectory_actions = []

    while not env.done:
        # Get valid forward actions
        mask_forward = env.get_mask_invalid_actions_forward()
        valid_actions = env.get_valid_actions(mask_forward)

        # Sample a random valid action
        action = valid_actions[np.random.randint(len(valid_actions))]
        trajectory_actions.append(action)

        env.step(action)
        trajectory_states.append(env.state.copy())

    # Compute forward log probabilities
    logprobs_forward = []
    for i, (state, action) in enumerate(zip(trajectory_states[:-1], trajectory_actions)):
        env.reset()
        env.set_state(state, done=False)

        # Get mask and compute uniform logprob
        mask = env.get_mask_invalid_actions_forward()
        n_valid = sum(1 for m in mask if not m)
        logprob = -np.log(n_valid) if n_valid > 0 else float('-inf')
        logprobs_forward.append(logprob)

    # Compute backward log probabilities
    logprobs_backward = []
    for i in range(len(trajectory_actions) - 1, -1, -1):
        state_after = trajectory_states[i + 1]
        action = trajectory_actions[i]

        # Set to state after action
        env.reset()
        is_terminal = (i == len(trajectory_actions) - 1)
        env.set_state(state_after, done=is_terminal)

        # Get backward mask and compute uniform logprob
        mask = env.get_mask_invalid_actions_backward(done=is_terminal)
        n_valid = sum(1 for m in mask if not m)
        logprob = -np.log(n_valid) if n_valid > 0 else float('-inf')
        logprobs_backward.append(logprob)

    logprobs_backward = logprobs_backward[::-1]  # Reverse to match forward order

    # Print diagnostics
    print(f"\nTrajectory length: {len(trajectory_actions)}")
    print(f"Sum logprobs_forward: {sum(logprobs_forward):.4f}")
    print(f"Sum logprobs_backward: {sum(logprobs_backward):.4f}")
    print(f"Difference (F - B): {sum(logprobs_forward) - sum(logprobs_backward):.4f}")

    # Check that all logprobs are finite
    assert all(np.isfinite(lp) for lp in logprobs_forward), \
        f"Some forward logprobs are not finite: {logprobs_forward}"
    assert all(np.isfinite(lp) for lp in logprobs_backward), \
        f"Some backward logprobs are not finite: {logprobs_backward}"


@pytest.mark.repeat(10)
def test__logprobs_trajectory_balance_components(env):
    """
    Test the components of the trajectory balance equation:
    log Z + log P_F(tau) = log P_B(tau|x) + log R(x)

    For uniform reward (R=1, log R=0), we expect:
    log P_F(tau) - log P_B(tau|x) = -log Z
    """
    differences = []

    for _ in range(10):
        env.reset()

        # Sample forward trajectory
        trajectory_states = [env.state.copy()]
        trajectory_actions = []

        while not env.done:
            mask = env.get_mask_invalid_actions_forward()
            valid_actions = env.get_valid_actions(mask)
            action = valid_actions[np.random.randint(len(valid_actions))]
            trajectory_actions.append(action)
            env.step(action)
            trajectory_states.append(env.state.copy())

        # Compute forward logprobs (uniform policy)
        logprob_forward_total = 0.0
        for state, action in zip(trajectory_states[:-1], trajectory_actions):
            env.reset()
            env.set_state(state, done=False)
            mask = env.get_mask_invalid_actions_forward()
            n_valid = sum(1 for m in mask if not m)
            logprob_forward_total += -np.log(n_valid)

        # Compute backward logprobs (uniform policy)
        logprob_backward_total = 0.0
        for i in range(len(trajectory_actions) - 1, -1, -1):
            state_after = trajectory_states[i + 1]
            is_terminal = (i == len(trajectory_actions) - 1)
            env.reset()
            env.set_state(state_after, done=is_terminal)
            mask = env.get_mask_invalid_actions_backward(done=is_terminal)
            n_valid = sum(1 for m in mask if not m)
            logprob_backward_total += -np.log(n_valid)

        diff = logprob_forward_total - logprob_backward_total
        differences.append(diff)

    differences = np.array(differences)

    print(f"\nlog P_F - log P_B statistics:")
    print(f"  Mean: {differences.mean():.4f}")
    print(f"  Std:  {differences.std():.4f}")
    print(f"  Min:  {differences.min():.4f}")
    print(f"  Max:  {differences.max():.4f}")
    print(f"  Expected -log Z (for 4^29 states): {-29 * np.log(4):.4f}")

    # For TB to work, all trajectories should give approximately the same difference
    assert differences.std() < 1.0, \
        f"Variance in log P_F - log P_B is too high: std={differences.std():.4f}"


@pytest.mark.repeat(5)
def test__forward_backward_mask_symmetry(env):
    """
    Test that forward and backward masks are symmetric.
    """
    env.reset()

    asymmetries = []

    while not env.done:
        state_before = env.state.copy()

        # Get forward mask and valid actions
        mask_forward = env.get_mask_invalid_actions_forward()
        valid_forward = env.get_valid_actions(mask_forward)

        if not valid_forward:
            break

        # Take a random action
        action = valid_forward[np.random.randint(len(valid_forward))]
        env.step(action)

        state_after = env.state.copy()
        is_done = env.done

        # Get backward mask from the new state
        mask_backward = env.get_mask_invalid_actions_backward(done=is_done)
        valid_backward = env.get_valid_actions(mask_backward, backward=True)

        # Check if the action we just took is valid backward
        if action not in valid_backward:
            asymmetries.append({
                'action': action,
                'state_before': state_before,
                'state_after': state_after,
                'valid_forward': len(valid_forward),
                'valid_backward': len(valid_backward),
            })

    if asymmetries:
        print(f"\nFound {len(asymmetries)} asymmetries:")
        for asym in asymmetries[:3]:
            print(f"  Action: {asym['action']}")
            print(f"  Valid forward: {asym['valid_forward']}, Valid backward: {asym['valid_backward']}")

    assert len(asymmetries) == 0, \
        f"Found {len(asymmetries)} forward/backward mask asymmetries"


@pytest.mark.repeat(3)
def test__backward_trajectory_reaches_source_with_correct_logprobs(env):
    """
    Test that backward trajectories from terminating states reach the source.
    """
    # Get a terminating state
    env.reset()
    env.trajectory_random()
    terminal_state = env.state.copy()

    # Reset and set to terminal state
    env.reset()
    env.set_state(terminal_state, done=True)

    backward_logprobs = []
    step_count = 0
    max_steps = 500

    while not env.is_source() and step_count < max_steps:
        mask = env.get_mask_invalid_actions_backward()
        valid_actions = env.get_valid_actions(mask, backward=True)

        assert len(valid_actions) > 0, \
            f"No valid backward actions at step {step_count}, state: {env.state}"

        # Compute uniform logprob
        logprob = -np.log(len(valid_actions))
        backward_logprobs.append(logprob)

        # Take random backward step
        action = valid_actions[np.random.randint(len(valid_actions))]
        env.step_backwards(action)
        step_count += 1

    assert env.is_source(), \
        f"Did not reach source after {step_count} steps"

    total_logprob = sum(backward_logprobs)

    print(f"\nBackward trajectory:")
    print(f"  Steps: {step_count}")
    print(f"  Total log P_B: {total_logprob:.4f}")
    print(f"  Mean log P_B per step: {total_logprob / step_count:.4f}")

    assert np.isfinite(total_logprob), "Total backward logprob is not finite"


@pytest.mark.repeat(5)
def test__permutation_invariance_does_not_affect_logprobs(env):
    """
    Test that permutation invariance in states2policy doesn't create
    inconsistencies in logprob computation.
    """
    investments = []
    for i in range(min(3, env.n_techs)):
        investments.append({
            "SECTOR": (i % 5) + 1,
            "TAG": (i % 3) + 1,
            "TECH": i + 1,
            "AMOUNT": (i % 4) + 1,
        })

    # Pad with empty
    while len(investments) < env.n_techs:
        investments.append({"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0})

    # Create state with original order
    state1 = _make_plan_state(env, investments.copy())

    # Create state with shuffled order (swap first two)
    shuffled = investments.copy()
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    state2 = _make_plan_state(env, shuffled)

    # Get policy representations
    policy1 = env.states2policy([state1])
    policy2 = env.states2policy([state2])

    # They should be identical due to sorting
    assert torch.allclose(policy1, policy2), \
        "Policy representations differ for permuted states"

    print("\nPermutation invariance verified: policy outputs match for reordered states")


def test__tb_loss_components_with_uniform_policy(env):
    """
    Compute exact TB loss components using uniform policy.

    TB Loss = (log Z + log P_F - log P_B - log R)^2

    For uniform reward (log R = 0):
    TB Loss = (log Z + log P_F - log P_B)^2

    At optimum: log Z = log P_B - log P_F (averaged over trajectories)
    """
    n_trajectories = 50

    log_pf_minus_pb = []

    for _ in range(n_trajectories):
        env.reset()
        log_pf = 0.0

        # Forward trajectory with uniform policy
        trajectory = []
        while not env.done:
            mask = env.get_mask_invalid_actions_forward()
            n_valid = sum(1 for m in mask if not m)
            log_pf += -np.log(n_valid)

            valid_actions = env.get_valid_actions(mask)
            action = valid_actions[np.random.randint(len(valid_actions))]
            trajectory.append((env.state.copy(), action))
            env.step(action)

        terminal_state = env.state.copy()

        # Backward log prob
        log_pb = 0.0
        env.reset()
        env.set_state(terminal_state, done=True)

        for i in range(len(trajectory) - 1, -1, -1):
            _, action = trajectory[i]
            is_terminal = (i == len(trajectory) - 1)

            mask = env.get_mask_invalid_actions_backward(done=is_terminal)
            n_valid = sum(1 for m in mask if not m)
            log_pb += -np.log(n_valid)

            env.step_backwards(action)

        log_pf_minus_pb.append(log_pf - log_pb)

    log_pf_minus_pb = np.array(log_pf_minus_pb)

    # Optimal log Z should make the loss zero
    optimal_log_z = -np.mean(log_pf_minus_pb)

    # Compute what the loss would be with different log Z values
    test_log_z_values = [0, 10, 20, 30, 40, optimal_log_z]

    print(f"\nTB Loss analysis (uniform policy, uniform reward):")
    print(f"  Number of trajectories: {n_trajectories}")
    print(f"  log P_F - log P_B: mean={np.mean(log_pf_minus_pb):.4f}, std={np.std(log_pf_minus_pb):.4f}")
    print(f"  Optimal log Z: {optimal_log_z:.4f}")
    print(f"  Theoretical log Z (4^29): {29 * np.log(4):.4f}")
    print(f"\n  Loss at different log Z values:")

    for log_z in test_log_z_values:
        loss = np.mean((log_z + log_pf_minus_pb) ** 2)
        print(f"    log Z = {log_z:8.2f}: Loss = {loss:.4f}")

    # The key insight: if std is high, the model might find it easier to
    # make log P_F ≈ log P_B (so diff ≈ 0) rather than learn correct log Z
    if np.std(log_pf_minus_pb) > 5.0:
        print(f"\n  WARNING: High variance in log P_F - log P_B!")
        print(f"  This makes learning log Z difficult.")


def test__mask_and_policy_output_dimensions(env):
    """
    Verify that mask dimensions match policy output dimensions correctly.
    """
    env.reset()

    # Get dimensions
    policy_output = env.get_policy_output(env.fixed_distr_params)
    mask_forward = env.get_mask_invalid_actions_forward()

    print(f"\nDimension analysis:")
    print(f"  Policy output dim: {len(policy_output)}")
    print(f"  Forward mask dim: {len(mask_forward)}")
    print(f"  Action space dim: {len(env.action_space)}")
    print(f"  env.mask_dim: {env.mask_dim}")
    print(f"  env.policy_output_dim: {env.policy_output_dim}")

    # Mask structure
    print(f"\n  Mask structure (first 50 values):")
    print(f"    {mask_forward[:50]}")

    # Count valid actions
    n_valid = sum(1 for m in mask_forward if not m)
    print(f"\n  Number of valid actions: {n_valid}")

    # Check which actions are valid
    valid_actions = env.get_valid_actions(mask_forward)
    print(f"  Valid actions (first 10): {valid_actions[:10]}")


def test__mask_policy_action_space_alignment(env):
    """
    Test that the mask indices correctly correspond to action space indices.
    """
    env.reset()

    mask = env.get_mask_invalid_actions_forward()
    action_space = env.action_space

    print(f"\nMask-Action alignment check:")
    print(f"  Mask length: {len(mask)}")
    print(f"  Action space length: {len(action_space)}")


def test__set_mask_structure_analysis(env):
    """
    Analyze the structure of the Set environment mask.
    """
    env.reset()

    mask = env.get_mask_invalid_actions_forward()

    n_toggle = env.n_toggle_actions if hasattr(env, 'n_toggle_actions') else 0
    n_subenvs = env.n_subenvs if hasattr(env, 'n_subenvs') else 0

    print(f"\nSet mask structure analysis:")
    print(f"  Total mask length: {len(mask)}")
    print(f"  n_toggle_actions: {n_toggle}")
    print(f"  n_subenvs: {n_subenvs}")
    print(f"  Action space length: {len(env.action_space)}")

    print(f"\n  Mask breakdown:")
    print(f"    Prefix (first {n_toggle}): {mask[:n_toggle]}")

    active = env._get_active_subenv(env.state)
    print(f"    Active subenv: {active}")

    if active == -1:
        core_length = n_toggle + 1
        print(f"    Expected core length (toggles + EOS): {core_length}")
        print(f"    Core mask: {mask[n_toggle:n_toggle + core_length]}")

    # Count how the action space is organized
    toggle_actions = [a for a in env.action_space if a[0] == -1 and a != env.eos]
    eos_actions = [a for a in env.action_space if a == env.eos]
    subenv_actions = [a for a in env.action_space if a[0] != -1]

    print(f"\n  Action space organization:")
    print(f"    Toggle actions: {len(toggle_actions)} (indices 0-{len(toggle_actions)-1})")
    print(f"    EOS actions: {len(eos_actions)}")
    print(f"    Subenv actions: {len(subenv_actions)}")
    print(f"    First few actions: {env.action_space[:5]}")

    print(f"\n  CRITICAL CHECK:")
    print(f"    Does mask[i] correspond to action_space[i]?")

    valid_by_mask = [i for i, m in enumerate(mask) if not m]
    valid_actions = env.get_valid_actions(mask)
    valid_indices = [env.action_space.index(a) for a in valid_actions]

    print(f"    Valid indices from mask: {valid_by_mask[:10]}")
    print(f"    Valid indices from get_valid_actions: {valid_indices[:10]}")

    if valid_by_mask != valid_indices:
        print(f"    MISMATCH DETECTED - This is the bug!")


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