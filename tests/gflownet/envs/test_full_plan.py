import common
import pytest

from gflownet.envs.iam.full_plan import FullPlan


@pytest.fixture
def env():
    return FullPlan()


def test__environment_initializes_properly():
    env = FullPlan()
    assert True


def test__get_right_mask__partial_complete_only_lock_valid(env):
    """When partial is complete, only the LOCK action for that tech should be valid."""
    # Complete partial for tech 3 (power_NUCLEAR)
    state = {
        "partial": {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 1},
        "plan": [0] * env.n_techs,
    }
    mask = env.get_mask_invalid_actions_forward(state)

    # Only LOCK for tech 3 should be valid
    lock_action = (env.token2idx_choices["LOCK"], 3)
    lock_idx = env.action2index(lock_action)

    for i, m in enumerate(mask):
        if i == lock_idx:
            assert m == False, f"LOCK action for tech 3 should be valid"
        else:
            assert m == True, f"Action {env.action_space[i]} should be invalid"


def test__get_right_mask__tech_set_forces_sector(env):
    """When TECH is set but SECTOR is not, only the forced SECTOR action should be valid."""
    # Tech 3 is power_NUCLEAR which belongs to POWER sector
    state = {
        "partial": {"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 0},
        "plan": [0] * env.n_techs,
    }
    mask = env.get_mask_invalid_actions_forward(state)

    # Only SECTOR=POWER (idx 1) should be valid
    forced_action = (env.token2idx_choices["SECTOR"], 1)
    forced_idx = env.action2index(forced_action)

    for i, m in enumerate(mask):
        if i == forced_idx:
            assert m == False, f"Forced SECTOR action should be valid"
        else:
            assert m == True, f"Action {env.action_space[i]} should be invalid"


def test__get_right_mask__tech_set_forces_tag(env):
    """When TECH and SECTOR are set but TAG is not, only the forced TAG action should be valid."""
    # Tech 3 is power_NUCLEAR which belongs to GREEN tag
    state = {
        "partial": {"SECTOR": 1, "TAG": 0, "TECH": 3, "AMOUNT": 0},
        "plan": [0] * env.n_techs,
    }
    mask = env.get_mask_invalid_actions_forward(state)

    # Only TAG=GREEN (idx 1) should be valid
    forced_action = (env.token2idx_choices["TAG"], 1)
    forced_idx = env.action2index(forced_action)

    for i, m in enumerate(mask):
        if i == forced_idx:
            assert m == False, f"Forced TAG action should be valid"
        else:
            assert m == True, f"Action {env.action_space[i]} should be invalid"


@pytest.mark.parametrize(
    "action_space_structure",
    [
        {
            "n_sector_actions": 5,  # POWER, ENERGY, VEHICLES, STORAGE, DAC
            "n_tag_actions": 3,  # GREEN, BROWN, CCS
            "n_tech_actions": 29,  # All technologies
            "n_amount_actions": 4,  # HIGH, MEDIUM, LOW, NONE
            "n_lock_actions": 29,  # One LOCK per tech
            "has_eos": True,
        },
    ],
)
def test__get_action_space__returns_expected(env, action_space_structure):
    expected_len = (
            action_space_structure["n_sector_actions"]
            + action_space_structure["n_tag_actions"]
            + action_space_structure["n_tech_actions"]
            + action_space_structure["n_amount_actions"]
            + action_space_structure["n_lock_actions"]
            + (1 if action_space_structure["has_eos"] else 0)
    )
    assert len(env.action_space) == expected_len
    assert env.eos in env.action_space

    # Check action types are present
    sector_actions = [(1, i) for i in range(1, env.n_sectors + 1)]
    tag_actions = [(2, i) for i in range(1, env.n_tags + 1)]
    tech_actions = [(3, i) for i in range(1, env.n_techs + 1)]
    amount_actions = [(4, i) for i in range(1, env.n_amounts + 1)]
    lock_actions = [(5, i) for i in range(1, env.n_techs + 1)]

    for a in sector_actions:
        assert a in env.action_space
    for a in tag_actions:
        assert a in env.action_space
    for a in tech_actions:
        assert a in env.action_space
    for a in amount_actions:
        assert a in env.action_space
    for a in lock_actions:
        assert a in env.action_space


@pytest.mark.parametrize(
    "state, parents_expected, parents_a_expected",
    [
        # Source state has no parents
        (
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                [],
                [],
        ),
        # State with only SECTOR assigned in partial
        (
                {
                    "partial": {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                [
                    {
                        "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                        "plan": [0] * 29,
                    }
                ],
                [(1, 1)],  # SECTOR=POWER action
        ),
        # State with TECH and AMOUNT in partial (can only undo TECH, not AMOUNT due to forced ordering)
        (
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 1},
                    "plan": [0] * 29,
                },
                [
                    {
                        "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 1},
                        "plan": [0] * 29,
                    },
                ],
                [(3, 3)],  # TECH=power_NUCLEAR action
        ),
        # State with SECTOR, TECH, and AMOUNT in partial
        (
                {
                    "partial": {"SECTOR": 1, "TAG": 0, "TECH": 3, "AMOUNT": 1},
                    "plan": [0] * 29,
                },
                [
                    {
                        "partial": {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 1},
                        "plan": [0] * 29,
                    },
                    {
                        "partial": {"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 1},
                        "plan": [0] * 29,
                    },
                ],
                [(3, 3), (1, 1)],
        ),
        # State with SECTOR, TAG, and AMOUNT in partial (no TECH)
        (
                {
                    "partial": {"SECTOR": 1, "TAG": 2, "TECH": 0, "AMOUNT": 1},
                    "plan": [0] * 29,
                },
                [
                    {
                        "partial": {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 1},
                        "plan": [0] * 29,
                    },
                    {
                        "partial": {"SECTOR": 0, "TAG": 2, "TECH": 0, "AMOUNT": 1},
                        "plan": [0] * 29,
                    },
                    {
                        "partial": {"SECTOR": 1, "TAG": 2, "TECH": 0, "AMOUNT": 0},
                        "plan": [0] * 29,
                    },
                ],
                [(2, 2), (1, 1), (4, 1)],
        ),
        # Empty partial with one tech locked in plan - parent is state with that tech in partial
        (
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    "plan": [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    # Tech 3 has amount 2
                },
                [
                    {
                        "partial": {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 2},  # power_NUCLEAR is POWER/GREEN
                        "plan": [0] * 29,
                    }
                ],
                [(5, 3)],  # LOCK for tech 3
        ),
    ],
)
def test__get_parents__returns_expected(env, state, parents_expected, parents_a_expected):
    parents, parents_a = env.get_parents(state)
    assert len(parents) == len(parents_expected)
    assert len(parents_a) == len(parents_a_expected)

    # Create sets of (parent_tuple, action) pairs for robust comparison
    def state_to_tuple(s):
        partial_tuple = tuple(sorted(s["partial"].items()))
        plan_tuple = tuple(s["plan"])
        return (partial_tuple, plan_tuple)

    actual_pairs = set(
        (state_to_tuple(p), a) for p, a in zip(parents, parents_a)
    )
    expected_pairs = set(
        (state_to_tuple(p), a) for p, a in zip(parents_expected, parents_a_expected)
    )

    assert actual_pairs == expected_pairs


@pytest.mark.parametrize(
    "state, action, next_state, valid",
    [
        # From source, assign SECTOR
        (
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                (1, 1),  # SECTOR=POWER
                {
                    "partial": {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                True,
        ),
        # From source, assign TAG
        (
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                (2, 1),  # TAG=GREEN
                {
                    "partial": {"SECTOR": 0, "TAG": 1, "TECH": 0, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                True,
        ),
        # From source, assign TECH directly
        (
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                (3, 3),  # TECH=power_NUCLEAR
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                True,
        ),
        # Assign AMOUNT from TECH -> FALSE, should fill in sector and tag first
        (
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 7, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                (4, 1),  # AMOUNT=HIGH
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 7, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                False,
        ),
        # Fill in SECTOR when TECH is set
        (
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                (1, 1),  # SECTOR=POWER (correct for power_NUCLEAR)
                {
                    "partial": {"SECTOR": 1, "TAG": 0, "TECH": 3, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                True,
        ),
        # Fill in TAG before SECTOR when TECH is set -> FALSE
        (
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                (2, 1),  # TAG=GREEN
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 3, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                False,
        ),
        # Fill in TAG after SECTOR when TECH is set
        (
                {
                    "partial": {"SECTOR": 1, "TAG": 0, "TECH": 3, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                (2, 1),  # TAG=GREEN (correct for power_NUCLEAR)
                {
                    "partial": {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                True,
        ),
        # LOCK action on complete partial
        (
                {
                    "partial": {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 2},
                    "plan": [0] * 29,
                },
                (5, 3),  # LOCK tech 3
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    "plan": [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                True,
        ),
        # LOCK action on incomplete partial -> FALSE
        (
                {
                    "partial": {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                (5, 3),  # LOCK tech 3
                {
                    "partial": {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                False,
        ),
        # LOCK action with wrong tech index -> FALSE
        (
                {
                    "partial": {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 2},
                    "plan": [0] * 29,
                },
                (5, 5),  # LOCK tech 5 (but partial has tech 3)
                {
                    "partial": {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 2},
                    "plan": [0] * 29,
                },
                False,
        ),
        # Invalid: Reassigning SECTOR
        (
                {
                    "partial": {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                (1, 2),  # Try to change SECTOR to ENERGY
                {
                    "partial": {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                False,
        ),
        # Invalid: incompatible Tech (nuclear on brown)
        (
                {
                    "partial": {"SECTOR": 1, "TAG": 2, "TECH": 0, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                (3, 3),  # TECH=power_NUCLEAR (GREEN, not BROWN)
                {
                    "partial": {"SECTOR": 1, "TAG": 2, "TECH": 0, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                False,
        ),
        # Invalid: selecting tech that's already in plan
        (
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    "plan": [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    # Tech 3 already assigned
                },
                (3, 3),  # TECH=power_NUCLEAR (already in plan)
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    "plan": [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                False,
        ),
    ],
)
def test__step__returns_expected(env, state, action, next_state, valid):
    env.set_state(state)
    result_state, result_action, result_valid = env.step(action)
    assert env.equal(result_state, next_state), f"State mismatch: {result_state} != {next_state}"
    assert result_action == action
    assert result_valid == valid


@pytest.mark.parametrize(
    "state, readable",
    [
        (
                {
                    "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                "PARTIAL: UNASSIGNED | UNASSIGNED | UNASSIGNED | UNASSIGNED",
        ),
        (
                {
                    "partial": {"SECTOR": 1, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                    "plan": [0] * 29,
                },
                "PARTIAL: POWER | UNASSIGNED | UNASSIGNED | UNASSIGNED",
        ),
        (
                {
                    "partial": {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 1},
                    "plan": [0] * 29,
                },
                "PARTIAL: POWER | GREEN | power_NUCLEAR | HIGH",
        ),
    ],
)
def test__state2readable__partial_component(env, state, readable):
    """Test that the partial component of state2readable is correct."""
    result = env.state2readable(state)
    # Check that the partial line is correct
    partial_line = result.split("\n")[0]
    assert partial_line == readable


@pytest.mark.parametrize(
    "partial_empty, plan_complete, is_well_defined",
    [
        (True, False, False),  # Empty partial, incomplete plan
        (False, False, False),  # Non-empty partial, incomplete plan
        (False, True, False),  # Non-empty partial, complete plan
        (True, True, True),  # Empty partial, complete plan
    ],
)
def test__well_defined_plan__returns_expected(env, partial_empty, plan_complete, is_well_defined):
    if partial_empty:
        partial = {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}
    else:
        partial = {"SECTOR": 1, "TAG": 1, "TECH": 3, "AMOUNT": 1}

    if plan_complete:
        plan = [1] * env.n_techs  # All techs assigned amount 1
    else:
        plan = [0] * env.n_techs  # No techs assigned

    state = {"partial": partial, "plan": plan}
    assert env.well_defined_plan(state) == is_well_defined


def test__get_mask_invalid_actions_forward__source_state(env):
    """Test masking at source state - should allow all initial assignments but not EOS or LOCK."""
    env.reset()
    mask = env.get_mask_invalid_actions_forward()

    # EOS should be masked (invalid) at source
    assert mask[env.action2index(env.eos)] == True

    # All LOCK actions should be invalid at source
    for tech_idx in range(1, env.n_techs + 1):
        lock_action = (env.token2idx_choices["LOCK"], tech_idx)
        assert mask[env.action2index(lock_action)] == True

    # SECTOR, TAG, TECH, AMOUNT actions should be valid
    for sector_idx in range(1, env.n_sectors + 1):
        action = (env.token2idx_choices["SECTOR"], sector_idx)
        assert mask[env.action2index(action)] == False

    for tag_idx in range(1, env.n_tags + 1):
        action = (env.token2idx_choices["TAG"], tag_idx)
        assert mask[env.action2index(action)] == False

    for tech_idx in range(1, env.n_techs + 1):
        action = (env.token2idx_choices["TECH"], tech_idx)
        assert mask[env.action2index(action)] == False

    for amount_idx in range(1, env.n_amounts + 1):
        action = (env.token2idx_choices["AMOUNT"], amount_idx)
        assert mask[env.action2index(action)] == False


def test__get_mask_invalid_actions_forward__complete_plan_empty_partial(env):
    """Test masking when plan is complete and partial is empty - should only allow EOS."""
    state = {
        "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        "plan": [1] * env.n_techs,  # All techs assigned
    }
    mask = env.get_mask_invalid_actions_forward(state)

    # Only EOS should be valid
    assert mask[env.action2index(env.eos)] == False

    # All other actions should be invalid
    for i, m in enumerate(mask):
        if env.action_space[i] != env.eos:
            assert m == True


def test__get_uniform_terminating_states__returns_valid_states(env):
    """Test that uniform sampling produces valid terminating states."""
    states = env.get_uniform_terminating_states(10, seed=42)

    assert len(states) == 10
    for state in states:
        assert env.well_defined_plan(state)
        # Partial should be empty
        assert state["partial"]["SECTOR"] == 0
        assert state["partial"]["TAG"] == 0
        assert state["partial"]["TECH"] == 0
        assert state["partial"]["AMOUNT"] == 0
        # All plan entries should be assigned
        for amount in state["plan"]:
            assert amount > 0


def test__network_structure_consistency(env):
    """Test that the network structure mappings are consistent."""
    for sector, techs in env.network_structure["sector2tech"].items():
        assert sector in env.sectors
        for tech in techs:
            assert tech in env.techs

    for sector, tags in env.network_structure["sector2tag"].items():
        assert sector in env.sectors
        for tag in tags:
            assert tag in env.tags

    for tag, techs in env.network_structure["tag2tech"].items():
        assert tag in env.tags
        for tech in techs:
            assert tech in env.techs

    for tag, sectors in env.network_structure["tag2sector"].items():
        assert tag in env.tags
        for sector in sectors:
            assert sector in env.sectors


def test__eos_action_completes_trajectory(env):
    """Test a full trajectory from source to done."""
    env.reset()

    # Build first investment: power_NUCLEAR (tech 3) with HIGH amount
    env.step((3, 3))  # TECH=power_NUCLEAR
    env.step((1, 1))  # SECTOR=POWER (forced)
    env.step((2, 1))  # TAG=GREEN (forced)
    env.step((4, 1))  # AMOUNT=HIGH
    env.step((5, 3))  # LOCK tech 3

    # Partial should be reset
    assert env.state["partial"]["TECH"] == 0
    assert env.state["plan"][2] == 1  # Tech 3 (0-indexed as 2) has amount 1

    # Complete all other techs
    for tech_idx in range(1, env.n_techs + 1):
        if env.state["plan"][tech_idx - 1] == 0:  # Not yet assigned
            env.step((3, tech_idx))  # TECH
            # Get forced sector and tag
            tech_token = env.idx2token_techs[tech_idx]
            sector_token = env.network_structure["tech2sector"][tech_token]
            tag_token = env.network_structure["tech2tag"][tech_token]
            env.step((1, env.token2idx_sectors[sector_token]))  # SECTOR
            env.step((2, env.token2idx_tags[tag_token]))  # TAG
            env.step((4, 1))  # AMOUNT=HIGH
            env.step((5, tech_idx))  # LOCK

    # Now EOS should be valid
    assert not env.done
    mask = env.get_mask_invalid_actions_forward()
    assert mask[env.action2index(env.eos)] == False

    env.step(env.eos)
    assert env.done


def test__tech_unavailable_after_lock(env):
    """Test that a tech becomes unavailable after being locked."""
    env.reset()

    # Lock tech 3
    env.step((3, 3))  # TECH=power_NUCLEAR
    env.step((1, 1))  # SECTOR=POWER
    env.step((2, 1))  # TAG=GREEN
    env.step((4, 1))  # AMOUNT=HIGH
    env.step((5, 3))  # LOCK

    # Tech 3 should no longer be available
    mask = env.get_mask_invalid_actions_forward()
    tech_action = (env.token2idx_choices["TECH"], 3)
    assert mask[env.action2index(tech_action)] == True


class TestFullPlanCommon(common.BaseTestsDiscrete):
    """Common tests for FullPlan."""

    @pytest.fixture(autouse=True)
    def setup(self, env):
        self.env = env
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__get_parents__all_parents_are_reached_with_different_actions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__trajectories_are_reversible": 10,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 2,
            "test__sample_backwards_reaches_source": 2,
        }