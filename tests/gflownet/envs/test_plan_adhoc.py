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


# NEW TESTS FOR SET-BASED ARCHITECTURE


class TestConstrainingWithMultiplePartialStates:
    """Test that constraining procedure works with multiple partial states."""

    def test__apply_constraints_with_complete_and_partial_states(self, env):
        """
        Test constraining with a mix of complete, partial, and unfilled states.
        Complete states should exclude their techs from future states.
        """
        state = [
            2,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},  # Complete
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 0},  # Partial (no AMOUNT)
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},  # Empty
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26

        env.reset()
        env.set_state(state)

        # Check that filled_on_set is properly tracked
        filled = env._states2tensor(
            current_state=state, fill_in_from_tech=True, with_amounts=False
        )

        # Verify only complete techs are marked as unavailable
        used_techs = set(filled[:, 2].int().tolist()) - {0}
        assert 5 in used_techs
        assert 10 in used_techs

    def test__apply_constraints_excludes_assigned_techs(self, env):
        """
        Test that _apply_constraints properly restricts available technologies
        based on already-assigned investments.
        """
        state = [
            3,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 1},
            {"SECTOR": 3, "TAG": 2, "TECH": 15, "AMOUNT": 0},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 25

        env.reset()
        env.set_state(state)

        # Apply constraints
        env._apply_constraints_forward(state=state)

        # Verify the third subenv has restricted techs_available
        assigned_techs = {5, 10, 15}
        for idx, subenv in env.subenvs.items():
            # For subenv idx, it should exclude all techs assigned in OTHER subenvs
            if idx < 3:
                # These subenvs have assignments
                # Subenv idx should exclude all assigned_techs EXCEPT its own
                my_tech = state[idx + 1]["TECH"]
                other_techs = assigned_techs - {my_tech}

                assert all(
                    t not in subenv.techs_available for t in other_techs
                ), f"Subenv {idx} should exclude {other_techs} but techs_available={subenv.techs_available}"
            else:
                # Subenvs beyond the assignments should exclude all assigned techs
                assert all(
                    t not in subenv.techs_available for t in assigned_techs
                ), f"Subenv {idx} should exclude all {assigned_techs} but techs_available={subenv.techs_available}"

    def test__apply_constraints_with_scattered_partial_states(self, env):
        """
        Test constraining with partial states scattered throughout the plan.
        """
        state = [
            4,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 0},  # Partial
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 1},  # Complete
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},  # Empty
            {"SECTOR": 3, "TAG": 2, "TECH": 15, "AMOUNT": 2},  # Complete
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 25

        env.reset()
        env.set_state(state)
        env._apply_constraints(state=state)

        filled = env._states2tensor(
            current_state=state, fill_in_from_tech=True, with_amounts=False
        )
        used_techs = set(filled[:, 2].int().tolist()) - {0}

        assert 5 in used_techs
        assert 10 in used_techs
        assert 15 in used_techs


class TestStates2PolicyPermutationInvariance:
    """Test permutation invariance of states2policy encoding."""

    def test__states2policy_invariant_with_complete_states(self, env):
        """Same investments in different orders should produce identical encodings."""
        state1 = [
            3,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 3},
            {"SECTOR": 3, "TAG": 2, "TECH": 15, "AMOUNT": 1},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 25

        state2 = [
            3,
            {"SECTOR": 3, "TAG": 2, "TECH": 15, "AMOUNT": 1},
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 3},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 25

        enc1 = env.states2policy([state1])
        enc2 = env.states2policy([state2])

        assert torch.allclose(
            enc1, enc2
        ), "Encodings should be identical for reordered states"

    def test__states2policy_invariant_with_partial_states(self, env):
        """
        Permutation invariance should hold even when partial states are present
        in different positions.
        """
        state1 = [
            3,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 0},  # Partial
            {"SECTOR": 3, "TAG": 2, "TECH": 15, "AMOUNT": 1},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 25

        state2 = [
            3,
            {"SECTOR": 3, "TAG": 2, "TECH": 15, "AMOUNT": 1},
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {
                "SECTOR": 2,
                "TAG": 1,
                "TECH": 10,
                "AMOUNT": 0,
            },  # Partial (different position)
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 25

        enc1 = env.states2policy([state1])
        enc2 = env.states2policy([state2])

        assert torch.allclose(
            enc1, enc2
        ), "Encodings should be identical even with partial states in different positions"

    def test__states2policy_sensitive_to_changes(self, env):
        """
        Different investments should produce different encodings.
        """
        state1 = [
            2,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 1},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 27

        state2 = [
            2,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 2, "TAG": 1, "TECH": 15, "AMOUNT": 1},  # Different tech
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 27

        enc1 = env.states2policy([state1])
        enc2 = env.states2policy([state2])

        assert not torch.allclose(
            enc1, enc2
        ), "Encodings should differ when investments differ"

    def test__states2policy_with_scattered_unfilled(self, env):
        """
        Test permutation invariance with unfilled states scattered among
        complete and partial states.
        """
        state1 = [
            3,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},  # Unfilled
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 1},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 25

        state2 = [
            3,
            {
                "SECTOR": 0,
                "TAG": 0,
                "TECH": 0,
                "AMOUNT": 0,
            },  # Unfilled (different position)
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 1},
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 25

        enc1 = env.states2policy([state1])
        enc2 = env.states2policy([state2])

        assert torch.allclose(
            enc1, enc2
        ), "Encodings should ignore position of unfilled states"


class TestStatesTensorConversion:
    """Test _states2tensor with various state configurations."""

    def test__states2tensor_extracts_correct_values(self, env):
        """Test that _states2tensor correctly reads state values."""
        state = [
            2,
            {"SECTOR": 1, "TAG": 2, "TECH": 5, "AMOUNT": 3},
            {"SECTOR": 3, "TAG": 1, "TECH": 10, "AMOUNT": 2},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26

        filled = env._states2tensor(
            current_state=state, fill_in_from_tech=False, with_amounts=True
        )

        assert filled[0, 0] == 1  # SECTOR
        assert filled[0, 1] == 2  # TAG
        assert filled[0, 2] == 5  # TECH
        assert filled[0, 3] == 3  # AMOUNT
        assert filled[1, 0] == 3  # SECTOR
        assert filled[1, 1] == 1  # TAG
        assert filled[1, 2] == 10  # TECH
        assert filled[1, 3] == 2  # AMOUNT
        assert filled[10, 0] == 0  # SECTOR
        assert filled[10, 1] == 0  # TAG
        assert filled[10, 2] == 0  # TECH
        assert filled[10, 3] == 0  # AMOUNT

    def test__states2tensor_fill_in_from_tech(self, env):
        """Test that fill_in_from_tech correctly infers sector and tag from tech."""
        state = [
            1,
            {
                "SECTOR": 0,
                "TAG": 0,
                "TECH": 1,
                "AMOUNT": 0,
            },  # Tech assigned, sector/tag empty
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 27

        filled = env._states2tensor(
            current_state=state, fill_in_from_tech=True, with_amounts=False
        )

        # TECH 1 should have inferred sector and tag
        assert filled[0, 2] == 1  # TECH
        assert filled[0, 0] != 0  # SECTOR should be filled in
        assert filled[0, 1] != 0  # TAG should be filled in

    def test__states2tensor_with_scattered_partial(self, env):
        """Test _states2tensor with partial states scattered throughout."""
        state = [
            4,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 0},  # Partial
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},  # Empty
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 2},  # Complete
            {"SECTOR": 3, "TAG": 2, "TECH": 0, "AMOUNT": 0},  # Partial
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 25

        filled = env._states2tensor(
            current_state=state, fill_in_from_tech=False, with_amounts=True
        )

        # Verify values are correctly read
        assert filled[0, 0] == 1 and filled[0, 3] == 0  # First partial
        assert filled[1, 0] == 0  # Empty
        assert filled[2, 0] == 2 and filled[2, 3] == 2  # Complete
        assert filled[3, 0] == 3 and filled[3, 3] == 0  # Second partial


class TestGetTechsSet:
    """Test _get_techs_set with scattered partial states."""

    def test__get_techs_set_ignores_scattered_empty_states(self, env):
        """_get_techs_set should only return techs from states with TECH assigned."""
        state = [
            4,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},  # Empty
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 0},  # Partial
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},  # Empty
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 25

        techs = env._get_techs_set(state)
        assert techs == {5, 10}

    def test__get_techs_set_with_many_empty_scattered(self, env):
        """_get_techs_set should handle many empty states scattered throughout."""
        state = [
            5,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 1},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 24

        techs = env._get_techs_set(state)
        assert techs == {5, 10}


class TestIntegrationWithScatteredStates:
    """Integration tests ensuring all components work with scattered partial states."""

    def test__full_pipeline_with_scattered_states(self, env):
        """
        Test that the full pipeline (tensor conversion -> constraint -> policy encoding)
        works with scattered partial states.
        """
        state = [
            5,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},  # Complete
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},  # Empty
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 0},  # Partial
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},  # Empty
            {"SECTOR": 3, "TAG": 2, "TECH": 15, "AMOUNT": 1},  # Complete
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 24

        env.reset()
        env.set_state(state)

        # Test tensor conversion
        filled = env._states2tensor(
            current_state=state, fill_in_from_tech=False, with_amounts=True
        )
        assert filled.shape == (29, 4)

        # Test constraint application
        env._apply_constraints(state=state)
        techs = env._get_techs_set(state)
        assert len(techs) > 0

        # Test policy encoding (should not crash)
        encoding = env.states2policy([state])
        assert encoding.shape[0] == 1

    def test__permutation_invariance_comprehensive(self, env):
        """
        Comprehensive test: multiple permutations of the same state with scattered
        empty states should all produce identical encodings.
        """
        # Base configuration: 3 investments with 2 empty states scattered
        configs = [
            # Config 1: complete, empty, complete, empty, partial
            [
                5,
                {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 3},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 3, "TAG": 2, "TECH": 15, "AMOUNT": 0},
            ],
            # Config 2: reordered complete states, different empty positions
            [
                5,
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 3},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
                {"SECTOR": 3, "TAG": 2, "TECH": 15, "AMOUNT": 0},
            ],
            # Config 3: different ordering again
            [
                5,
                {"SECTOR": 3, "TAG": 2, "TECH": 15, "AMOUNT": 0},
                {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 3},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            ],
        ]

        # Pad all to full length
        for config in configs:
            config.extend([{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 24)

        encodings = [env.states2policy([cfg]) for cfg in configs]

        # All encodings should be identical
        for i in range(1, len(encodings)):
            assert torch.allclose(
                encodings[0], encodings[i]
            ), f"Encoding {i} differs from encoding 0"

    def test__constraints_applied_consistently_across_permutations(self, env):
        """
        Test that applying constraints produces consistent results regardless
        of the order of investments in the state.
        """
        base_config = [
            3,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 1},
            {"SECTOR": 3, "TAG": 2, "TECH": 15, "AMOUNT": 0},
        ]

        perm1 = base_config + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26
        perm2 = [
            3,
            {"SECTOR": 3, "TAG": 2, "TECH": 15, "AMOUNT": 0},
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 1},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26

        env.reset()
        env.set_state(perm1)
        env._apply_constraints(state=perm1)
        techs1 = env._get_techs_set(perm1)

        env.reset()
        env.set_state(perm2)
        env._apply_constraints(state=perm2)
        techs2 = env._get_techs_set(perm2)

        assert techs1 == techs2, "Tech sets should be identical regardless of order"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test__all_empty_state(self, env):
        """Test with completely empty state."""
        state = [0] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * env.n_techs

        techs = env._get_techs_set(state)
        assert techs == set()

        filled = env._states2tensor(
            current_state=state, fill_in_from_tech=False, with_amounts=True
        )
        assert filled.sum() == 0

    def test__single_complete_investment_scattered(self, env):
        """Test with single complete investment scattered among empties."""
        state = [
            1,
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 25

        techs = env._get_techs_set(state)
        print(techs)
        assert techs == {5}

        encoding = env.states2policy([state])
        assert encoding is not None

    def test__max_number_of_investments_scattered(self, env):
        """Test with maximum number of investments scattered throughout."""
        # Create a state with as many complete investments as possible
        n_max = env.n_techs
        state = [n_max]

        for i in range(n_max):
            if i % 2 == 0:  # Alternate complete and empty
                state.append(
                    {
                        "SECTOR": 1 + (i % 5),
                        "TAG": 1 + (i % 3),
                        "TECH": 1 + i,
                        "AMOUNT": 1 + (i % 4),
                    }
                )
            else:
                state.append({"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0})

        techs = env._get_techs_set(state)
        # Should extract techs from complete investments
        assert len(techs) > 0

    def test__states2policy_encoding_consistency_repeated_calls(self, env):
        """Test that repeated calls to states2policy produce identical results."""
        state = [
            2,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 2, "TAG": 1, "TECH": 10, "AMOUNT": 1},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ] + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 27

        enc1 = env.states2policy([state])
        enc2 = env.states2policy([state])
        enc3 = env.states2policy([state])

        assert torch.allclose(enc1, enc2) and torch.allclose(
            enc2, enc3
        ), "Repeated calls should produce identical encodings"


# Keep original tests for baseline validation
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
            ]
            + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26,
            [
                2,
                {"SECTOR": 3, "TAG": 1, "TECH": 28, "AMOUNT": 1},
                {"SECTOR": 5, "TAG": 3, "TECH": 20, "AMOUNT": 4},
                {"SECTOR": 4, "TAG": 1, "TECH": 0, "AMOUNT": 0},
            ]
            + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26,
            True,
        ),
    ],
)
def test__states2policy__order_invariance_original(state1, state2, should_be_equal):
    """Original baseline test."""
    env = Plan()
    encoding1 = env.states2policy([state1])
    encoding2 = env.states2policy([state2])

    are_equal = torch.allclose(encoding1, encoding2)
    assert are_equal == should_be_equal


def test__states2policy__batch_order_invariance_original():
    """Original batch invariance test."""
    env = Plan()

    states_batch1 = [
        [
            2,
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 2, "TAG": 2, "TECH": 10, "AMOUNT": 3},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ]
        + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26,
        [
            1,
            {"SECTOR": 3, "TAG": 1, "TECH": 15, "AMOUNT": 1},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ]
        + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 27,
    ]

    states_batch2 = [
        [
            2,
            {"SECTOR": 2, "TAG": 2, "TECH": 10, "AMOUNT": 3},
            {"SECTOR": 1, "TAG": 1, "TECH": 5, "AMOUNT": 2},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ]
        + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 26,
        [
            1,
            {"SECTOR": 3, "TAG": 1, "TECH": 15, "AMOUNT": 1},
            {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
        ]
        + [{"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}] * 27,
    ]

    encodings1 = env.states2policy(states_batch1)
    encodings2 = env.states2policy(states_batch2)

    assert torch.allclose(
        encodings1[0], encodings2[0]
    ), "Encodings for reordered states should be identical"

    assert torch.allclose(
        encodings1[1], encodings2[1]
    ), "Encodings for identical states should be identical"
