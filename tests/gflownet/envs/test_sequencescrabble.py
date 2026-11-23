import pytest
from gflownet.envs.sequence_scrabble import SequenceScrabble


@pytest.fixture
def env():
    """Create an independent environment for each test."""
    return SequenceScrabble()


def generate_fad(env):
    """Generate FAD. Environment type indices: D=0, A=1, F=2."""
    actions = [
        (-1, 1, 0),  # Insert A
        (1, 1, 0),   # Add letter A
        (1, -1, 0),  # End A
        (-1, 1, 0),  # Deactivate A

        (-1, 0, 1),  # Insert D on the right
        (0, 1, 0),   # Add letter D
        (0, -1, 0),  # End D
        (-1, 0, 1),  # Deactivate D

        (-1, 2, 0),  # Insert F on the left
        (2, 1, 0),   # Add letter F
        (2, -1, 0),  # End F
        (-1, 2, 0),  # Deactivate F

        env.eos,     # End the entire sequence
    ]

    for action in actions:
        assert action in env.get_valid_actions(), (
            f"Action {action} is invalid in state {env.state}"
        )
        _, _, valid = env.step(action)
        assert valid, f"Failed to execute action {action}"


def test_initial_state(env):
    """The environment should start with an empty sequence."""
    assert env.is_source()
    assert not env.done
    assert env.subenvs == []
    assert env.state["_indices"] == []
    assert env.n_unique_envs == 3


def test_forward_generates_fad(env):
    """Forward transitions should generate FAD with consistent metadata."""
    generate_fad(env)

    assert env.done
    assert env.state2readable() == "FAD"
    assert env.state["_active"] == 0
    assert env.state["_dones"] == [1, 1, 1]

    # Creation order: A, D, F. Spatial order: F, A, D.
    assert env.state["_envs_unique"] == [1, 0, 2]
    assert env.state["_indices"] == [2, 0, 1]

    print(f"Generate sequence: {env.state2readable()}")


def test_backward_returns_to_source(env):
    """Backward transitions should return the environment to its source."""
    generate_fad(env)

    for _ in range(30):
        if not env.done and env.is_source():
            break

        actions = env.get_valid_actions(backward=True)
        assert actions, f"No valid backward actions in state {env.state}"

        _, _, valid = env.step_backwards(actions[0])
        assert valid, f"Failed to execute backward action {actions[0]}"

    assert not env.done
    assert env.is_source()
    assert env.subenvs == []
    assert env.state["_indices"] == []
    assert env.state["_envs_unique"] == []
    assert env.state["_dones"] == []

