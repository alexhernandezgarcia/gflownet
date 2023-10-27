import pytest
import torch
import numpy as np

from gflownet.envs.amp import AMP


@pytest.fixture
def env():
    return AMP(proxy_state_format="state")


def test__environment__initializes_properly():
    env = AMP(proxy_state_format="state")
    assert torch.eq(
        env.source, torch.ones(env.max_seq_length, dtype=torch.int64) * env.padding_idx
    ).all()
    assert torch.eq(
        env.state, torch.ones(env.max_seq_length, dtype=torch.int64) * env.padding_idx
    ).all()


def test__environment__action_space_has_eos():
    env = AMP(proxy_state_format="state")
    assert (env.eos,) in env.action_space


@pytest.mark.parametrize(
    "state, expected_state_policy",
    [
        (
            torch.tensor([[3, 2, 21, 21, 21]]),
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ),
        (
            torch.tensor([[3, 2, 4, 2, 0]]),
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ),
        (
            torch.tensor([[21, 21, 21, 21, 21]]),
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ),
    ],
)
def test_environment_policy_transformation(state, expected_state_policy):
    env = AMP(proxy_state_format="state", max_seq_length=5)
    expected_state_policy_tensor = torch.tensor(
        expected_state_policy, dtype=env.float, device=env.device
    ).reshape(state.shape[0], -1)
    state_policy = env.statetorch2policy(state)
    assert torch.eq(state_policy, expected_state_policy_tensor).all()


@pytest.mark.parametrize(
    "state, done, expected_parent, expected_parent_action",
    [
        (
            torch.tensor([3, 21, 21, 21, 21]),
            False,
            [torch.tensor([21, 21, 21, 21, 21])],
            [(3,)],
        ),
        (
            torch.tensor([3, 2, 4, 2, 0]),
            False,
            [torch.tensor([3, 2, 4, 2, 21])],
            [(0,)],
        ),
        (
            torch.tensor([3, 21, 21, 21, 21]),
            False,
            [torch.tensor([21, 21, 21, 21, 21])],
            [(3,)],
        ),
        (
            torch.tensor([3, 21, 21, 21, 21]),
            True,
            [torch.tensor([3, 21, 21, 21, 21])],
            [(20,)],
        ),
        (
            torch.tensor([21, 21, 21, 21, 21]),
            False,
            [],
            [],
        ),
    ],
)
def test_environment_get_parents(state, done, expected_parent, expected_parent_action):
    env = AMP(proxy_state_format="state", max_seq_length=5)
    parent, parent_action = env.get_parents(state, done)
    print(parent, parent_action)
    if parent != []:
        parent_tensor = torch.vstack(parent).to(env.device).to(env.float)
        expected_parent_tensor = (
            torch.vstack(expected_parent).to(env.device).to(env.float)
        )
        assert torch.eq(parent_tensor, expected_parent_tensor).all()
    else:
        assert parent == expected_parent
    assert parent_action == expected_parent_action


@pytest.mark.parametrize(
    "state, action, expected_next_state, expected_executed_action, expected_valid",
    [
        (
            torch.tensor([3, 21, 21, 21, 21]),
            (2,),
            torch.tensor([3, 2, 21, 21, 21]),
            (2,),
            True,
        ),
        (
            torch.tensor([3, 2, 4, 2, 0]),
            (2,),
            torch.tensor([3, 2, 4, 2, 0]),
            (20,),
            True,
        ),
        (
            torch.tensor([21, 21, 21, 21, 21]),
            (20,),
            torch.tensor([21, 21, 21, 21, 21]),
            (20,),
            False,
        ),
        (
            torch.tensor([3, 21, 21, 21, 21]),
            (20,),
            torch.tensor([3, 21, 21, 21, 21]),
            (20,),
            True,
        ),
    ],
)
def test_environment_step(
    state, action, expected_next_state, expected_executed_action, expected_valid
):
    env = AMP(proxy_state_format="state", max_seq_length=5)
    env.state = state
    n_actions = env.n_actions
    next_state, action_executed, valid = env.step(action)
    if expected_executed_action == (20,) and expected_valid == True:
        assert env.done == True
    if expected_valid == True:
        assert env.n_actions == n_actions + 1
    assert torch.eq(next_state, expected_next_state).all()
    assert action_executed == expected_executed_action
    assert valid == expected_valid
