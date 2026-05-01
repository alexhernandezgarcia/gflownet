import common
import numpy as np
import pytest
import torch

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.choices import Choices
from gflownet.utils.common import tbool, tfloat


@pytest.fixture
def env_default():
    return Choices()


@pytest.fixture
def env_with_replacement():
    return Choices(
        options=["A", "B", "C", "D", "E"], max_selection=3, with_replacement=True
    )


@pytest.fixture
def env_without_replacement():
    return Choices(
        options=["A", "B", "C", "D", "E"], max_selection=3, with_replacement=False
    )


@pytest.fixture
def env_2of3_without_replacement():
    return Choices(n_options=3, max_selection=2, with_replacement=False)


@pytest.fixture
def env_2of3_with_replacement():
    return Choices(n_options=3, max_selection=2, with_replacement=True)


@pytest.fixture
def env_3of3_without_replacement():
    return Choices(n_options=3, max_selection=3, with_replacement=False)


@pytest.fixture
def env_3of3_with_replacement():
    return Choices(n_options=3, max_selection=3, with_replacement=True)


@pytest.mark.parametrize(
    "env",
    [
        "env_default",
        "env_with_replacement",
        "env_without_replacement",
    ],
)
def test__environment__initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


@pytest.mark.parametrize(
    "env, has_constraints_exp",
    [
        ("env_default", False),
        ("env_with_replacement", False),
        ("env_without_replacement", True),
    ],
)
def test__constraints_are_as_expected(env, has_constraints_exp, request):
    env = request.getfixturevalue(env)
    assert env.has_constraints == has_constraints_exp


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env",
    [
        "env_default",
        "env_with_replacement",
        "env_without_replacement",
    ],
)
def test__trajectory_random__does_not_crash_and_reaches_done(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    env.trajectory_random()
    assert env.done


@pytest.mark.repeat(100)
@pytest.mark.parametrize(
    "env",
    [
        "env_without_replacement",
    ],
)
def test__envs_without_replacement_do_not_repeat_elements(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    env.trajectory_random()
    assert len(env.get_options()) == env.max_selection


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env",
    [
        "env_default",
        "env_with_replacement",
        "env_without_replacement",
    ],
)
def test__trajectory_backwards_random__does_not_crash_and_reaches_source(env, request):
    env = request.getfixturevalue(env)
    env.reset()
    env.trajectory_random()
    assert env.done
    env.trajectory_random(backward=True)
    assert env.is_source()


@pytest.mark.parametrize(
    "env, state, mask_expected",
    [
        # Source state
        (
            "env_without_replacement",
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [0, 0, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [0],
                1: [0],
                2: [0],
            },
            # fmt: off
            [
                False, # Set actions
                False, True, # Mask (EOS invalid)
                False, False, False, False # Padding
            ],
            # fmt: on
        ),
        # Source -> active env
        (
            "env_without_replacement",
            {
                "_active": 0,
                "_toggle": 0,
                "_dones": [0, 0, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [0],
                1: [0],
                2: [0],
            },
            # fmt: off
            [
                True, # Active unique env 0
                False, False, False, False, False, True, # Mask of Choice: EOS invalid
            ],
            # fmt: on
        ),
        # Source -> active env -> choice selected
        (
            "env_without_replacement",
            {
                "_active": 0,
                "_toggle": 0,
                "_dones": [0, 0, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [2],
                1: [0],
                2: [0],
            },
            # fmt: off
            [
                True, # Active unique env 0
                True, True, True, True, True, False, # Mask of Choice: only EOS valid
            ],
            # fmt: on
        ),
        # Source -> active env -> choice selected and done
        (
            "env_without_replacement",
            {
                "_active": 0,
                "_toggle": 0,
                "_dones": [1, 0, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [2],
                1: [0],
                2: [0],
            },
            # fmt: off
            [
                False, # Set actions
                False, True, # Mask (EOS invalid)
                False, False, False, False # Padding
            ],
            # fmt: on
        ),
        # One subenv is done
        (
            "env_without_replacement",
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [1, 0, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [2],
                1: [0],
                2: [0],
            },
            # fmt: off
            [
                False, # Set actions
                False, True, # Mask (EOS invalid)
                False, False, False, False # Padding
            ],
            # fmt: on
        ),
        # One subenv is done -> active env
        (
            "env_without_replacement",
            {
                "_active": 1,
                "_toggle": 0,
                "_dones": [1, 0, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [2],
                1: [0],
                2: [0],
            },
            # fmt: off
            [
                True, # Active unique env 0
                # Mask of Choice: Choice already selected and EOS invalid
                False, True, False, False, False, True,
            ],
            # fmt: on
        ),
    ],
)
def test__get_mask_invalid_actions_forward__returns_expected(
    env, state, mask_expected, request
):
    env = request.getfixturevalue(env)
    mask = env.get_mask_invalid_actions_forward(state, done=False)
    assert mask == mask_expected
    env.set_state(state)
    mask = env.get_mask_invalid_actions_forward()
    assert mask == mask_expected


@pytest.mark.parametrize(
    "env, states, states_policy_exp",
    [
        (
            "env_default",
            [
                {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [0],
                    1: [0],
                },
                {
                    "_active": 0,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [0],
                    1: [0],
                },
                {
                    "_active": 0,
                    "_toggle": 0,
                    "_dones": [0, 0],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [2],
                    1: [0],
                },
                {
                    "_active": 0,
                    "_toggle": 0,
                    "_dones": [1, 0],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [2],
                    1: [0],
                },
                {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [1, 0],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [2],
                    1: [0],
                },
                {
                    "_active": 1,
                    "_toggle": 0,
                    "_dones": [1, 0],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [2],
                    1: [0],
                },
                {
                    "_active": 1,
                    "_toggle": 0,
                    "_dones": [1, 0],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [2],
                    1: [1],
                },
                {
                    "_active": 1,
                    "_toggle": 0,
                    "_dones": [1, 1],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [2],
                    1: [1],
                },
                {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [1, 1],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [2],
                    1: [1],
                },
            ],
            torch.stack(
                [
                    torch.tensor(
                        [
                            # fmt: off
                            -1.0, # FLAG
                            # OPTIONS
                            0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, # FLAG
                            # OPTIONS
                            0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, # FLAG
                            # OPTIONS
                            0.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            1.0, # FLAG
                            # OPTIONS
                            0.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            -1.0, # FLAG
                            # OPTIONS
                            0.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, # FLAG
                            # OPTIONS
                            0.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, # FLAG
                            # OPTIONS
                            1.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            1.0, # FLAG
                            # OPTIONS
                            1.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            -1.0, # FLAG
                            # OPTIONS
                            1.0, 1.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                ],
                dim=0,
            ),
        ),
        (
            "env_with_replacement",
            [
                {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [0, 0, 0],
                    "_envs_unique": [0, 0, 0],
                    "_keys": [0, 1, 2],
                    0: [0],
                    1: [0],
                    2: [0],
                },
                {
                    "_active": 0,
                    "_toggle": 0,
                    "_dones": [0, 0, 0],
                    "_envs_unique": [0, 0, 0],
                    "_keys": [0, 1, 2],
                    0: [0],
                    1: [0],
                    2: [0],
                },
                {
                    "_active": 0,
                    "_toggle": 0,
                    "_dones": [0, 0, 0],
                    "_envs_unique": [0, 0, 0],
                    "_keys": [0, 1, 2],
                    0: [2],
                    1: [0],
                    2: [0],
                },
                {
                    "_active": 0,
                    "_toggle": 0,
                    "_dones": [1, 0, 0],
                    "_envs_unique": [0, 0, 0],
                    "_keys": [0, 1, 2],
                    0: [2],
                    1: [0],
                    2: [0],
                },
                {
                    "_active": -1,
                    "_toggle": 0,
                    "_dones": [1, 1, 1],
                    "_envs_unique": [0, 0, 0],
                    "_keys": [0, 1, 2],
                    0: [2],
                    1: [2],
                    2: [5],
                },
                {
                    "_active": 2,
                    "_toggle": 0,
                    "_dones": [1, 1, 0],
                    "_envs_unique": [0, 0, 0],
                    "_keys": [0, 1, 2],
                    0: [2],
                    1: [2],
                    2: [0],
                },
            ],
            torch.stack(
                [
                    torch.tensor(
                        [
                            # fmt: off
                            -1.0, # FLAG
                            # OPTIONS
                            0.0, 0.0, 0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, # FLAG
                            # OPTIONS
                            0.0, 0.0, 0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, # FLAG
                            # OPTIONS
                            0.0, 1.0, 0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            1.0, # FLAG
                            # OPTIONS
                            0.0, 1.0, 0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            -1.0, # FLAG
                            # OPTIONS
                            0.0, 2.0, 0.0, 0.0, 1.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                    torch.tensor(
                        [
                            # fmt: off
                            0.0, # FLAG
                            # OPTIONS
                            0.0, 2.0, 0.0, 0.0, 0.0,
                            # fmt: on
                        ],
                        dtype=torch.float,
                    ),
                ],
                dim=0,
            ),
        ),
    ],
)
def test__states2policy__returns_expected(env, states, states_policy_exp, request):
    env = request.getfixturevalue(env)
    assert torch.equal(states_policy_exp, env.states2policy(states))


@pytest.mark.parametrize(
    "env, state, action",
    [
        (
            "env_without_replacement",
            {
                "_active": 0,
                "_toggle": 0,
                "_dones": [0, 0, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [0],
                1: [0],
                2: [0],
            },
            (0, 3),
        ),
    ],
)
def test__get_logprobs_forward__is_finite_state_explicit(env, state, action, request):
    env = request.getfixturevalue(env)
    masks = torch.unsqueeze(
        tbool(
            env.get_mask_invalid_actions_forward(state, done=False), device=env.device
        ),
        0,
    )
    policy_outputs = torch.unsqueeze(env.random_policy_output, 0)
    actions_torch = torch.unsqueeze(torch.tensor(action), 0)
    logprobs = env.get_logprobs(
        policy_outputs=policy_outputs,
        actions=actions_torch,
        mask=masks,
        states_from=[state],
        is_backward=False,
    )
    assert torch.all(torch.isfinite(logprobs))


@pytest.mark.parametrize(
    "env, state, action",
    [
        (
            "env_without_replacement",
            {
                "_active": 0,
                "_toggle": 0,
                "_dones": [0, 0, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [0],
                1: [0],
                2: [0],
            },
            (0, 3),
        ),
    ],
)
def test__get_logprobs_forward__is_finite_state_implicit(env, state, action, request):
    env = request.getfixturevalue(env)
    env.set_state(state, done=False)
    masks = torch.unsqueeze(
        tbool(env.get_mask_invalid_actions_forward(), device=env.device),
        0,
    )
    policy_outputs = torch.unsqueeze(env.random_policy_output, 0)
    actions_torch = torch.unsqueeze(torch.tensor(action), 0)
    logprobs = env.get_logprobs(
        policy_outputs=policy_outputs,
        actions=actions_torch,
        mask=masks,
        states_from=[state],
        is_backward=False,
    )
    assert torch.all(torch.isfinite(logprobs))


@pytest.mark.parametrize(
    "env, state, action",
    [
        (
            "env_without_replacement",
            {
                "_active": 0,
                "_toggle": 0,
                "_dones": [0, 0, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [3],
                1: [0],
                2: [0],
            },
            (0, 3),
        ),
    ],
)
def test__get_logprobs_backward__is_finite_state_explicit(env, state, action, request):
    env = request.getfixturevalue(env)
    masks = torch.unsqueeze(
        tbool(
            env.get_mask_invalid_actions_backward(state, done=False), device=env.device
        ),
        0,
    )
    policy_outputs = torch.unsqueeze(env.random_policy_output, 0)
    actions_torch = torch.unsqueeze(torch.tensor(action), 0)
    logprobs = env.get_logprobs(
        policy_outputs=policy_outputs,
        actions=actions_torch,
        mask=masks,
        states_from=[state],
        is_backward=True,
    )
    assert torch.all(torch.isfinite(logprobs))


@pytest.mark.parametrize(
    "env, state, action",
    [
        (
            "env_without_replacement",
            {
                "_active": 0,
                "_toggle": 0,
                "_dones": [0, 0, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [3],
                1: [0],
                2: [0],
            },
            (0, 3),
        ),
    ],
)
def test__get_logprobs_backward__is_finite_state_implicit(env, state, action, request):
    env = request.getfixturevalue(env)
    env.set_state(state, done=False)
    masks = torch.unsqueeze(
        tbool(env.get_mask_invalid_actions_backward(), device=env.device),
        0,
    )
    policy_outputs = torch.unsqueeze(env.random_policy_output, 0)
    actions_torch = torch.unsqueeze(torch.tensor(action), 0)
    logprobs = env.get_logprobs(
        policy_outputs=policy_outputs,
        actions=actions_torch,
        mask=masks,
        states_from=[state],
        is_backward=True,
    )
    assert torch.all(torch.isfinite(logprobs))


@pytest.mark.parametrize(
    "env, state_a, options_avail_a, state_b, options_avail_b",
    [
        (
            "env_without_replacement",
            {
                "_active": 1,
                "_toggle": 0,
                "_dones": [1, 1, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [3],
                1: [4],
                2: [0],
            },
            {1, 2, 5},
            {
                "_active": 0,
                "_toggle": 0,
                "_dones": [0, 0, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [0],
                1: [0],
                2: [0],
            },
            {1, 2, 3, 4, 5},
        ),
    ],
)
def test__set_state__sets_expected_constraints(
    env, state_a, options_avail_a, state_b, options_avail_b, request
):
    env = request.getfixturevalue(env)
    assert env.choice_env.options_available == {1, 2, 3, 4, 5}
    env.set_state(state_a, done=False)
    assert env.choice_env.options_available == options_avail_a
    env.set_state(state_b, done=False)
    assert env.choice_env.options_available == options_avail_b


@pytest.mark.parametrize(
    "env, state, all_parents_perms, parent_actions_perms",
    [
        # All done
        (
            "env_2of3_with_replacement",
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [1, 1],
                "_envs_unique": [0, 0],
                "_keys": [0, 1],
                0: [1],
                1: [2],
            },
            [
                {
                    "_active": 1,
                    "_toggle": 0,
                    "_dones": [1, 1],
                    "_envs_unique": [0, 0],
                    "_keys": [0, 1],
                    0: [1],
                    1: [2],
                },
                {
                    "_active": 1,
                    "_toggle": 0,
                    "_dones": [1, 1],
                    "_envs_unique": [0, 0],
                    "_keys": [1, 0],
                    0: [1],
                    1: [2],
                },
            ],
            [(-1, 0), (-1, 0)],
        ),
        (
            "env_3of3_with_replacement",
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [1, 1, 1],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [1],
                1: [2],
                2: [3],
            },
            [
                {
                    "_active": 2,
                    "_toggle": 0,
                    "_dones": [1, 1, 1],
                    "_envs_unique": [0, 0, 0],
                    "_keys": [0, 1, 2],
                    0: [1],
                    1: [2],
                    2: [3],
                },
                {
                    "_active": 2,
                    "_toggle": 0,
                    "_dones": [1, 1, 1],
                    "_envs_unique": [0, 0, 0],
                    "_keys": [0, 2, 1],
                    0: [1],
                    1: [2],
                    2: [3],
                },
                {
                    "_active": 2,
                    "_toggle": 0,
                    "_dones": [1, 1, 1],
                    "_envs_unique": [0, 0, 0],
                    "_keys": [1, 0, 2],
                    0: [1],
                    1: [2],
                    2: [3],
                },
                {
                    "_active": 2,
                    "_toggle": 0,
                    "_dones": [1, 1, 1],
                    "_envs_unique": [0, 0, 0],
                    "_keys": [1, 2, 0],
                    0: [1],
                    1: [2],
                    2: [3],
                },
                {
                    "_active": 2,
                    "_toggle": 0,
                    "_dones": [1, 1, 1],
                    "_envs_unique": [0, 0, 0],
                    "_keys": [2, 0, 1],
                    0: [1],
                    1: [2],
                    2: [3],
                },
                {
                    "_active": 2,
                    "_toggle": 0,
                    "_dones": [1, 1, 1],
                    "_envs_unique": [0, 0, 0],
                    "_keys": [2, 1, 0],
                    0: [1],
                    1: [2],
                    2: [3],
                },
            ],
            [(-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0)],
        ),
    ],
)
def test__get_parents__with_permutations_can_return_all_possible_parents(
    env, state, all_parents_perms, parent_actions_perms, request
):
    env = request.getfixturevalue(env)

    parents_found = [False] * len(all_parents_perms)

    # Keep obtaining parents until all possible expected parents are found at least
    # once
    count = 0
    max_iters = 1e6
    while not all(parents_found):
        parents, parent_actions = env.get_parents(state, done=False)
        for p, p_a in zip(parents, parent_actions):
            for idx, (p_exp, p_a_exp) in enumerate(
                zip(all_parents_perms, parent_actions_perms)
            ):
                # If a match is found between parents, the action must match too. Add
                # the index to the found parents and break the inner loop to check the
                # next actual parent
                if GFlowNetEnv.equal(p, p_exp):
                    assert p_a == p_a_exp
                    parents_found[idx] = True
                    break
            else:
                # If a parent is not among the expected parents, the test is failed
                assert False
        count += 1
        if count > max_iters:
            assert False
    assert True


@pytest.mark.parametrize(
    "env, state, action, logprob",
    [
        # All done
        # There are 2 permutations
        (
            "env_2of3_with_replacement",
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [1, 1],
                "_envs_unique": [0, 0],
                "_keys": [0, 1],
                0: [1],
                1: [2],
            },
            (-1, 0),
            np.log(1.0 / 2),
        ),
        # All done
        # There are 6 permutations
        (
            "env_3of3_with_replacement",
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [1, 1, 1],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [1],
                1: [2],
                2: [3],
            },
            (-1, 0),
            np.log(1.0 / 6),
        ),
        # All done
        # There are 3 permutations because two choices are the same
        (
            "env_3of3_with_replacement",
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [1, 1, 1],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [1],
                1: [2],
                2: [2],
            },
            (-1, 0),
            np.log(1.0 / 3),
        ),
        # All done
        # There is 1 permutations because all choices are the same
        (
            "env_3of3_with_replacement",
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [1, 1, 1],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [1],
                1: [1],
                2: [1],
            },
            (-1, 0),
            np.log(1.0 / 1),
        ),
        # All done but one
        # There are 2 permutations
        (
            "env_3of3_with_replacement",
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [1, 1, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [1],
                1: [2],
                2: [0],
            },
            (-1, 0),
            np.log(1.0 / 2),
        ),
        # All done but one
        # There are 2 permutations
        (
            "env_3of3_with_replacement",
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [1, 1, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [1],
                1: [2],
                2: [0],
            },
            (-1, 0),
            np.log(1.0 / 2),
        ),
        # All done but one
        # There is 1 permutations because both choices are the same
        (
            "env_3of3_with_replacement",
            {
                "_active": -1,
                "_toggle": 0,
                "_dones": [1, 1, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [2],
                1: [2],
                2: [0],
            },
            (-1, 0),
            np.log(1.0 / 1),
        ),
    ],
)
def test__get_logprobs_backward__calculates_permutations(
    env, state, action, logprob, request
):
    env = request.getfixturevalue(env)
    masks = torch.unsqueeze(
        tbool(
            env.get_mask_invalid_actions_backward(state, done=False), device=env.device
        ),
        0,
    )
    policy_outputs = torch.unsqueeze(env.random_policy_output, 0)
    actions_torch = torch.unsqueeze(torch.tensor(action), 0)
    logprobs = env.get_logprobs(
        policy_outputs=policy_outputs,
        actions=actions_torch,
        mask=masks,
        states_from=[state],
        is_backward=True,
    )
    assert torch.isclose(logprobs[0], torch.tensor(logprob).to(logprobs))


@pytest.mark.parametrize(
    "env, state, action",
    [
        # Two sub-envs
        # One done, one at source and active
        (
            "env_2of3_with_replacement",
            {
                "_active": 1,
                "_toggle": 0,
                "_dones": [1, 0],
                "_envs_unique": [0, 0],
                "_keys": [0, 1],
                0: [1],
                1: [0],
            },
            (-1, 0),
        ),
        # Three sub-envs
        # Two done, one at source and active
        (
            "env_3of3_with_replacement",
            {
                "_active": 2,
                "_toggle": 0,
                "_dones": [1, 1, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [1],
                1: [2],
                2: [0],
            },
            (-1, 0),
        ),
        # Three sub-envs
        # Two done, one at source and active
        (
            "env_3of3_with_replacement",
            {
                "_active": 2,
                "_toggle": 0,
                "_dones": [1, 1, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [2],
                1: [2],
                2: [0],
            },
            (-1, 0),
        ),
        # Three sub-envs
        # One done, one at source and active, one at source
        (
            "env_3of3_with_replacement",
            {
                "_active": 1,
                "_toggle": 0,
                "_dones": [1, 0, 0],
                "_envs_unique": [0, 0, 0],
                "_keys": [0, 1, 2],
                0: [1],
                1: [0],
                2: [0],
            },
            (-1, 0),
        ),
    ],
)
def test__get_logprobs_backward__returns_zero_if_action_is_deactivate(
    env, state, action, request
):
    env = request.getfixturevalue(env)
    masks = torch.unsqueeze(
        tbool(
            env.get_mask_invalid_actions_backward(state, done=False), device=env.device
        ),
        0,
    )
    policy_outputs = torch.unsqueeze(env.random_policy_output, 0)
    actions_torch = torch.unsqueeze(torch.tensor(action), 0)
    logprobs = env.get_logprobs(
        policy_outputs=policy_outputs,
        actions=actions_torch,
        mask=masks,
        states_from=[state],
        is_backward=True,
    )
    if not torch.isclose(logprobs[0], torch.tensor(0.0).to(logprobs)):
        import ipdb

        ipdb.set_trace()
    assert torch.isclose(logprobs[0], torch.tensor(0.0).to(logprobs))


class TestChoicesDefault(common.BaseTestsDiscrete):
    """Common tests for default Choices environment"""

    @pytest.fixture(autouse=True)
    def setup(self, env_default):
        self.env = env_default
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 0,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__state2readable__is_reversible": 10,
        }
        self.n_states = {
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }


class TestChoicesWithReplacement(common.BaseTestsDiscrete):
    """Common tests for Choices with replacement environment"""

    @pytest.fixture(autouse=True)
    def setup(self, env_with_replacement):
        self.env = env_with_replacement
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 0,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__state2readable__is_reversible": 10,
        }
        self.n_states = {
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }


class TestChoicesWithoutReplacement(common.BaseTestsDiscrete):
    """Common tests for Choices with replacement environment"""

    @pytest.fixture(autouse=True)
    def setup(self, env_without_replacement):
        self.env = env_without_replacement
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 0,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__state2readable__is_reversible": 10,
        }
        self.n_states = {
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
        }
