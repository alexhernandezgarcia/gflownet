from copy import copy
from itertools import chain

import common
import numpy as np
import pytest
import torch
from scipy import special
from torch.distributions import Uniform

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.utils.common import tbool, tfloat


@pytest.fixture
def env():
    return ContinuousTorus(n_dim=2, length_traj=3)


@pytest.fixture
def env_su():
    return ContinuousTorus(n_dim=2, length_traj=3, start_uniform=True)


@pytest.fixture
def env_diff():
    return ContinuousTorus(n_dim=2, length_traj=3, distr_type="diffusion", n_copm=1)


@pytest.mark.parametrize(
    "action_space",
    [
        [
            (0.0, 0.0),
            (np.inf, np.inf),
        ],
    ],
)
def test__get_action_space__returns_expected(env, action_space):
    assert set(action_space) == set(env.action_space)


@pytest.mark.parametrize(
    "env_local, state, done, is_backward, action_expected",
    [
        ("env", [0.0, 0.0, 3.0], False, False, (np.inf, np.inf)),
        ("env", [0.0, 0.0, 3.0], True, False, (np.inf, np.inf)),
        ("env", [0.0, 0.0, 3.0], True, True, (np.inf, np.inf)),
        ("env", [1.37, 2.49, 3.0], False, False, (np.inf, np.inf)),
        ("env", [1.37, 2.49, 3.0], True, False, (np.inf, np.inf)),
        ("env", [1.37, 2.49, 3.0], True, True, (np.inf, np.inf)),
        ("env", [0.0, 0.0, 1.0], False, True, (0.0, 0.0)),
        ("env", [1.37, 2.49, 1.0], False, True, (1.37, 2.49)),
        ("env_su", [0.0, 0.0, 3.0], False, False, (np.inf, np.inf)),
        ("env_su", [0.0, 0.0, 3.0], True, False, (np.inf, np.inf)),
        ("env_su", [0.0, 0.0, 3.0], True, True, (np.inf, np.inf)),
        ("env_su", [1.37, 2.49, 3.0], False, False, (np.inf, np.inf)),
        ("env_su", [1.37, 2.49, 3.0], True, False, (np.inf, np.inf)),
        ("env_su", [1.37, 2.49, 3.0], True, True, (np.inf, np.inf)),
        ("env_su", [0.0, 0.0, 1.0], False, True, (0.0, 0.0)),
        ("env_su", [1.37, 2.49, 1.0], False, True, (1.37, 2.49)),
        ("env_diff", [0.0, 0.0, 3.0], False, False, (np.inf, np.inf)),
        ("env_diff", [0.0, 0.0, 3.0], True, False, (np.inf, np.inf)),
        ("env_diff", [0.0, 0.0, 3.0], True, True, (np.inf, np.inf)),
        ("env_diff", [1.37, 2.49, 3.0], False, False, (np.inf, np.inf)),
        ("env_diff", [1.37, 2.49, 3.0], True, False, (np.inf, np.inf)),
        ("env_diff", [1.37, 2.49, 3.0], True, True, (np.inf, np.inf)),
        ("env_diff", [0.0, 0.0, 1.0], False, True, (0.0, 0.0)),
        ("env_diff", [1.37, 2.49, 1.0], False, True, (1.37, 2.49)),
    ],
)
def test__sample_actions_batch__special_cases(
    env_local, state, done, is_backward, action_expected, request
):
    """
    Test a few of all (known...) special cases, both forward and backward.
    """
    env = request.getfixturevalue(env_local)
    env.set_state(state, done=done)
    if is_backward:
        mask = torch.unsqueeze(
            tbool(env.get_mask_invalid_actions_backward(), device=env.device), 0
        )
    else:
        mask = torch.unsqueeze(
            tbool(env.get_mask_invalid_actions_forward(), device=env.device), 0
        )
    random_policy = torch.unsqueeze(env.random_policy_output, 0)
    action_sampled = env.sample_actions_batch(
        random_policy,
        mask,
        [state],
        is_backward,
    )[0]
    assert all(np.isclose(action_sampled, action_expected))


@pytest.mark.repeat(10)
@pytest.mark.parametrize(
    "env_local, state, done, is_backward, action_special",
    [
        ("env", [0.0, 0.0, 2.0], False, False, (np.inf, np.inf)),
        ("env", [0.0, 0.0, 3.0], False, True, (np.inf, np.inf)),
        ("env", [1.37, 2.49, 2.0], False, False, (np.inf, np.inf)),
        ("env", [1.37, 2.49, 2.0], False, True, (np.inf, np.inf)),
        ("env", [0.0, 0.0, 2.0], False, True, (0.0, 0.0)),
        ("env", [1.37, 2.49, 2.0], False, True, (1.37, 2.49)),
        ("env", [1.37, 2.49, 1.0], False, False, (1.37, 2.49)),
        ("env_su", [0.0, 0.0, 2.0], False, False, (np.inf, np.inf)),
        ("env_su", [0.0, 0.0, 3.0], False, True, (np.inf, np.inf)),
        ("env_su", [1.37, 2.49, 2.0], False, False, (np.inf, np.inf)),
        ("env_su", [1.37, 2.49, 2.0], False, True, (np.inf, np.inf)),
        ("env_su", [0.0, 0.0, 2.0], False, True, (0.0, 0.0)),
        ("env_su", [1.37, 2.49, 2.0], False, True, (1.37, 2.49)),
        ("env_su", [1.37, 2.49, 1.0], False, False, (1.37, 2.49)),
        ("env_diff", [0.0, 0.0, 2.0], False, False, (np.inf, np.inf)),
        ("env_diff", [0.0, 0.0, 3.0], False, True, (np.inf, np.inf)),
        ("env_diff", [1.37, 2.49, 2.0], False, False, (np.inf, np.inf)),
        ("env_diff", [1.37, 2.49, 2.0], False, True, (np.inf, np.inf)),
        ("env_diff", [0.0, 0.0, 2.0], False, True, (0.0, 0.0)),
        ("env_diff", [1.37, 2.49, 2.0], False, True, (1.37, 2.49)),
        ("env_diff", [1.37, 2.49, 1.0], False, False, (1.37, 2.49)),
    ],
)
def test__sample_actions_batch__not_special_cases(
    env_local, state, done, is_backward, action_special, request
):
    """
    Test a few seemingly special cases, both forward and backward, and check that the
    special action is not sampled. Some of the tests may fail once in a blue moon if at
    all.
    """
    env = request.getfixturevalue(env_local)
    env.set_state(state, done=done)
    if is_backward:
        mask = torch.unsqueeze(
            tbool(env.get_mask_invalid_actions_backward(), device=env.device), 0
        )
    else:
        mask = torch.unsqueeze(
            tbool(env.get_mask_invalid_actions_forward(), device=env.device), 0
        )
    random_policy = torch.unsqueeze(env.random_policy_output, 0)
    action_sampled = env.sample_actions_batch(
        random_policy,
        mask,
        [state],
        is_backward,
    )[0]
    assert action_sampled != action_special


class TestContinuousTorusBasic(common.BaseTestsContinuous):
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
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }
        self.n_states = {}  # TODO: Populate.


# copypasted from above but wuth env_diff instead of env
class TestContinuousTorusDiffusion(common.BaseTestsContinuous):
    @pytest.fixture(autouse=True)
    def setup(self, env_diff):
        self.env = env_diff
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }
        self.n_states = {}  # TODO: Populate.


def test__sample_start_uniform(env_su):
    # test that first step is sampled from uniform and others are from policy
    # test that first logprob is uniform
    states = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    masks = []
    for state in states:
        env_su.set_state(state, done=False)
        mask = env_su.get_mask_invalid_actions_forward()
        masks.append(mask)
    masks = torch.tensor(masks, device=env_su.device)
    distr_params = {
        "vonmises_mean": 0.0,
        "vonmises_concentration": 8.0,
    }
    policy_output = torch.stack([env_su.get_policy_output(distr_params)] * len(states))
    unilogprob = -env_su.n_dim * np.log(2 * np.pi)
    list_actions_first = []
    list_actions_others = []
    for _ in range(300):
        actions = env_su.sample_actions_batch(
            policy_output,
            masks,
            states,
            is_backward=False,
        )
        logprobs = env_su.get_logprobs(
            policy_output, actions, masks, states, is_backward=False
        )
        assert all(np.isclose(logprobs[2:].detach().cpu().numpy(), unilogprob))
        assert not all(np.isclose(logprobs[:2].detach().cpu().numpy(), unilogprob))
        list_actions_first.extend(list(chain(*actions[2:])))
        list_actions_others.extend(list(chain(*actions[:2])))

    assert np.isclose(np.mean(list_actions_first), np.pi, atol=0.1)
    assert np.isclose(np.std(list_actions_first) ** 2, np.pi**2 / 3, atol=0.5)
    assert np.isclose(np.mean(list_actions_others), 0.0, atol=0.1)
    conc = distr_params["vonmises_concentration"]
    vm_var = 1 - special.i1(conc) / special.i0(conc)
    assert np.isclose(np.std(list_actions_others) ** 2, vm_var, atol=0.1)


def test__get_logprobs_start_uniform(env_su):
    states = [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    n_from_source = 3
    from_source = torch.tensor([False, False, True, True, True])
    masks = []
    for state in states:
        env_su.set_state(state, done=False)
        mask = env_su.get_mask_invalid_actions_forward()
        masks.append(mask)
    masks = torch.tensor(masks, device=env_su.device)
    distr_params = {
        "vonmises_mean": 0.0,
        "vonmises_concentration": 8.0,
    }
    policy_output = torch.stack([env_su.get_policy_output(distr_params)] * len(states))
    actions = env_su.sample_actions_batch(
        policy_output,
        masks,
        states,
        is_backward=False,
    )
    actions = torch.tensor(actions)
    logprobs = env_su.get_logprobs(
        policy_output, actions, masks, states, is_backward=False
    )
    distr_fs_angles = Uniform(
        torch.zeros(n_from_source, env_su.n_dim).to(actions),
        2 * torch.pi * torch.ones(n_from_source, env_su.n_dim).to(actions),
    )
    expected_logprobs = distr_fs_angles.log_prob(actions[from_source]).sum(axis=1)
    assert torch.allclose(logprobs[from_source], expected_logprobs)
    assert logprobs[0] != logprobs[-1]
    assert logprobs[1] != logprobs[-1]


@pytest.mark.parametrize(
    "env_local",
    [
        "env",
        "env_su",
        "env_diff",
    ],
)
def test__backward_logprob(env_local, request):
    env = request.getfixturevalue(env_local)
    for _ in range(100):
        angles = np.random.rand(2).tolist()
        state = angles + [1.0]
        env.set_state(state, done=False)
        mask = torch.unsqueeze(
            tbool(env.get_mask_invalid_actions_backward(), device=env.device), 0
        )
        policy_output = env.get_policy_output(env.fixed_distr_params).unsqueeze(0)
        actions = torch.tensor([angles])
        logprobs = env.get_logprobs(
            policy_output, actions, mask, [state], is_backward=True
        )
        assert torch.allclose(logprobs, torch.tensor([0.0]))


def tests__unroll_trajectory(env):
    traj_states = []
    traj_actions = []

    env.reset()

    done = False

    while not done:
        traj_states.append(copy(env.state))
        random_policy = torch.unsqueeze(env.random_policy_output, 0)
        mask = torch.unsqueeze(
            tbool(env.get_mask_invalid_actions_forward(), device=env.device), 0
        )
        actions = env.sample_actions_batch(
            random_policy,
            mask,
            [env.state],
            is_backward=False,
        )
        traj_actions.append(actions[0])
        env.step(actions[0])
        done = env.done
    assert len(traj_states) == len(traj_actions) == env.length_traj + 1
