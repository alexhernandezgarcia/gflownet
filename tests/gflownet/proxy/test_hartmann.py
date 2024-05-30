import pytest
import torch

from gflownet.envs.cube import ContinuousCube
from gflownet.envs.grid import Grid
from gflownet.proxy.box.hartmann import X_DOMAIN, Hartmann
from gflownet.utils.common import tfloat


@pytest.fixture()
def proxy_default():
    return Hartmann(device="cpu", float_precision=32)


@pytest.fixture()
def proxy_negate_exp_reward():
    return Hartmann(
        negate=True,
        reward_function="exponential",
        reward_function_kwargs={"beta": 1.0},
        device="cpu",
        float_precision=32,
    )


@pytest.fixture()
def proxy_fid01_exp_reward():
    return Hartmann(
        fidelity=0.1,
        reward_function="exponential",
        reward_function_kwargs={"beta": -1.0},
        device="cpu",
        float_precision=32,
    )


@pytest.fixture
def grid():
    return Grid(n_dim=6, length=10, device="cpu")


@pytest.fixture
def cube():
    return ContinuousCube(n_dim=6, n_comp=3, min_incr=0.1)


@pytest.mark.parametrize(
    "samples, samples_standard_domain",
    [
        (
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -0.5, 0.5, -0.2, 0.2, 1.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.25, 0.75, 0.4, 0.6, 1.0],
            ],
        ),
    ],
)
def test__map_to_standard_domain__returns_expected(
    proxy_default, samples, samples_standard_domain
):
    proxy = proxy_default
    samples = tfloat(samples, float_type=proxy.float, device=proxy.device)
    samples_standard_domain = tfloat(
        samples_standard_domain, float_type=proxy.float, device=proxy.device
    )
    assert torch.allclose(
        proxy.map_to_standard_domain(samples), samples_standard_domain
    )


@pytest.mark.parametrize(
    "proxy, samples, proxy_expected",
    [
        (
            "proxy_default",
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -0.5, 0.5, -0.2, 0.2, 1.0],
                [0.2, 0.2, 0.5, 0.3, 0.3, 0.7],
            ],
            [-5.4972e-35, -3.4085e-05, -2.5341e-04, -3.2216],
        ),
        (
            "proxy_negate_exp_reward",
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -0.5, 0.5, -0.2, 0.2, 1.0],
                [0.2, 0.2, 0.5, 0.3, 0.3, 0.7],
            ],
            [5.4972e-35, 3.4085e-05, 2.5341e-04, 3.2216],
        ),
        (
            "proxy_fid01_exp_reward",
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -0.5, 0.5, -0.2, 0.2, 1.0],
                [0.2, 0.2, 0.5, 0.3, 0.3, 0.7],
            ],
            [-5.4971e-35, -3.4084e-05, -2.5340e-04, -3.1874],
        ),
    ],
)
def test__proxy__returns_expected(proxy, samples, proxy_expected, request):
    proxy = request.getfixturevalue(proxy)
    samples = tfloat(samples, float_type=proxy.float, device=proxy.device)
    proxy_expected = tfloat(proxy_expected, float_type=proxy.float, device=proxy.device)
    assert torch.allclose(proxy(samples), proxy_expected, atol=1e-04)


@pytest.mark.parametrize(
    "proxy, samples, rewards_expected",
    [
        (
            "proxy_default",
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -0.5, 0.5, -0.2, 0.2, 1.0],
                [0.2, 0.2, 0.5, 0.3, 0.3, 0.7],
            ],
            [5.4972e-35, 3.4085e-05, 2.5341e-04, 3.2216],
        ),
        (
            "proxy_negate_exp_reward",
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -0.5, 0.5, -0.2, 0.2, 1.0],
                [0.2, 0.2, 0.5, 0.3, 0.3, 0.7],
            ],
            [1.0000, 1.0000, 1.0003, 25.0672],
        ),
        (
            "proxy_fid01_exp_reward",
            [
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-1.0, -0.5, 0.5, -0.2, 0.2, 1.0],
                [0.2, 0.2, 0.5, 0.3, 0.3, 0.7],
            ],
            [1.0000, 1.0000, 1.0003, 24.2251],
        ),
    ],
)
def test__rewards__returns_expected(proxy, samples, rewards_expected, request):
    proxy = request.getfixturevalue(proxy)
    samples = tfloat(samples, float_type=proxy.float, device=proxy.device)
    rewards_expected = tfloat(
        rewards_expected, float_type=proxy.float, device=proxy.device
    )
    assert torch.allclose(proxy.rewards(samples), rewards_expected, atol=1e-04)


@pytest.mark.parametrize(
    "proxy, max_reward_expected",
    [
        (
            "proxy_default",
            3.32237,
        ),
        (
            "proxy_negate_exp_reward",
            27.7260,
        ),
    ],
)
def test__get_max_reward__returns_expected(proxy, max_reward_expected, request):
    proxy = request.getfixturevalue(proxy)
    assert torch.isclose(proxy.get_max_reward(), torch.tensor(max_reward_expected))


@pytest.mark.parametrize(
    "proxy, env",
    [
        (
            "proxy_default",
            "grid",
        ),
        (
            "proxy_negate_exp_reward",
            "grid",
        ),
        (
            "proxy_fid01_exp_reward",
            "grid",
        ),
        (
            "proxy_default",
            "cube",
        ),
        (
            "proxy_negate_exp_reward",
            "cube",
        ),
        (
            "proxy_fid01_exp_reward",
            "cube",
        ),
    ],
)
def test__env_states_are_within_expected_domain(proxy, env, request):
    proxy = request.getfixturevalue(proxy)
    env = request.getfixturevalue(env)
    # Generate a batch of states
    states = env.states2proxy(env.get_random_terminating_states(100))
    # Map states to Hartmann domain
    states_hartmann_domain = proxy.map_to_standard_domain(states)
    # Check that domain is correct
    assert torch.all(states_hartmann_domain >= X_DOMAIN[0])
    assert torch.all(states_hartmann_domain <= X_DOMAIN[1])
    # Simple checks that proxy values and rewards can be computed
    assert torch.all(torch.isfinite(proxy(states)))
    assert torch.all(proxy.rewards(states) >= 0.0)
