import pytest
import torch

from gflownet.envs.cube import ContinuousCube
from gflownet.envs.grid import Grid
from gflownet.proxy.box.branin import X1_DOMAIN, X2_DOMAIN, Branin
from gflownet.utils.common import tfloat


@pytest.fixture()
def proxy_default():
    return Branin(device="cpu", float_precision=32)


@pytest.fixture()
def proxy_negate_exp_reward():
    return Branin(
        negate=True,
        reward_function="exponential",
        reward_function_kwargs={"beta": 0.01},
        device="cpu",
        float_precision=32,
    )


@pytest.fixture()
def proxy_fid05_exp_reward():
    return Branin(
        fidelity=0.5,
        reward_function="exponential",
        reward_function_kwargs={"beta": -0.01},
        device="cpu",
        float_precision=32,
    )


@pytest.fixture
def grid():
    return Grid(n_dim=2, length=10, device="cpu")


@pytest.fixture
def cube():
    return ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1)


@pytest.mark.parametrize(
    "samples, samples_standard_domain",
    [
        (
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            [
                [-5.0, 0.0],
                [-5.0, 15.0],
                [10.0, 0.0],
                [10.0, 15.0],
                [2.5, 0.0],
                [2.5, 15.0],
                [2.5, 7.5],
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
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            [308.1291, 17.5083, 10.9609, 145.8722, 10.3079, 150.4520, 24.1300],
        ),
        (
            "proxy_negate_exp_reward",
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            [-308.1291, -17.5083, -10.9609, -145.8722, -10.3079, -150.4520, -24.1300],
        ),
        (
            "proxy_fid05_exp_reward",
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            [266.7232, 13.6024, 5.9313, 290.8426, 8.6377, 158.1568, 27.1473],
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
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            [0.8709, 291.4917, 298.0391, 163.1278, 298.6921, 158.548, 284.87],
        ),
        (
            "proxy_negate_exp_reward",
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            [0.0458, 0.8393, 0.8961, 0.2325, 0.9020, 0.2221, 0.7856],
        ),
        (
            "proxy_fid05_exp_reward",
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ],
            [0.0694, 0.8728, 0.9424, 0.0546, 0.9172, 0.2057, 0.7623],
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
            308.602113308,
        ),
        (
            "proxy_negate_exp_reward",
            0.9960290352151552,
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
            "proxy_fid05_exp_reward",
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
            "proxy_fid05_exp_reward",
            "cube",
        ),
    ],
)
def test__env_states_are_within_expected_domain(proxy, env, request):
    proxy = request.getfixturevalue(proxy)
    env = request.getfixturevalue(env)
    # Generate a batch of states
    if hasattr(env, "get_all_terminating_states"):
        states = env.states2proxy(env.get_all_terminating_states())
    else:
        states = env.states2proxy(env.get_random_terminating_states(100))
    # Map states to Branin domain
    states_branin_domain = proxy.map_to_standard_domain(states)
    # Check that domain is correct
    assert torch.all(states_branin_domain[:, 0] >= X1_DOMAIN[0])
    assert torch.all(states_branin_domain[:, 0] <= X1_DOMAIN[1])
    assert torch.all(states_branin_domain[:, 1] >= X2_DOMAIN[0])
    assert torch.all(states_branin_domain[:, 1] <= X2_DOMAIN[1])
    # Simple checks that proxy values and rewards can be computed
    assert torch.all(torch.isfinite(proxy(states)))
    assert torch.all(proxy.rewards(states) >= 0.0)
