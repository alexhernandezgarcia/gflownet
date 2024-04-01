import pytest
import torch

from gflownet.proxy.base import Proxy
from gflownet.proxy.uniform import Uniform
from gflownet.utils.common import tfloat


@pytest.fixture()
def uniform():
    return Uniform(device="cpu", float_precision=32)


@pytest.fixture()
def uniform_power(beta):
    return Uniform(
        reward_function="power",
        reward_function_kwargs={"beta": beta},
        device="cpu",
        float_precision=32,
    )


@pytest.fixture()
def uniform_exponential(beta):
    return Uniform(
        reward_function="exponential",
        reward_function_kwargs={"beta": beta},
        device="cpu",
        float_precision=32,
    )


@pytest.fixture()
def uniform_shift(beta):
    return Uniform(
        reward_function="shift",
        reward_function_kwargs={"beta": beta},
        device="cpu",
        float_precision=32,
    )


@pytest.mark.parametrize(
    "proxy, beta",
    [
        ("uniform", None),
        ("uniform_power", 1),
        ("uniform_power", 2),
        ("uniform_exponential", 1),
        ("uniform_exponential", -1),
        ("uniform_shift", 5),
        ("uniform_shift", -5),
    ],
)
def test__uniform_proxy_initializes_without_errors(proxy, beta, request):
    proxy = request.getfixturevalue(proxy)
    assert True


@pytest.mark.parametrize(
    "beta, proxy_values, rewards_exp",
    [
        (
            1,
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
        ),
        (
            2,
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
            [10000, 100, 1, 0.25, 0.01, 0.0, 0.01, 0.25, 1, 100, 10000],
        ),
    ],
)
def test_reward_function_power__behaves_as_expected(
    uniform_power, beta, proxy_values, rewards_exp
):
    proxy = uniform_power
    proxy_values = tfloat(proxy_values, device=proxy.device, float_type=proxy.float)
    rewards_exp = tfloat(rewards_exp, device=proxy.device, float_type=proxy.float)
    assert all(torch.isclose(proxy._reward_function(proxy_values), rewards_exp))
    assert all(torch.isclose(proxy.proxy2reward(proxy_values), rewards_exp))


@pytest.mark.parametrize(
    "beta, proxy_values, rewards_exp",
    [
        (
            1.0,
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
            [
                4.54e-05,
                3.6788e-01,
                6.0653e-01,
                9.0484e-01,
                1.0,
                1.1052,
                1.6487e00,
                2.7183,
                22026.4648,
            ],
        ),
        (
            -1.0,
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
            [
                22026.4648,
                2.7183,
                1.6487,
                1.1052,
                1.0,
                9.0484e-01,
                6.0653e-01,
                3.6788e-01,
                4.54e-05,
            ],
        ),
    ],
)
def test_reward_function_exponential__behaves_as_expected(
    uniform_exponential, beta, proxy_values, rewards_exp
):
    proxy = uniform_exponential
    proxy_values = tfloat(proxy_values, device=proxy.device, float_type=proxy.float)
    rewards_exp = tfloat(rewards_exp, device=proxy.device, float_type=proxy.float)
    assert all(
        torch.isclose(proxy._reward_function(proxy_values), rewards_exp, atol=1e-4)
    )
    assert all(torch.isclose(proxy.proxy2reward(proxy_values), rewards_exp, atol=1e-4))


@pytest.mark.parametrize(
    "beta, proxy_values, rewards_exp",
    [
        (
            5,
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
            [-95, -5, 4, 4.5, 4.9, 5.0, 5.1, 5.5, 6, 15, 105],
        ),
        (
            -5,
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
            [-105, -15, -6, -5.5, -5.1, -5.0, -4.9, -4.5, -4, 5, 95],
        ),
    ],
)
def test_reward_function_shift__behaves_as_expected(
    uniform_shift, beta, proxy_values, rewards_exp
):
    proxy = uniform_shift
    proxy_values = tfloat(proxy_values, device=proxy.device, float_type=proxy.float)
    rewards_exp = tfloat(rewards_exp, device=proxy.device, float_type=proxy.float)
    assert all(torch.isclose(proxy._reward_function(proxy_values), rewards_exp))
    assert all(torch.isclose(proxy.proxy2reward(proxy_values), rewards_exp))
