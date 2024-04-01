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


@pytest.mark.parametrize("proxy, beta", [("uniform", None), ("uniform_power", 1)])
def test__uniform_proxy_initializes_without_errors(proxy, beta, request):
    proxy = request.getfixturevalue(proxy)
    return proxy


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
