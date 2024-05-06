import numpy as np
import pytest
import torch

from gflownet.proxy.base import LOGZERO, Proxy
from gflownet.proxy.uniform import Uniform
from gflownet.utils.common import tfloat


@pytest.fixture()
def uniform():
    return Uniform(device="cpu", float_precision=32)


@pytest.fixture()
def proxy_identity(beta):
    return Uniform(
        reward_function="power",
        reward_function_kwargs={"beta": beta},
        device="cpu",
        float_precision=32,
    )


@pytest.fixture()
def proxy_power(beta):
    return Uniform(
        reward_function="power",
        reward_function_kwargs={"beta": beta},
        device="cpu",
        float_precision=32,
    )


@pytest.fixture()
def proxy_exponential(beta):
    return Uniform(
        reward_function="exponential",
        reward_function_kwargs={"beta": beta},
        device="cpu",
        float_precision=32,
    )


@pytest.fixture()
def proxy_shift(beta):
    return Uniform(
        reward_function="shift",
        reward_function_kwargs={"beta": beta},
        device="cpu",
        float_precision=32,
    )


@pytest.fixture()
def proxy_product(beta):
    return Uniform(
        reward_function="product",
        reward_function_kwargs={"beta": beta},
        device="cpu",
        float_precision=32,
    )


@pytest.fixture()
def proxy_callable(reward_function, logreward_function):
    return Uniform(
        reward_function=reward_function,
        logreward_function=logreward_function,
        device="cpu",
        float_precision=32,
    )


@pytest.mark.parametrize(
    "proxy, beta",
    [
        ("uniform", None),
        ("proxy_power", 1),
        ("proxy_power", 2),
        ("proxy_exponential", 1),
        ("proxy_exponential", -1),
        ("proxy_shift", 5),
        ("proxy_shift", -5),
        ("proxy_product", 2),
        ("proxy_product", -2),
    ],
)
def test__uniform_proxy_initializes_without_errors(proxy, beta, request):
    proxy = request.getfixturevalue(proxy)
    assert True


@pytest.mark.parametrize(
    "proxy, reward_function, logreward_function",
    [
        ("proxy_callable", lambda x: x + 1, None),
        ("proxy_callable", lambda x: torch.exp(x - 1), lambda x: x - 1),
    ],
)
def test__uniform_proxy_callable_initializes_without_errors(
    proxy, reward_function, logreward_function, request
):
    proxy = request.getfixturevalue(proxy)
    assert True


def check_proxy2reward(rewards_computed, rewards_expected, atol=1e-3):
    comp_nan = rewards_computed.isnan()
    exp_nan = rewards_expected.isnan()
    notnan_allclose = torch.all(
        torch.isclose(
            rewards_computed[~comp_nan], rewards_expected[~exp_nan], atol=atol
        )
    )
    nan_equal = torch.equal(comp_nan, exp_nan)
    return notnan_allclose, nan_equal


@pytest.mark.parametrize(
    "beta, proxy_values, rewards_exp, logrewards_exp, logrewards_exp_clipped",
    [
        (
            1,
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -np.inf,
                -2.3025,
                -0.6931,
                0.0,
                2.3025,
                4.6052,
            ],
            [
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                -2.3025,
                -0.6931,
                0.0,
                2.3025,
                4.6052,
            ],
        ),
    ],
)
def test_reward_function_identity__behaves_as_expected(
    proxy_identity,
    beta,
    proxy_values,
    rewards_exp,
    logrewards_exp,
    logrewards_exp_clipped,
):
    proxy = proxy_identity
    proxy_values = tfloat(proxy_values, device=proxy.device, float_type=proxy.float)
    # Rewards
    rewards_exp = tfloat(rewards_exp, device=proxy.device, float_type=proxy.float)
    assert all(check_proxy2reward(proxy._reward_function(proxy_values), rewards_exp))
    assert all(check_proxy2reward(proxy.proxy2reward(proxy_values), rewards_exp))
    # Log Rewards
    logrewards_exp = tfloat(logrewards_exp, device=proxy.device, float_type=proxy.float)
    logrewards_exp_clipped = tfloat(
        logrewards_exp_clipped, device=proxy.device, float_type=proxy.float
    )
    assert all(
        check_proxy2reward(proxy._logreward_function(proxy_values), logrewards_exp)
    )
    assert all(
        check_proxy2reward(proxy.proxy2logreward(proxy_values), logrewards_exp_clipped)
    )


@pytest.mark.parametrize(
    "beta, proxy_values, rewards_exp, logrewards_exp, logrewards_exp_clipped",
    [
        (
            1,
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -np.inf,
                -2.3025,
                -0.6931,
                0.0,
                2.3025,
                4.6052,
            ],
            [
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                -2.3025,
                -0.6931,
                0.0,
                2.3025,
                4.6052,
            ],
        ),
        (
            2,
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
            [10000, 100, 1, 0.25, 0.01, 0.0, 0.01, 0.25, 1, 100, 10000],
            [
                9.2103,
                4.6052,
                0.0,
                -1.3863,
                -4.6052,
                -np.inf,
                -4.6052,
                -1.3863,
                0.0,
                4.6052,
                9.2103,
            ],
            [
                9.2103,
                4.6052,
                0.0,
                -1.3863,
                -4.6052,
                LOGZERO,
                -4.6052,
                -1.3863,
                0.0,
                4.6052,
                9.2103,
            ],
        ),
    ],
)
def test_reward_function_power__behaves_as_expected(
    proxy_power, beta, proxy_values, rewards_exp, logrewards_exp, logrewards_exp_clipped
):
    proxy = proxy_power
    proxy_values = tfloat(proxy_values, device=proxy.device, float_type=proxy.float)
    # Rewards
    rewards_exp = tfloat(rewards_exp, device=proxy.device, float_type=proxy.float)
    assert all(check_proxy2reward(proxy._reward_function(proxy_values), rewards_exp))
    assert all(check_proxy2reward(proxy.proxy2reward(proxy_values), rewards_exp))
    # Log Rewards
    logrewards_exp = tfloat(logrewards_exp, device=proxy.device, float_type=proxy.float)
    logrewards_exp_clipped = tfloat(
        logrewards_exp_clipped, device=proxy.device, float_type=proxy.float
    )
    assert all(
        check_proxy2reward(proxy._logreward_function(proxy_values), logrewards_exp)
    )
    assert all(
        check_proxy2reward(proxy.proxy2logreward(proxy_values), logrewards_exp_clipped)
    )


@pytest.mark.parametrize(
    "beta, proxy_values, rewards_exp, logrewards_exp, logrewards_exp_clipped",
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
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
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
            [10, 1, 0.5, 0.1, 0.0, -0.1, -0.5, -1, -10],
            [10, 1, 0.5, 0.1, 0.0, -0.1, -0.5, -1, -10],
        ),
    ],
)
def test_reward_function_exponential__behaves_as_expected(
    proxy_exponential,
    beta,
    proxy_values,
    rewards_exp,
    logrewards_exp,
    logrewards_exp_clipped,
):
    proxy = proxy_exponential
    proxy_values = tfloat(proxy_values, device=proxy.device, float_type=proxy.float)
    # Rewards
    rewards_exp = tfloat(rewards_exp, device=proxy.device, float_type=proxy.float)
    assert all(check_proxy2reward(proxy._reward_function(proxy_values), rewards_exp))
    assert all(check_proxy2reward(proxy.proxy2reward(proxy_values), rewards_exp))
    # Log Rewards
    logrewards_exp = tfloat(logrewards_exp, device=proxy.device, float_type=proxy.float)
    logrewards_exp_clipped = tfloat(
        logrewards_exp_clipped, device=proxy.device, float_type=proxy.float
    )
    assert all(
        check_proxy2reward(proxy._logreward_function(proxy_values), logrewards_exp)
    )
    assert all(
        check_proxy2reward(proxy.proxy2logreward(proxy_values), logrewards_exp_clipped)
    )


@pytest.mark.parametrize(
    "beta, proxy_values, rewards_exp, logrewards_exp, logrewards_exp_clipped",
    [
        (
            5,
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
            [-95, -5, 4, 4.5, 4.9, 5.0, 5.1, 5.5, 6, 15, 105],
            [
                np.nan,
                np.nan,
                1.3863,
                1.5041,
                1.5892,
                1.6094,
                1.6292,
                1.7047,
                1.7918,
                2.7081,
                4.6540,
            ],
            [
                LOGZERO,
                LOGZERO,
                1.3863,
                1.5041,
                1.5892,
                1.6094,
                1.6292,
                1.7047,
                1.7918,
                2.7081,
                4.6540,
            ],
        ),
        (
            -5,
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
            [-105, -15, -6, -5.5, -5.1, -5.0, -4.9, -4.5, -4, 5, 95],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                1.6094,
                4.5539,
            ],
            [
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                1.6094,
                4.5539,
            ],
        ),
    ],
)
def test_reward_function_shift__behaves_as_expected(
    proxy_shift, beta, proxy_values, rewards_exp, logrewards_exp, logrewards_exp_clipped
):
    proxy = proxy_shift
    proxy_values = tfloat(proxy_values, device=proxy.device, float_type=proxy.float)
    # Rewards
    rewards_exp = tfloat(rewards_exp, device=proxy.device, float_type=proxy.float)
    assert all(check_proxy2reward(proxy._reward_function(proxy_values), rewards_exp))
    assert all(check_proxy2reward(proxy.proxy2reward(proxy_values), rewards_exp))
    # Log Rewards
    logrewards_exp = tfloat(logrewards_exp, device=proxy.device, float_type=proxy.float)
    logrewards_exp_clipped = tfloat(
        logrewards_exp_clipped, device=proxy.device, float_type=proxy.float
    )
    assert all(
        check_proxy2reward(proxy._logreward_function(proxy_values), logrewards_exp)
    )
    assert all(
        check_proxy2reward(proxy.proxy2logreward(proxy_values), logrewards_exp_clipped)
    )


@pytest.mark.parametrize(
    "beta, proxy_values, rewards_exp, logrewards_exp, logrewards_exp_clipped",
    [
        (
            2,
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
            [-200, -20, -2, -1.0, -0.2, 0.0, 0.2, 1.0, 2, 20, 200],
            [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -np.inf,
                -1.6094,
                0.0,
                0.6931,
                2.9957,
                5.2983,
            ],
            [
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                -1.6094,
                0.0,
                0.6931,
                2.9957,
                5.2983,
            ],
        ),
        (
            -2,
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
            [200, 20, 2, 1.0, 0.2, 0.0, -0.2, -1.0, -2, -20, -200],
            [
                5.2983,
                2.9957,
                0.6931,
                0.0,
                -1.6094,
                -np.inf,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            [
                5.2983,
                2.9957,
                0.6931,
                0.0,
                -1.6094,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
                LOGZERO,
            ],
        ),
    ],
)
def test_reward_function_product__behaves_as_expected(
    proxy_product,
    beta,
    proxy_values,
    rewards_exp,
    logrewards_exp,
    logrewards_exp_clipped,
):
    proxy = proxy_product
    proxy_values = tfloat(proxy_values, device=proxy.device, float_type=proxy.float)
    # Rewards
    rewards_exp = tfloat(rewards_exp, device=proxy.device, float_type=proxy.float)
    assert all(check_proxy2reward(proxy._reward_function(proxy_values), rewards_exp))
    assert all(check_proxy2reward(proxy.proxy2reward(proxy_values), rewards_exp))
    # Log Rewards
    logrewards_exp = tfloat(logrewards_exp, device=proxy.device, float_type=proxy.float)
    logrewards_exp_clipped = tfloat(
        logrewards_exp_clipped, device=proxy.device, float_type=proxy.float
    )
    assert all(
        check_proxy2reward(proxy._logreward_function(proxy_values), logrewards_exp)
    )
    assert all(
        check_proxy2reward(proxy.proxy2logreward(proxy_values), logrewards_exp_clipped)
    )


@pytest.mark.parametrize(
    "reward_function, logreward_function, proxy_values, rewards_exp, logrewards_exp",
    [
        (
            lambda x: x + 1,
            None,
            [-100, -10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10, 100],
            [-99, -9, 0, 0.5, 0.9, 1.0, 1.1, 1.5, 2, 11, 101],
            [
                LOGZERO,
                LOGZERO,
                LOGZERO,
                -0.6931,
                -0.1054,
                0.0,
                0.0953,
                0.4055,
                0.6931,
                2.3979,
                4.6151,
            ],
        ),
        (
            lambda x: torch.exp(x - 1),
            lambda x: x - 1,
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
            [
                1.6702e-05,
                1.3534e-01,
                2.2313e-01,
                3.3287e-01,
                3.6788e-01,
                4.0657e-01,
                6.0653e-01,
                1.0,
                8.1031e03,
            ],
            [-11, -2, -1.5, -1.1, -1, -0.9, -0.5, 0, 9],
        ),
        (
            lambda x: torch.exp(x - 1),
            None,
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
            [
                1.6702e-05,
                1.3534e-01,
                2.2313e-01,
                3.3287e-01,
                3.6788e-01,
                4.0657e-01,
                6.0653e-01,
                1.0,
                8.1031e03,
            ],
            [-11, -2, -1.5, -1.1, -1, -0.9, -0.5, 0, 9],
        ),
    ],
)
def test_reward_function_callable__behaves_as_expected(
    proxy_callable,
    reward_function,
    logreward_function,
    proxy_values,
    rewards_exp,
    logrewards_exp,
):
    proxy = proxy_callable
    proxy_values = tfloat(proxy_values, device=proxy.device, float_type=proxy.float)
    # Rewards
    rewards_exp = tfloat(rewards_exp, device=proxy.device, float_type=proxy.float)
    assert all(check_proxy2reward(proxy._reward_function(proxy_values), rewards_exp))
    assert all(check_proxy2reward(proxy.proxy2reward(proxy_values), rewards_exp))
    # Log Rewards
    logrewards_exp = tfloat(logrewards_exp, device=proxy.device, float_type=proxy.float)
    assert all(check_proxy2reward(proxy.proxy2logreward(proxy_values), logrewards_exp))
