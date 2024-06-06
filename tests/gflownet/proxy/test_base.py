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
        reward_function="identity",
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
def proxy_exponential(beta, alpha):
    return Uniform(
        reward_function="exponential",
        reward_function_kwargs={"beta": beta, "alpha": alpha},
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
def proxy_rbf_exponential(center, beta, alpha, distance):
    return Uniform(
        reward_function="rbf_exp",
        reward_function_kwargs={
            "center": center,
            "beta": beta,
            "alpha": alpha,
            "distance": distance,
        },
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


@pytest.fixture()
def proxy_exponential_clipped(beta, alpha):
    return Uniform(
        reward_function="exponential",
        reward_function_kwargs={"beta": beta, "alpha": alpha},
        reward_min=1e-03,
        do_clip_rewards=True,
        device="cpu",
        float_precision=32,
    )


@pytest.mark.parametrize(
    "proxy, beta, alpha, center, distance",
    [
        ("uniform", None, None, None, None),
        ("proxy_power", 1, None, None, None),
        ("proxy_power", 2, None, None, None),
        ("proxy_exponential", 1, 1, None, None),
        ("proxy_exponential", -1, 1, None, None),
        ("proxy_exponential", 1, 2, None, None),
        ("proxy_exponential", -1, 3, None, None),
        ("proxy_shift", 5, None, None, None),
        ("proxy_shift", -5, None, None, None),
        ("proxy_product", 2, None, None, None),
        ("proxy_product", -2, None, None, None),
        ("proxy_rbf_exponential", 1.0, 1.0, 0.0, "squared"),
        ("proxy_rbf_exponential", -1.0, 2.0, 1.34, "squared"),
        ("proxy_rbf_exponential", -1.0, 2.0, 1.34, "euclidean"),
        ("proxy_rbf_exponential", 2.0, 2.0, -0.5, "abs"),
    ],
)
def test__uniform_proxy_initializes_without_errors(
    proxy, beta, alpha, center, distance, request
):
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


def check_proxy2reward(rewards_computed, rewards_expected, atol=1e-4, ignore_inf=False):
    if ignore_inf:
        rewards_expected = rewards_expected[torch.isfinite(rewards_computed)]
        rewards_computed = rewards_computed[torch.isfinite(rewards_computed)]
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
        (
            2,
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
    # Log rewards, computing the log of the rewards
    assert all(
        check_proxy2reward(
            torch.log(rewards_exp),
            proxy.proxy2logreward(proxy_values),
            atol=1e-1,
            ignore_inf=True,
        )
    )
    assert all(
        check_proxy2reward(
            torch.log(proxy.proxy2reward(proxy_values)),
            proxy.proxy2logreward(proxy_values),
            atol=1e-1,
            ignore_inf=True,
        )
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
    # Log rewards, computing the log of the rewards
    assert all(
        check_proxy2reward(
            torch.log(rewards_exp),
            proxy.proxy2logreward(proxy_values),
            atol=1e-1,
            ignore_inf=True,
        )
    )
    assert all(
        check_proxy2reward(
            torch.log(proxy.proxy2reward(proxy_values)),
            proxy.proxy2logreward(proxy_values),
            atol=1e-1,
            ignore_inf=True,
        )
    )


@pytest.mark.parametrize(
    "beta, alpha, proxy_values, rewards_exp, logrewards_exp, logrewards_exp_clipped",
    [
        (
            1.0,
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
            1.0,
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
        (
            1.0,
            2.0,
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
            [
                4.54e-05 * 2.0,
                3.6788e-01 * 2.0,
                6.0653e-01 * 2.0,
                9.0484e-01 * 2.0,
                1.0 * 2.0,
                1.1052 * 2.0,
                1.6487e00 * 2.0,
                2.7183 * 2.0,
                22026.4648 * 2.0,
            ],
            [
                -10 + np.log(2.0),
                -1 + np.log(2.0),
                -0.5 + np.log(2.0),
                -0.1 + np.log(2.0),
                0.0 + np.log(2.0),
                0.1 + np.log(2.0),
                0.5 + np.log(2.0),
                1 + np.log(2.0),
                10 + np.log(2.0),
            ],
            [
                -10 + np.log(2.0),
                -1 + np.log(2.0),
                -0.5 + np.log(2.0),
                -0.1 + np.log(2.0),
                0.0 + np.log(2.0),
                0.1 + np.log(2.0),
                0.5 + np.log(2.0),
                1 + np.log(2.0),
                10 + np.log(2.0),
            ],
        ),
        (
            -1.0,
            2.0,
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
            [
                22026.4648 * 2.0,
                2.7183 * 2.0,
                1.6487 * 2.0,
                1.1052 * 2.0,
                1.0 * 2.0,
                9.0484e-01 * 2.0,
                6.0653e-01 * 2.0,
                3.6788e-01 * 2.0,
                4.54e-05 * 2.0,
            ],
            [
                10 + np.log(2.0),
                1 + np.log(2.0),
                0.5 + np.log(2.0),
                0.1 + np.log(2.0),
                0.0 + np.log(2.0),
                -0.1 + np.log(2.0),
                -0.5 + np.log(2.0),
                -1 + np.log(2.0),
                -10 + np.log(2.0),
            ],
            [
                10 + np.log(2.0),
                1 + np.log(2.0),
                0.5 + np.log(2.0),
                0.1 + np.log(2.0),
                0.0 + np.log(2.0),
                -0.1 + np.log(2.0),
                -0.5 + np.log(2.0),
                -1 + np.log(2.0),
                -10 + np.log(2.0),
            ],
        ),
    ],
)
def test_reward_function_exponential__behaves_as_expected(
    proxy_exponential,
    beta,
    alpha,
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
    # Log rewards, computing the log of the rewards
    assert all(
        check_proxy2reward(
            torch.log(rewards_exp),
            proxy.proxy2logreward(proxy_values),
            atol=1e-1,
            ignore_inf=True,
        )
    )
    assert all(
        check_proxy2reward(
            torch.log(proxy.proxy2reward(proxy_values)),
            proxy.proxy2logreward(proxy_values),
            atol=1e-1,
            ignore_inf=True,
        )
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
    # Log rewards, computing the log of the rewards
    assert all(
        check_proxy2reward(
            torch.log(rewards_exp),
            proxy.proxy2logreward(proxy_values),
            atol=1e-1,
            ignore_inf=True,
        )
    )
    assert all(
        check_proxy2reward(
            torch.log(proxy.proxy2reward(proxy_values)),
            proxy.proxy2logreward(proxy_values),
            atol=1e-1,
            ignore_inf=True,
        )
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
    # Log rewards, computing the log of the rewards
    assert all(
        check_proxy2reward(
            torch.log(rewards_exp),
            proxy.proxy2logreward(proxy_values),
            atol=1e-1,
            ignore_inf=True,
        )
    )
    assert all(
        check_proxy2reward(
            torch.log(proxy.proxy2reward(proxy_values)),
            proxy.proxy2logreward(proxy_values),
            atol=1e-1,
            ignore_inf=True,
        )
    )


@pytest.mark.parametrize(
    "beta, alpha, center, distance, proxy_values, rewards_exp, logrewards_exp, "
    "logrewards_exp_clipped",
    [
        (
            1.0,
            1.0,
            0.0,
            "squared",
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
            [
                2.6881e43,
                2.7182e00,
                1.2840e00,
                1.0100e00,
                1.0e00,
                1.0100e00,
                1.2840e00,
                2.7182e00,
                2.6881e43,
            ],
            [
                100,
                1.0,
                0.25,
                0.01,
                0.0,
                0.01,
                0.25,
                1.0,
                100,
            ],
            [
                100,
                1.0,
                0.25,
                0.01,
                0.0,
                0.01,
                0.25,
                1.0,
                100,
            ],
        ),
        (
            -1.0,
            2.0,
            0.5,
            "squared",
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
            [
                2.6306e-48,
                0.2107,
                0.7357,
                1.3953,
                1.5576,
                1.7042,
                2.0,
                1.5576,
                1.2763e-39,
            ],
            [
                -109.5568,
                -1.5568,
                -0.3068,
                0.3331,
                0.4431,
                0.5331,
                0.6931,
                0.4431,
                -89.5568,
            ],
            [
                -109.5568,
                -1.5568,
                -0.3068,
                0.3331,
                0.4431,
                0.5331,
                0.6931,
                0.4431,
                -89.5568,
            ],
        ),
        (
            -1.0,
            2.0,
            0.5,
            "euclidean",
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
            [
                5.5072e-05,
                0.4462,
                0.7357,
                1.0976,
                1.2130,
                1.3406,
                2.0,
                1.2130,
                1.4970e-04,
            ],
            [
                -9.8068,
                -0.8068,
                -0.3068,
                0.0931,
                0.1931,
                0.2931,
                0.6931,
                0.1931,
                -8.8068,
            ],
            [
                -9.8068,
                -0.8068,
                -0.3068,
                0.0931,
                0.1931,
                0.2931,
                0.6931,
                0.1931,
                -8.8068,
            ],
        ),
        (
            -8.0,
            2.0,
            -0.5,
            "abs",
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
            [
                1.9708e-33,
                0.0366,
                2.0,
                0.0815,
                0.0366,
                0.0164,
                6.7092e-04,
                1.2288e-05,
                6.6114e-37,
            ],
            [
                -75.3068,
                -3.3068,
                0.6931,
                -2.5068,
                -3.3068,
                -4.1068,
                -7.3068,
                -11.3068,
                -83.3068,
            ],
            [
                -75.3068,
                -3.3068,
                0.6931,
                -2.5068,
                -3.3068,
                -4.1068,
                -7.3068,
                -11.3068,
                -83.3068,
            ],
        ),
    ],
)
def test_reward_function_rbf_exponential__behaves_as_expected(
    proxy_rbf_exponential,
    beta,
    alpha,
    center,
    distance,
    proxy_values,
    rewards_exp,
    logrewards_exp,
    logrewards_exp_clipped,
):
    proxy = proxy_rbf_exponential
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
    # Log rewards, computing the log of the rewards
    assert all(
        check_proxy2reward(
            torch.log(rewards_exp),
            proxy.proxy2logreward(proxy_values),
            atol=1e-1,
            ignore_inf=True,
        )
    )
    assert all(
        check_proxy2reward(
            torch.log(proxy.proxy2reward(proxy_values)),
            proxy.proxy2logreward(proxy_values),
            atol=1e-1,
            ignore_inf=True,
        )
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


@pytest.mark.parametrize(
    "beta, alpha, proxy_values, rewards_exp, logrewards_exp",
    [
        (
            1.0,
            1.0,
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
            [
                1e-03,
                3.6788e-01,
                6.0653e-01,
                9.0484e-01,
                1.0,
                1.1052,
                1.6487e00,
                2.7183,
                22026.4648,
            ],
            [-6.9077, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
        ),
        (
            -1.0,
            1.0,
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
                1e-03,
            ],
            [10, 1, 0.5, 0.1, 0.0, -0.1, -0.5, -1, -6.9077],
        ),
        (
            1.0,
            2.0,
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
            [
                1e-03,
                3.6788e-01 * 2.0,
                6.0653e-01 * 2.0,
                9.0484e-01 * 2.0,
                1.0 * 2.0,
                1.1052 * 2.0,
                1.6487e00 * 2.0,
                2.7183 * 2.0,
                22026.4648 * 2.0,
            ],
            [
                -6.9077,
                -1 + np.log(2.0),
                -0.5 + np.log(2.0),
                -0.1 + np.log(2.0),
                0.0 + np.log(2.0),
                0.1 + np.log(2.0),
                0.5 + np.log(2.0),
                1 + np.log(2.0),
                10 + np.log(2.0),
            ],
        ),
        (
            -1.0,
            2.0,
            [-10, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 10],
            [
                22026.4648 * 2.0,
                2.7183 * 2.0,
                1.6487 * 2.0,
                1.1052 * 2.0,
                1.0 * 2.0,
                9.0484e-01 * 2.0,
                6.0653e-01 * 2.0,
                3.6788e-01 * 2.0,
                1e-03,
            ],
            [
                10 + np.log(2.0),
                1 + np.log(2.0),
                0.5 + np.log(2.0),
                0.1 + np.log(2.0),
                0.0 + np.log(2.0),
                -0.1 + np.log(2.0),
                -0.5 + np.log(2.0),
                -1 + np.log(2.0),
                -6.9077,
            ],
        ),
    ],
)
def test_reward_function_exponential__clipped__behaves_as_expected(
    proxy_exponential_clipped,
    beta,
    alpha,
    proxy_values,
    rewards_exp,
    logrewards_exp,
):
    proxy = proxy_exponential_clipped
    proxy_values = tfloat(proxy_values, device=proxy.device, float_type=proxy.float)
    # Rewards
    rewards_exp = tfloat(rewards_exp, device=proxy.device, float_type=proxy.float)
    assert all(check_proxy2reward(proxy.proxy2reward(proxy_values), rewards_exp))
    # Log Rewards
    logrewards_exp = tfloat(logrewards_exp, device=proxy.device, float_type=proxy.float)
    assert all(check_proxy2reward(proxy.proxy2logreward(proxy_values), logrewards_exp))


@pytest.mark.parametrize(
    "proxy, beta, alpha, center, distance, optimum, reward_max",
    [
        ("uniform", None, None, None, None, 1.0, 1.0),
        ("uniform", None, None, None, None, 2.0, 2.0),
        ("proxy_power", 1, None, None, None, 2.0, 2.0),
        ("proxy_power", 2, None, None, None, 2.0, 4.0),
        ("proxy_exponential", 1, 1, None, None, 1.0, np.exp(1.0)),
        ("proxy_exponential", -1, 1, None, None, -1.0, np.exp(1.0)),
        ("proxy_exponential", 1, 2, None, None, 1.0, 2 * np.exp(1.0)),
        ("proxy_exponential", -1, 3, None, None, -1.0, 3 * np.exp(1.0)),
        ("proxy_shift", 5, None, None, None, 10.0, 15.0),
        ("proxy_shift", -5, None, None, None, 10.0, 5.0),
        ("proxy_product", 2, None, None, None, 2.0, 4.0),
        ("proxy_product", -2, None, None, None, -5.0, 10.0),
        ("proxy_rbf_exponential", -1, 1, 0, "squared", 0.0, 1.0),
        ("proxy_rbf_exponential", -1, 2, 0, "squared", 0.0, 2.0),
        ("proxy_rbf_exponential", 1, 1, 0.5, "squared", -1.0, 9.487735836358526),
        ("proxy_rbf_exponential", -8, 2, 1.34, "abs", 1.34, 2.0),
    ],
)
def test__get_max_rewards__returns_expected(
    proxy, beta, alpha, center, distance, optimum, reward_max, request
):
    proxy = request.getfixturevalue(proxy)
    reward_max = torch.tensor(reward_max, dtype=proxy.float, device=proxy.device)
    # Forcibly set the optimum for testing purposes, even if the proxy is uniform.
    proxy.optimum = torch.tensor(optimum)
    assert torch.isclose(proxy.get_max_reward(log=False), reward_max)
    assert torch.isclose(proxy.get_max_reward(log=True), torch.log(reward_max))
