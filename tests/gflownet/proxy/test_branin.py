import pytest
import torch

from gflownet.envs.cube import ContinuousCube
from gflownet.envs.grid import Grid
from gflownet.proxy.box.branin import Branin
from gflownet.utils.common import tfloat


@pytest.fixture()
def proxy_default():
    return Branin(device="cpu", float_precision=32)


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
    "samples, proxy_expected",
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
            [308.1291, 17.5083, 10.9609, 145.8722, 10.3079, 150.4520, 24.1300],
        ),
    ],
)
def test__proxy__returns_expected(proxy_default, samples, proxy_expected):
    proxy = proxy_default
    samples = tfloat(samples, float_type=proxy.float, device=proxy.device)
    proxy_expected = tfloat(proxy_expected, float_type=proxy.float, device=proxy.device)
    assert torch.allclose(proxy(samples), proxy_expected)
