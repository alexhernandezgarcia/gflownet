import pytest
import torch

from gflownet.envs.cube import ContinuousCube
from gflownet.envs.grid import Grid
from gflownet.proxy.box.rings import Rings
from gflownet.utils.common import tfloat


@pytest.fixture()
def proxy_default():
    return Rings(device="cpu", float_precision=32)


@pytest.fixture()
def proxy_offcenter():
    """
    Rings proxy with the following characteristics:
    - The center of the rings is not the center of the box (square).
    - The two Gaussians overlap: their separation (0.3) is twice sigma, so the maximum
    of the mixture lies between the two radii and is larger than the score on either
    ring.
    """
    return Rings(
        radii=(0.3, 0.6),
        sigma=0.15,
        center=(0.5, 0.5),
        device="cpu",
        float_precision=32,
    )


@pytest.fixture()
def proxy_exp_reward():
    return Rings(
        radii=(0.2, 0.8),
        sigma=0.1,
        reward_function="exponential",
        reward_function_kwargs={"beta": 0.01},
        device="cpu",
        float_precision=32,
    )


@pytest.fixture
def grid2d():
    return Grid(n_dim=2, length=10, device="cpu")


@pytest.fixture
def grid3d():
    return Grid(n_dim=3, length=10, device="cpu")


@pytest.fixture
def cube2d():
    return ContinuousCube(n_dim=2, n_comp=3, min_incr=0.1)


@pytest.fixture
def cube3d():
    return ContinuousCube(n_dim=3, n_comp=3, min_incr=0.1)


@pytest.mark.parametrize(
    "proxy",
    [
        ("proxy_default"),
        ("proxy_offcenter"),
        ("proxy_exp_reward"),
    ],
)
def test__proxy__initializes_properly(proxy, request):
    proxy = request.getfixturevalue(proxy)
    assert True


@pytest.mark.parametrize(
    "env",
    [
        ("grid2d"),
        ("grid3d"),
        ("cube2d"),
        ("cube3d"),
    ],
)
def test__env__initializes_properly(env, request):
    env = request.getfixturevalue(env)
    assert True


@pytest.mark.parametrize(
    "proxy, env",
    [
        ("proxy_default", "grid2d"),
        ("proxy_default", "grid3d"),
        ("proxy_default", "cube2d"),
        ("proxy_default", "cube3d"),
        ("proxy_offcenter", "grid2d"),
        ("proxy_offcenter", "cube2d"),
        ("proxy_exp_reward", "grid2d"),
        ("proxy_exp_reward", "grid3d"),
        ("proxy_exp_reward", "cube2d"),
        ("proxy_exp_reward", "cube3d"),
    ],
)
def test__proxy_is_setup_from_env_without_errors(proxy, env, request):
    proxy = request.getfixturevalue(proxy)
    env = request.getfixturevalue(env)
    proxy.setup(env)
    assert True


@pytest.mark.parametrize(
    "center",
    [
        # Wrong length
        (0.5,),
        # Wrong length
        (0.5, 0.5, 0.5),
        # Out of the box
        (1.5, 0.0),
        # Out of the box
        (0.0, -1.5),
    ],
)
def test__invalid_center__raises_value_error(center):
    proxy = Rings(n_dim=2, radii=(0.2, 0.8), sigma=0.1, center=center)
    with pytest.raises(ValueError):
        proxy.setup()


@pytest.mark.parametrize(
    "proxy, env",
    [
        ("proxy_offcenter", "grid3d"),
        ("proxy_offcenter", "cube3d"),
    ],
)
def test__invalid_center__raises_value_error__from_env_setup(proxy, env, request):
    proxy = request.getfixturevalue(proxy)
    env = request.getfixturevalue(env)
    with pytest.raises(ValueError):
        proxy.setup(env)


@pytest.mark.parametrize(
    "proxy, samples",
    [
        (
            "proxy_default",
            [
                [0.2, 0.0],
                [0.8, 0.0],
                [0.0, 0.8],
                [0.0, 0.0],
                [0.5, 0.5],
                [1.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 1.0],
            ],
        ),
        (
            "proxy_default",
            [
                [0.2, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 0.8],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
            ],
        ),
        (
            "proxy_offcenter",
            [
                [0.8, 0.5],
                [0.5, 0.8],
                [-0.1, 0.5],
                [0.5, 0.5],
                [1.0, 1.0],
                [-1.0, -1.0],
            ],
        ),
        (
            "proxy_exp_reward",
            [
                [0.2, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 0.8],
                [0.0, 0.0, 0.2],
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
            ],
        ),
    ],
)
def test__proxy__returns_expected(proxy, samples, request):
    proxy = request.getfixturevalue(proxy)
    samples = tfloat(samples, float_type=proxy.float, device=proxy.device)
    proxy.n_dim = samples.shape[1]
    proxy.setup()
    proxy_values = proxy(samples)
    # Calculate proxy values more manually, one by one
    for x, p_val in zip(samples, proxy_values):
        proxy_expected = 0.0
        for radius in proxy.radii:
            proxy_expected += torch.exp(
                -((torch.linalg.norm(x - proxy.center_vec) - radius) ** 2)
                / (2 * proxy.sigma**2)
            )
        assert torch.isclose(p_val, proxy_expected, atol=1e-04)


@pytest.mark.parametrize(
    "proxy, optimum_expected",
    [
        # The ring peaks barely overlap (separation 0.6 = 6 sigma), so the
        # optimum is essentially the score on a ring: 1.0.
        ("proxy_default", 1.0),
        # With overlapping Gaussians, the maximum of the mixture (1.213061)
        # lies at the midpoint between the two radii, not on a ring.
        ("proxy_offcenter", 1.213061),
    ],
)
def test__optimum__returns_expected(proxy, optimum_expected, request):
    proxy = request.getfixturevalue(proxy)
    proxy.n_dim = 2
    proxy.setup()
    assert torch.isclose(
        proxy.optimum,
        tfloat([optimum_expected], float_type=proxy.float, device=proxy.device)[0],
    )


@pytest.mark.parametrize(
    "proxy",
    [
        ("proxy_default"),
        ("proxy_offcenter"),
        ("proxy_exp_reward"),
    ],
)
def test__optimum__is_not_exceeded_by_any_state(proxy, request):
    proxy = request.getfixturevalue(proxy)
    proxy.n_dim = 2
    proxy.setup()
    # No point of the box should score above the optimum
    xs = torch.linspace(-1.0, 1.0, 201)
    xx, yy = torch.meshgrid(xs, xs, indexing="xy")
    states = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    scores = proxy(states.to(device=proxy.device, dtype=proxy.float))
    assert torch.all(scores <= proxy.optimum + 1e-5)


def test__call__three_dimensions():
    proxy = Rings(n_dim=3, radii=(0.5,), sigma=0.1, device="cpu", float_precision=32)
    proxy.setup()
    # All four points are at an L2 distance of 0.5 from the center
    samples = [
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5 / 3**0.5, 0.5 / 3**0.5, 0.5 / 3**0.5],
    ]
    samples = tfloat(samples, float_type=proxy.float, device=proxy.device)
    assert torch.allclose(proxy(samples), torch.ones(4, dtype=proxy.float), atol=1e-04)


@pytest.mark.parametrize(
    "proxy, env",
    [
        ("proxy_default", "grid2d"),
        ("proxy_default", "grid3d"),
        ("proxy_default", "cube2d"),
        ("proxy_default", "cube3d"),
        ("proxy_offcenter", "grid2d"),
        ("proxy_offcenter", "cube2d"),
        ("proxy_exp_reward", "grid2d"),
        ("proxy_exp_reward", "grid3d"),
        ("proxy_exp_reward", "cube2d"),
        ("proxy_exp_reward", "cube3d"),
    ],
)
def test__env_states_are_scored_within_expected_range(proxy, env, request):
    proxy = request.getfixturevalue(proxy)
    env = request.getfixturevalue(env)
    proxy.setup(env)
    # Generate a batch of states and map them to the proxy domain
    if hasattr(env, "get_all_terminating_states"):
        states = env.states2proxy(env.get_all_terminating_states())
    else:
        states = env.states2proxy(env.get_random_terminating_states(100))
    # Simple checks that proxy values and rewards can be computed, and that
    # they lie in the expected range
    scores = proxy(states)
    assert torch.all(torch.isfinite(scores))
    assert torch.all(scores >= 0.0)
    assert torch.all(scores <= proxy.optimum + 1e-5)
    assert torch.all(proxy.rewards(states) >= 0.0)
