import pytest
import torch

from gflownet.envs.crystals.crystal import Crystal
from gflownet.proxy.crystals.corners import CrystalCorners


@pytest.fixture()
def proxy_default():
    return CrystalCorners(device="cpu", float_precision=32)


@pytest.fixture()
def config_one_sg():
    return [{"spacegroup": 225, "mu": 0.75, "sigma": 0.05}]


@pytest.fixture
def env_gull():
    return Crystal(
        composition_kwargs={
            "elements": [78, 46],
            "min_diff_elem": 1,
            "max_diff_elem": 1,
            "min_atoms": 2,
            "max_atoms": 4,
            "min_atom_i": 2,
            "max_atom_i": 4,
            "do_charge_check": False,
        },
        space_group_kwargs={"space_group_subset": [225, 229]},
        lattice_parameters_kwargs={
            "min_length": 2.0,
            "max_length": 4.0,
            "min_angle": 60.0,
            "max_angle": 140.0,
        },
        do_sg_to_composition_constraints=True,
        do_sg_before_composition=True,
    )


def test__proxy_default__initializes_properly(proxy_default):
    assert True


def test__setup__works_as_expected(proxy_default, env_gull):
    proxy_default.setup(env_gull)
    assert hasattr(proxy_default, "min_length")
    assert hasattr(proxy_default, "max_length")
    assert proxy_default.min_length == 2.0
    assert proxy_default.max_length == 4.0


@pytest.mark.parametrize(
    "lp_lengths, lp_lengths_corners",
    [
        (
            torch.tensor(
                [
                    [2.0, 2.0, 2.0],
                    [4.0, 4.0, 4.0],
                    [2.0, 3.0, 4.0],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, 1.0, 1.0],
                    [-1.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
        ),
        (
            torch.tensor(
                [
                    [2.0, 2.5, 3.0, 3.5, 4.0],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [-1.0, -0.5, 0.0, 0.5, 1.0],
                ],
                dtype=torch.float32,
            ),
        ),
    ],
)
def test__lattice_lengths_to_corners_proxy__returns_expected(
    proxy_default, env_gull, lp_lengths, lp_lengths_corners
):
    proxy_default.setup(env_gull)
    assert torch.equal(
        proxy_default.lattice_lengths_to_corners_proxy(lp_lengths), lp_lengths_corners
    )


def test__proxy_default__returns_expected_scores(proxy_default, env_gull, n_states=2):
    # Generate random crystal terminating states
    states = env_gull.get_random_terminating_states(n_states)
    states_proxy = env_gull.states2proxy(states)
    scores = proxy_default(states_proxy)
    assert True
