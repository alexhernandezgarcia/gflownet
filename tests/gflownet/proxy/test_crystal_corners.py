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
            "max_diff_elem": 1,
            "min_atoms": 2,
            "max_atoms": 4,
            "min_atom_i": 2,
            "max_atom_i": 4,
            "do_charge_check": False,
        },
        space_group_kwargs={"space_group_subset": [225, 229]},
        do_sg_to_composition_constraints=True,
        do_sg_before_composition=True,
    )


def test__proxy_default__initializes_properly(proxy_default):
    assert True
