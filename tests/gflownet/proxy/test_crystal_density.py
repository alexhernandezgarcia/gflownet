import pytest
import torch

from gflownet.envs.crystals.crystal import Crystal
from gflownet.proxy.crystals.density import Density


@pytest.fixture()
def proxy():
    return Density(device="cpu", float_precision=32)


def test_density(proxy):
    env = Crystal(composition_kwargs={"elements": 94})

    # Initialize a proxy-format tensor with two (source) states
    states = env.states2proxy([env.source, env.source])
    # Li2O mp-1960
    states[0, [3, 8]] = torch.tensor([8, 4], dtype=torch.float)  # 8XLi, 4XO
    states[0, -6:-3] = 4.65
    states[0, -3:] = 90  # cubic lattice

    # Monoclinic Si2O mp-1063118
    states[1, [8, 14]] = torch.tensor([2, 4], dtype=torch.float)  # 2XO, 4XSi
    states[1, -6] = 7.03  # a
    states[1, -5] = 3.09  # b
    states[1, -4] = 4.46  # c
    states[1, -3] = 90  # alpha
    states[1, -2] = 114.15  # beta
    states[1, -1] = 90  # gamma

    proxy.setup(env=env)
    result = proxy(states)
    assert result == pytest.approx(torch.tensor([1.97, 2.71]), rel=1e-2)
