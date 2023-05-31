import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from gflownet.envs.crystals.composition import Composition
from gflownet.proxy.crystals.composition import CompositionMPFrequency

DEVICE = "cpu"
FLOAT = 32


@pytest.fixture
def proxy_no_norm():
    return CompositionMPFrequency(normalise=False, device=DEVICE, float_precision=FLOAT)


@pytest.fixture
def proxy_norm():
    return CompositionMPFrequency(normalise=True, device=DEVICE, float_precision=FLOAT)


@pytest.mark.parametrize(
    (
        "elements, required_elements, max_diff_elem, min_diff_elem, min_atoms,"
        "max_atoms, min_atom_i, max_atom_i, expected_state"
    ),
    [
        ([1, 10, 25, 16, 50], [], 4, 2, 2, 20, 1, 10, [0, 0, 0, 10, 10]),
        ([1, 10, 25, 16, 50], [1], 4, 2, 2, 20, 1, 10, [1, 0, 0, 9, 10]),
        ([1, 10, 25, 16, 50], [1], 4, 2, 2, 20, 2, 10, [2, 0, 0, 8, 10]),
        ([1, 10, 25, 16, 50], [1, 10, 16], 4, 2, 2, 20, 2, 10, [2, 2, 6, 0, 10]),
        ([1, 10, 25, 16, 50], [1, 10, 50], 4, 2, 2, 20, 2, 10, [2, 2, 0, 6, 10]),
        ([1, 10, 25, 16, 50], [1, 10, 50], 5, 4, 2, 20, 2, 10, [2, 2, 0, 6, 10]),
        ([1, 10, 25, 16, 50], [1, 10, 50], 5, 4, 2, 20, 2, 20, [2, 2, 0, 2, 14]),
    ],
)
def test_max_protons_number(
    proxy_no_norm,
    elements,
    required_elements,
    max_diff_elem,
    min_diff_elem,
    min_atoms,
    max_atoms,
    min_atom_i,
    max_atom_i,
    expected_state,
):
    env = Composition(
        proxy=proxy_no_norm,
        elements=elements,
        required_elements=required_elements,
        max_diff_elem=max_diff_elem,
        min_diff_elem=min_diff_elem,
        min_atoms=min_atoms,
        max_atoms=max_atoms,
        min_atom_i=min_atom_i,
        max_atom_i=max_atom_i,
        float_precision=FLOAT,
        device=DEVICE,
    )
    # import ipdb; ipdb.set_trace();
    max_protons_state = proxy_no_norm._get_max_protons_state(env)
    assert max_protons_state == expected_state
    expected_n_protons = np.sum(np.array(sorted(elements)) * np.array(expected_state))
    max_protons_number = proxy_no_norm.get_max_n_protons(env)
    assert max_protons_number == expected_n_protons


@pytest.fixture
def small_env_no_norm(proxy_no_norm):
    env = Composition(
        proxy=proxy_no_norm,
        elements=10,
        required_elements=[],
        max_diff_elem=4,
        min_diff_elem=2,
        min_atoms=2,
        max_atoms=20,
        min_atom_i=1,
        max_atom_i=10,
        float_precision=FLOAT,
        device=DEVICE,
    )
    return env


@pytest.mark.parametrize(
    "state",
    [
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ([0, 10, 0, 0, 10, 0, 0, 0, 0, 0]),
        ([0, 0, 0, 0, 10, 1, 0, 9, 0, 0]),
        ([0, 0, 0, 0, 2, 0, 8, 0, 0, 10]),
        ([0, 0, 0, 1, 1, 0, 0, 0, 1, 7]),
    ],
)
def test_proxy_call(proxy_no_norm, small_env_no_norm, state):
    with open(
        Path(__file__).parents[4]
        / "gflownet/proxy/crystals/number_of_protons_counts.pkl",
        "rb",
    ) as handle:
        protons_number_counts = pickle.load(handle)
    idx = np.sum(np.array(state) * np.arange(1, 11))
    expected = -protons_number_counts[idx] if idx in protons_number_counts.keys() else 0
    assert (
        expected
        == proxy_no_norm(
            small_env_no_norm.statetorch2proxy(
                torch.tensor([state]).to(proxy_no_norm.atomic_numbers)
            )
        ).item()
    )
