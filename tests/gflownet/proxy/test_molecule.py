import numpy as np
import pytest
import torch

# Skip the entire module if torch_geometric is not installed / cannot be imported
try:
    import torchani
except:
    pytest.skip(
        "Skipping all tests in test_molecule.py because torch nicould not be "
        "imported.",
        allow_module_level=True,
    )

from gflownet.proxy.molecule import TorchANIMoleculeEnergy
from gflownet.utils.molecule.conformer_base import get_dummy_ad_conf_base


@pytest.fixture()
def proxy():
    return TorchANIMoleculeEnergy(use_ensemble=False, device="cpu", float_precision=32)


def test__torchani_molecule_energy__predicts_energy_for_a_single_numpy_conformer(proxy):
    conf = get_dummy_ad_conf_base()
    coordinates, elements = conf.get_atom_positions(), conf.get_atomic_numbers()

    proxy(elements[np.newaxis, ...], coordinates[np.newaxis, ...])


def test__torchani_molecule_energy__predicts_energy_for_a_pytorch_batch(proxy):
    conf = get_dummy_ad_conf_base()
    coordinates, elements = conf.get_atom_positions(), conf.get_atomic_numbers()

    coordinates = torch.Tensor(coordinates).repeat(3, 1, 1)
    elements = torch.Tensor(elements).repeat(3, 1)

    proxy(elements, coordinates)
