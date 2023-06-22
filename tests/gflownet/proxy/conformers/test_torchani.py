import numpy as np
import pytest
import torch

from gflownet.proxy.conformers.torchani import TorchANIMoleculeEnergy
from gflownet.utils.molecule.rdkit_conformer import get_dummy_ad_rdkconf


@pytest.fixture()
def proxy():
    return TorchANIMoleculeEnergy(use_ensemble=False, device="cpu", float_precision=32)


def test__torchani_molecule_energy__predicts_energy_for_a_single_numpy_conformer(proxy):
    conf = get_dummy_ad_rdkconf()
    coordinates, elements = conf.get_atom_positions(), conf.get_atomic_numbers()
    state = np.concatenate((np.expand_dims(elements, axis=1), coordinates), axis=1)

    assert proxy(state[np.newaxis, ...]).shape == torch.Size([1])


def test__torchani_molecule_energy__predicts_energy_for_a_pytorch_batch(proxy):
    conf = get_dummy_ad_rdkconf()
    coordinates, elements = conf.get_atom_positions(), conf.get_atomic_numbers()

    coordinates = torch.Tensor(coordinates).repeat(3, 1, 1)
    elements = torch.Tensor(elements).repeat(3, 1).unsqueeze(-1)
    state = torch.concat((elements, coordinates), dim=-1)

    assert proxy(state).shape == torch.Size([3])
