import numpy as np
import torch

from gflownet.utils.metrics import angles_allclose, angles_allclose_np


def test_angles_allclose():
    pi = np.pi
    two_pi = 2 * pi
    atol = 1e-2
    one = [two_pi + atol, 0.0, two_pi + atol / 2, 0.0]
    other = [0.0, 0.0, 0.0, 0.0]
    assert angles_allclose_np(np.array(one), np.array(other), atol=atol)
    assert angles_allclose(torch.tensor(one), torch.tensor(other), atol=atol)
    assert angles_allclose(one, other, atol=atol)
    atol = atol / 2
    assert not angles_allclose_np(np.array(one), np.array(other), atol=atol)
    assert not angles_allclose(torch.tensor(one), torch.tensor(other), atol=atol)
    assert not angles_allclose(one, other, atol=atol)


def test_angles_allclose_vector():
    pi = np.pi
    two_pi = 2 * pi
    atol = 1e-2
    one = [two_pi + atol, 0.0, two_pi + atol / 2, 0.0]
    other = [0.0, 0.0, 0.0, 0.0]

    result = angles_allclose_np(
        np.array([one, one, one]), np.array([other, other, other]), atol=atol
    )
    assert len(result) == 3
    assert result.all()
    result = angles_allclose(
        torch.tensor([one, other, one]), torch.tensor([other, one, other]), atol=atol
    )
    assert len(result) == 3
    assert result.all()
    result = angles_allclose([one, one, one], [other, other, other], atol=atol)
    assert len(result) == 3
    assert sum(result) == 3

    atol = atol / 2
    assert not angles_allclose_np(
        np.array([one, one, one]), np.array([other, other, other]), atol=atol
    ).all()
    assert not angles_allclose(
        torch.tensor([one, other, one]), torch.tensor([other, one, other]), atol=atol
    ).all()
    assert sum(angles_allclose([one, one, one], [other, other, other], atol=atol)) == 0
