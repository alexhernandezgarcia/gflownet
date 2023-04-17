import pytest
import torch
from gflownet.utils.batch import Batch


@pytest.fixture
def batch_tb():
    return Batch(loss="trajectorybalance")


@pytest.fixture
def batch_fm():
    return Batch(loss="flowmatch")


def test__len__returnszero_at_init(batch_tb, batch_fm):
    assert len(batch_tb) == 0
    assert len(batch_fm) == 0
