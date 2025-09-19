import numpy as np
import pytest
import torch

from gflownet.utils.common import index_select


@pytest.mark.parametrize(
    "iterable, index, expected",
    [
        ([1, 2, 3, 4, 5, 6], [0, 4, 3], [1, 5, 4]),
        (
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [1, 0],
            [[5, 6, 7, 8], [1, 2, 3, 4]],
        ),
    ],
)
def test__index_select(iterable, index, expected):
    output = index_select(iterable, index)
    assert output == expected
    index_tuple = tuple(index)
    output = index_select(iterable, index_tuple)
    assert output == expected
    index_tensor = torch.tensor(index)
    output = index_select(iterable, index_tensor)
    assert output == expected
    index_np = np.array(index)
    output = index_select(iterable, index_np)
    assert output == expected

    iterable_tuple = tuple(iterable)
    expected_tuple = tuple(expected)
    output = index_select(iterable_tuple, index)
    assert output == expected_tuple
    output = index_select(iterable_tuple, index_tuple)
    assert output == expected_tuple
    output = index_select(iterable_tuple, index_tensor)
    assert output == expected_tuple
    output = index_select(iterable_tuple, index_np)
    assert output == expected_tuple

    iterable_tensor = torch.tensor(iterable)
    expected_tensor = torch.tensor(expected)
    output = index_select(iterable_tensor, index)
    assert torch.equal(output, expected_tensor)
    output = index_select(iterable_tensor, index_tuple)
    assert torch.equal(output, expected_tensor)

    output = index_select(iterable_tensor, index_tensor)
    assert torch.equal(output, expected_tensor)
    output = index_select(iterable_tensor, index_np)
    assert torch.equal(output, expected_tensor)

    iterable_np = np.array(iterable)
    expected_np = np.array(expected)
    output = index_select(iterable_np, index)
    assert np.all(output == expected_np)
    output = index_select(iterable_np, index_tuple)
    assert np.all(output == expected_np)
    output = index_select(iterable_np, index_tensor)
    assert np.all(output == expected_np)
    output = index_select(iterable_np, index_np)
    assert np.all(output == expected_np)
