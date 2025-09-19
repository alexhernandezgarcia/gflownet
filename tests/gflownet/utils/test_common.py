import numpy as np
import pytest
import torch

from gflownet.utils.common import select_indices


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
def test__select_indices(iterable, index, expected):
    output = select_indices(iterable, index)
    assert output == expected
    index_tuple = tuple(index)
    output = select_indices(iterable, index_tuple)
    assert output == expected
    index_tensor = torch.tensor(index)
    output = select_indices(iterable, index_tensor)
    assert output == expected
    index_np = np.array(index)
    output = select_indices(iterable, index_np)
    assert output == expected

    iterable_tuple = tuple(iterable)
    expected_tuple = tuple(expected)
    output = select_indices(iterable_tuple, index)
    assert output == expected_tuple
    output = select_indices(iterable_tuple, index_tuple)
    assert output == expected_tuple
    output = select_indices(iterable_tuple, index_tensor)
    assert output == expected_tuple
    output = select_indices(iterable_tuple, index_np)
    assert output == expected_tuple

    iterable_tensor = torch.tensor(iterable)
    expected_tensor = torch.tensor(expected)
    output = select_indices(iterable_tensor, index)
    assert torch.equal(output, expected_tensor)
    output = select_indices(iterable_tensor, index_tuple)
    assert torch.equal(output, expected_tensor)

    output = select_indices(iterable_tensor, index_tensor)
    assert torch.equal(output, expected_tensor)
    output = select_indices(iterable_tensor, index_np)
    assert torch.equal(output, expected_tensor)

    iterable_np = np.array(iterable)
    expected_np = np.array(expected)
    output = select_indices(iterable_np, index)
    assert np.all(output == expected_np)
    output = select_indices(iterable_np, index_tuple)
    assert np.all(output == expected_np)
    output = select_indices(iterable_np, index_tensor)
    assert np.all(output == expected_np)
    output = select_indices(iterable_np, index_np)
    assert np.all(output == expected_np)
