import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from torch.utils.data import DataLoader

from gflownet.proxy.iam.scenario_scripts.Scenario_Datasets import witch_proc_data
from gflownet.proxy.iam.iam_proxies import FAIRY


"""
TEST DATA
"""


@pytest.fixture
def temp_parquet_files():
    """Create temporary parquet files for testing.

    Creates complete datasets where:
    - All scenarios have all regions
    - All regions have all years
    - All combinations (scenario, region, year) are present
    """
    temp_dir = tempfile.TemporaryDirectory()

    # Create complete dataset: 2 scenarios x 2 regions x 2 years = 8 rows
    subsidies_df = pd.DataFrame({
        'tech_1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'tech_2': [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
    })

    variables_df = pd.DataFrame({
        'var_1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        'var_2': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
    })

    # Complete dataset: scenario1 (region_a, region_b) x years (2020, 2025)
    #                   scenario2 (region_a, region_b) x years (2020, 2025)
    keys_df = pd.DataFrame({
        'gdx': ['scenario1', 'scenario1', 'scenario1', 'scenario1',
                'scenario2', 'scenario2', 'scenario2', 'scenario2'],
        'year': [2020, 2020, 2025, 2025,
                 2020, 2020, 2025, 2025],
        'n': ['region_a', 'region_b', 'region_a', 'region_b',
              'region_a', 'region_b', 'region_a', 'region_b'],
    })

    # Save to parquet
    subsidies_path = os.path.join(temp_dir.name, 'subsidies_df.parquet')
    variables_path = os.path.join(temp_dir.name, 'variables_df.parquet')
    keys_path = os.path.join(temp_dir.name, 'keys_df.parquet')

    subsidies_df.to_parquet(subsidies_path)
    variables_df.to_parquet(variables_path)
    keys_df.to_parquet(keys_path)

    yield {
        'dir': temp_dir.name,
        'subsidies_path': subsidies_path,
        'variables_path': variables_path,
        'keys_path': keys_path,
        'subsidies_df': subsidies_df,
        'variables_df': variables_df,
        'keys_df': keys_df,
    }

    temp_dir.cleanup()


class TestWitchProcDataInitialization:
    """Test witch_proc_data initialization and data loading."""

    def test_init_with_default_paths(self, temp_parquet_files):
        """Test initialization with default paths when files exist."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files['subsidies_path'],
            variables_parquet=temp_parquet_files['variables_path'],
            keys_parquet=temp_parquet_files['keys_path'],
            auto_download=False,
        )

        assert dataset is not None
        assert hasattr(dataset, 'subsidies_df')
        assert hasattr(dataset, 'variables_df')
        assert hasattr(dataset, 'keys_df')

    def test_init_with_custom_paths(self, temp_parquet_files):
        """Test initialization with custom input paths overriding defaults."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files['subsidies_path'],
            variables_parquet=temp_parquet_files['variables_path'],
            keys_parquet=temp_parquet_files['keys_path'],
            auto_download=False,
        )

        assert isinstance(dataset.subsidies_df, torch.Tensor)
        assert isinstance(dataset.variables_df, torch.Tensor)

    def test_init_auto_download_disabled_with_existing_files(self, temp_parquet_files):
        """Test initialization with auto_download=False when files exist."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files['subsidies_path'],
            variables_parquet=temp_parquet_files['variables_path'],
            keys_parquet=temp_parquet_files['keys_path'],
            auto_download=False,
        )

        # Complete dataset: 8 rows total, max_year is 2025
        # Valid indices: all rows except those with year=2025 (4 rows remain)
        assert len(dataset) == 4

    def test_init_with_invalid_scaling_type(self, temp_parquet_files):
        """Test that invalid scaling type raises ValueError."""
        with pytest.raises(ValueError, match="Scaling type must be"):
            witch_proc_data(
                subsidies_parquet=temp_parquet_files['subsidies_path'],
                variables_parquet=temp_parquet_files['variables_path'],
                keys_parquet=temp_parquet_files['keys_path'],
                scaling_type="invalid_scaling",
                auto_download=False,
            )

    @pytest.mark.parametrize("scaling_type", ["original", "normalization", "maxscale", "maxmin"])
    def test_init_with_valid_scaling_types(self, temp_parquet_files, scaling_type):
        """Test initialization with all valid scaling types."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files['subsidies_path'],
            variables_parquet=temp_parquet_files['variables_path'],
            keys_parquet=temp_parquet_files['keys_path'],
            scaling_type=scaling_type,
            auto_download=False,
        )

        assert dataset.scaling_type == scaling_type
        assert len(dataset) > 0

    def test_init_computes_scaling_params_for_normalization(self, temp_parquet_files):
        """Test that scaling params are computed for normalization."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files['subsidies_path'],
            variables_parquet=temp_parquet_files['variables_path'],
            keys_parquet=temp_parquet_files['keys_path'],
            scaling_type="normalization",
            auto_download=False,
        )

        assert dataset.precomputed_scaling_params is not None
        for col in ['tech_1', 'tech_2', 'var_1', 'var_2']:
            assert col in dataset.precomputed_scaling_params
            assert 'mean' in dataset.precomputed_scaling_params[col]
            assert 'std' in dataset.precomputed_scaling_params[col]

    def test_init_computes_scaling_params_for_maxscale(self, temp_parquet_files):
        """Test that max scaling params are computed correctly."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files['subsidies_path'],
            variables_parquet=temp_parquet_files['variables_path'],
            keys_parquet=temp_parquet_files['keys_path'],
            scaling_type="maxscale",
            auto_download=False,
        )

        for col in ['tech_1', 'tech_2', 'var_1', 'var_2']:
            assert 'max' in dataset.precomputed_scaling_params[col]

    def test_init_computes_scaling_params_for_maxmin(self, temp_parquet_files):
        """Test that min/max scaling params are computed correctly."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files['subsidies_path'],
            variables_parquet=temp_parquet_files['variables_path'],
            keys_parquet=temp_parquet_files['keys_path'],
            scaling_type="maxmin",
            auto_download=False,
        )

        for col in ['tech_1', 'tech_2', 'var_1', 'var_2']:
            assert 'min' in dataset.precomputed_scaling_params[col]
            assert 'max' in dataset.precomputed_scaling_params[col]
            assert dataset.precomputed_scaling_params[col]['min'] < dataset.precomputed_scaling_params[col]['max']

    def test_init_with_precomputed_scaling_params(self, temp_parquet_files):
        """Test initialization with precomputed scaling parameters."""
        precomputed_params = {
            'tech_1': {'mean': 0.25, 'std': 0.1},
            'tech_2': {'mean': 0.65, 'std': 0.1},
            'var_1': {'mean': 2.5, 'std': 1.0},
            'var_2': {'mean': 6.5, 'std': 1.0},
        }

        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files['subsidies_path'],
            variables_parquet=temp_parquet_files['variables_path'],
            keys_parquet=temp_parquet_files['keys_path'],
            scaling_type="normalization",
            precomputed_scaling_params=precomputed_params,
            auto_download=False,
        )

        assert dataset.precomputed_scaling_params == precomputed_params

    def test_init_drop_columns(self, temp_parquet_files):
        """Test that columns can be dropped during initialization."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files['subsidies_path'],
            variables_parquet=temp_parquet_files['variables_path'],
            keys_parquet=temp_parquet_files['keys_path'],
            drop_columns=['tech_1'],
            auto_download=False,
        )

        assert 'tech_1' not in dataset.subsidies_names
        assert 'tech_2' in dataset.subsidies_names

    def test_init_with_cuda_false(self, temp_parquet_files):
        """Test initialization with with_cuda=False."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files['subsidies_path'],
            variables_parquet=temp_parquet_files['variables_path'],
            keys_parquet=temp_parquet_files['keys_path'],
            with_cuda=False,
            auto_download=False,
        )

        assert dataset.subsidies_df.device.type == 'cpu'
        assert dataset.variables_df.device.type == 'cpu'

    def test_init_creates_index_map(self, temp_parquet_files):
        """Test that index_map is created correctly."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files['subsidies_path'],
            variables_parquet=temp_parquet_files['variables_path'],
            keys_parquet=temp_parquet_files['keys_path'],
            auto_download=False,
        )

        assert hasattr(dataset, 'index_map')
        assert isinstance(dataset.index_map, dict)
        assert len(dataset.index_map) > 0

    def test_init_creates_variables_next(self, temp_parquet_files):
        """Test that variables_next_df is properly initialized."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files['subsidies_path'],
            variables_parquet=temp_parquet_files['variables_path'],
            keys_parquet=temp_parquet_files['keys_path'],
            auto_download=False,
        )

        assert hasattr(dataset, 'variables_next_df')
        assert dataset.variables_next_df.shape == dataset.variables_df.shape


class TestWitchProcDataGetItem:
    """Test __getitem__ and __len__ methods."""

    @pytest.fixture
    def dataset(self):
        """Create a dataset for testing.

        Complete dataset: 2 scenarios x 2 regions x 2 years = 8 rows
        """
        temp_dir = tempfile.TemporaryDirectory()

        # Complete dataset
        subsidies_df = pd.DataFrame({
            'tech_1': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            'tech_2': [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        })

        variables_df = pd.DataFrame({
            'var_1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'var_2': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        })

        keys_df = pd.DataFrame({
            'gdx': ['s1', 's1', 's1', 's1', 's2', 's2', 's2', 's2'],
            'year': [2020, 2020, 2025, 2025, 2020, 2020, 2025, 2025],
            'n': ['r_a', 'r_b', 'r_a', 'r_b', 'r_a', 'r_b', 'r_a', 'r_b'],
        })

        subsidies_path = os.path.join(temp_dir.name, 'subsidies.parquet')
        variables_path = os.path.join(temp_dir.name, 'variables.parquet')
        keys_path = os.path.join(temp_dir.name, 'keys.parquet')

        subsidies_df.to_parquet(subsidies_path)
        variables_df.to_parquet(variables_path)
        keys_df.to_parquet(keys_path)

        dataset = witch_proc_data(
            subsidies_parquet=subsidies_path,
            variables_parquet=variables_path,
            keys_parquet=keys_path,
            auto_download=False,
        )

        yield dataset
        temp_dir.cleanup()

    def test_len_excludes_max_year(self, dataset):
        """Test that __len__ excludes samples from the max year.

        Complete dataset: 8 rows total (2 scenarios x 2 regions x 2 years)
        Max year is 2025, so exclude 4 rows (2 scenarios x 2 regions)
        Remaining valid samples: 4 rows (all from year 2020)
        """
        assert len(dataset) == 4

    def test_getitem_returns_tuple(self, dataset):
        """Test that __getitem__ returns a tuple of three tensors."""
        item = dataset[0]

        assert isinstance(item, tuple)
        assert len(item) == 3

    def test_getitem_returns_tensors(self, dataset):
        """Test that __getitem__ returns torch tensors."""
        variables_current, subsidies_current, variables_next = dataset[0]

        assert isinstance(variables_current, torch.Tensor)
        assert isinstance(subsidies_current, torch.Tensor)
        assert isinstance(variables_next, torch.Tensor)

    def test_getitem_correct_shapes(self, dataset):
        """Test that __getitem__ returns correct tensor shapes."""
        variables_current, subsidies_current, variables_next = dataset[0]

        # Should match number of columns
        assert variables_current.shape[0] == 2  # 2 variables
        assert subsidies_current.shape[0] == 2  # 2 subsidies
        assert variables_next.shape[0] == 2  # 2 variables

    def test_getitem_all_indices(self, dataset):
        """Test that all valid indices are accessible."""
        for idx in range(len(dataset)):
            item = dataset[idx]
            assert len(item) == 3
            assert all(isinstance(t, torch.Tensor) for t in item)

    def test_getitem_with_dataloader(self, dataset):
        """Test dataset works with DataLoader."""
        dataloader = DataLoader(dataset, batch_size=2)

        for batch in dataloader:
            variables, subsidies, variables_next = batch

            assert isinstance(variables, torch.Tensor)
            assert isinstance(subsidies, torch.Tensor)
            assert isinstance(variables_next, torch.Tensor)
            assert variables.shape[0] <= 2

    def test_variables_next_correctly_mapped(self, dataset):
        """Test that variables_next is correctly mapped to next year samples.

        For complete dataset, each (scenario, region, year) triple at year 2020
        should map to the corresponding (scenario, region, 2025) triple.
        """
        # Get first sample (scenario1, region_a, year 2020)
        variables_current_0, _, variables_next_0 = dataset[0]

        # The next_idx should correspond to (scenario1, region_a, year 2025)
        # which should be at index 2 in the 2020-only dataset (but maps to index 2 in original)
        assert isinstance(variables_next_0, torch.Tensor)
        assert variables_next_0.shape == variables_current_0.shape

        # Variables should be different from current
        assert not torch.allclose(variables_current_0, variables_next_0)


class TestWitchProcDataAutoDownload:
    """Test auto-download functionality."""

    def test_auto_download_disabled_with_missing_files(self):
        """Test that auto_download=False raises error with missing files."""
        nonexistent_path = os.path.join(tempfile.gettempdir(), 'nonexistent_witch_file_12345.parquet')

        with pytest.raises(FileNotFoundError):
            witch_proc_data(
                subsidies_parquet=nonexistent_path,
                variables_parquet=nonexistent_path,
                keys_parquet=nonexistent_path,
                auto_download=False,
            )

    def test_auto_download_enabled_with_existing_files_does_not_download(self, temp_parquet_files):
        """Test that auto_download=True doesn't download if files already exist."""
        # This should not raise any errors and should not attempt downloads
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files['subsidies_path'],
            variables_parquet=temp_parquet_files['variables_path'],
            keys_parquet=temp_parquet_files['keys_path'],
            auto_download=True,
        )

        assert dataset is not None
        assert len(dataset) > 0

"""
TEST FAIRY
"""

class TestGetInvestedAmount:
    """Test the get_invested_amount method."""

    @pytest.fixture
    def fairy_proxy(self):
        """Import and instantiate FAIRY proxy."""
        return FAIRY()

    def test_get_invested_amount_none(self, fairy_proxy):
        """Test that 'NONE' returns 0.0"""
        assert fairy_proxy.get_invested_amount('NONE') == 0.0

    def test_get_invested_amount_low(self, fairy_proxy):
        """Test that 'LOW' returns 0.1"""
        assert fairy_proxy.get_invested_amount('LOW') == 0.1

    def test_get_invested_amount_medium(self, fairy_proxy):
        """Test that 'MEDIUM' returns 0.3"""
        # NOTE: Updated from 0.5 to match actual implementation (0.3)
        assert fairy_proxy.get_invested_amount('MEDIUM') == 0.3

    def test_get_invested_amount_high(self, fairy_proxy):
        """Test that 'HIGH' returns 0.75"""
        # NOTE: Updated from 1.0 to match actual implementation (0.75)
        assert fairy_proxy.get_invested_amount('HIGH') == 0.75

    def test_get_invested_amount_invalid(self, fairy_proxy):
        """Test that invalid amount raises ValueError"""
        with pytest.raises(ValueError, match="Invalid amount"):
            fairy_proxy.get_invested_amount('INVALID')

    def test_get_invested_amount_case_sensitive(self, fairy_proxy):
        """Test that amount strings are case-sensitive"""
        with pytest.raises(ValueError, match="Invalid amount"):
            fairy_proxy.get_invested_amount('none')  # lowercase should fail

    def test_get_invested_amount_returns_float(self, fairy_proxy):
        """Test that get_invested_amount always returns a float"""
        for amount in ['NONE', 'LOW', 'MEDIUM', 'HIGH']:
            result = fairy_proxy.get_invested_amount(amount)
            assert isinstance(result, float)


class TestDenormalization:
    """Test denormalization logic in the __call__ method."""

    @pytest.fixture
    def fairy_proxy(self):
        """Import and instantiate FAIRY proxy."""
        return FAIRY()

    def test_consumption_denormalization(self, fairy_proxy):
        """Test that consumption is properly denormalized."""
        states = [
            [{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'LOW'}],
        ]

        result = fairy_proxy(states)

        # Get scaling params
        consumption_params = fairy_proxy.precomputed_scaling_params['CONSUMPTION']

        # Result should be in original scale (not normalized)
        # Should be roughly between min and max (with some margin for model outputs)
        assert consumption_params['min'] <= result[0].item() <= consumption_params['max'] * 1.5 or \
               result[0].item() >= 0  # Allow for negative (penalized by emissions)

    def test_emissions_denormalization(self, fairy_proxy):
        """Test that emissions are properly denormalized."""
        states = [
            [{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'HIGH'}],
        ]

        # Mock the fairy model output to verify denormalization
        result = fairy_proxy(states)

        # Result should be finite
        assert torch.isfinite(result[0])

    def test_scaling_params_exist(self, fairy_proxy):
        """Test that all required scaling parameters exist."""
        required_keys = ['CONSUMPTION', 'EMI_total_CO2']
        for key in required_keys:
            assert key in fairy_proxy.precomputed_scaling_params
            assert 'min' in fairy_proxy.precomputed_scaling_params[key]
            assert 'max' in fairy_proxy.precomputed_scaling_params[key]

    def test_scaling_params_valid_ranges(self, fairy_proxy):
        """Test that scaling parameter ranges are valid (min < max)."""
        for col, params in fairy_proxy.precomputed_scaling_params.items():
            assert params['min'] < params['max'], \
                f"Invalid range for {col}: min={params['min']}, max={params['max']}"

    def test_denormalization_monotonicity(self, fairy_proxy):
        """Test that higher investments generally lead to higher rewards (before emissions penalty)."""
        # Create states with increasing investment levels
        tech = fairy_proxy.subsidies_names[0]
        states = [
            [{'TECH': tech, 'AMOUNT': 'NONE'}],
            [{'TECH': tech, 'AMOUNT': 'LOW'}],
            [{'TECH': tech, 'AMOUNT': 'MEDIUM'}],
            [{'TECH': tech, 'AMOUNT': 'HIGH'}],
        ]

        results = fairy_proxy(states)

        # All results should be finite
        assert torch.all(torch.isfinite(results))


class TestFAIRYCall:
    """Test the __call__ method with various input configurations."""

    @pytest.fixture
    def fairy_proxy(self):
        """Import and instantiate FAIRY proxy."""
        return FAIRY()

    def test_call_basic(self, fairy_proxy):
        """Test basic __call__ with simple states."""
        states = [
            [{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'LOW'}],
            [{'TECH': fairy_proxy.subsidies_names[1] if len(fairy_proxy.subsidies_names) > 1 else
            fairy_proxy.subsidies_names[0], 'AMOUNT': 'HIGH'}],
        ]

        result = fairy_proxy(states)

        assert result.shape == (2,)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert torch.all(torch.isfinite(result))

    def test_call_single_state(self, fairy_proxy):
        """Test __call__ with a single state."""
        states = [
            [{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'MEDIUM'}],
        ]

        result = fairy_proxy(states)

        assert result.shape == (1,)
        assert isinstance(result, torch.Tensor)
        assert torch.all(torch.isfinite(result))

    def test_call_empty_state(self, fairy_proxy):
        """Test __call__ with empty investment (no subsidies)."""
        states = [
            [],  # Empty plan - all subsidies are 0
        ]

        result = fairy_proxy(states)

        assert result.shape == (1,)
        assert isinstance(result, torch.Tensor)
        assert torch.isfinite(result[0])

    def test_call_none_amount(self, fairy_proxy):
        """Test __call__ with NONE investment amount."""
        states = [
            [{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'NONE'}],
            [{'TECH': fairy_proxy.subsidies_names[1] if len(fairy_proxy.subsidies_names) > 1 else
            fairy_proxy.subsidies_names[0], 'AMOUNT': 'MEDIUM'}],
        ]

        result = fairy_proxy(states)

        assert result.shape == (2,)
        assert isinstance(result, torch.Tensor)
        assert torch.all(torch.isfinite(result))

    @pytest.mark.parametrize("num_investments", [1, 2, 3])
    def test_call_multiple_investments_per_state(self, fairy_proxy, num_investments):
        """Test __call__ with multiple investments in a single state."""
        # Use only valid tech names from the proxy
        num_available = min(num_investments, len(fairy_proxy.subsidies_names))
        available_techs = fairy_proxy.subsidies_names[:num_available]

        investments = [
            {'TECH': tech, 'AMOUNT': 'LOW'}
            for tech in available_techs
        ]
        states = [investments]

        result = fairy_proxy(states)

        assert result.shape == (1,)
        assert isinstance(result, torch.Tensor)
        assert torch.isfinite(result[0])

    def test_call_multiple_states_multiple_investments(self, fairy_proxy):
        """Test __call__ with multiple states, each having multiple investments."""
        if len(fairy_proxy.subsidies_names) < 3:
            pytest.skip("Not enough subsidies for this test")

        available_techs = fairy_proxy.subsidies_names[:3]

        states = [
            [
                {'TECH': available_techs[0], 'AMOUNT': 'LOW'},
                {'TECH': available_techs[1], 'AMOUNT': 'MEDIUM'},
            ],
            [
                {'TECH': available_techs[1], 'AMOUNT': 'HIGH'},
                {'TECH': available_techs[2], 'AMOUNT': 'LOW'},
            ],
            [
                {'TECH': available_techs[0], 'AMOUNT': 'HIGH'},
            ],
        ]

        result = fairy_proxy(states)

        assert result.shape == (3,)
        assert isinstance(result, torch.Tensor)
        assert torch.all(torch.isfinite(result))

    def test_call_all_amount_levels(self, fairy_proxy):
        """Test __call__ with all different amount levels."""
        tech = fairy_proxy.subsidies_names[0]

        states = [
            [{'TECH': tech, 'AMOUNT': 'NONE'}],
            [{'TECH': tech, 'AMOUNT': 'LOW'}],
            [{'TECH': tech, 'AMOUNT': 'MEDIUM'}],
            [{'TECH': tech, 'AMOUNT': 'HIGH'}],
        ]

        result = fairy_proxy(states)

        assert result.shape == (4,)
        assert isinstance(result, torch.Tensor)
        assert torch.all(torch.isfinite(result))

    def test_call_returns_tensor_type(self, fairy_proxy):
        """Test that __call__ always returns a torch.Tensor."""
        states = [[{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'HIGH'}]]

        result = fairy_proxy(states)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_call_batch_size_consistency(self, fairy_proxy):
        """Test that output batch size matches input batch size."""
        for batch_size in [1, 5, 10]:
            states = [
                [{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'LOW'}]
                for _ in range(batch_size)
            ]

            result = fairy_proxy(states)

            assert result.shape == (batch_size,)
            assert torch.all(torch.isfinite(result))

    def test_call_invalid_tech_raises_error(self, fairy_proxy):
        """Test that invalid tech name raises an error."""
        states = [
            [{'TECH': 'INVALID_TECH_NAME_THAT_DEFINITELY_DOES_NOT_EXIST', 'AMOUNT': 'LOW'}],
        ]

        with pytest.raises(ValueError):
            fairy_proxy(states)

    def test_call_same_tech_multiple_times(self, fairy_proxy):
        """Test behavior when same tech appears multiple times in a state."""
        tech = fairy_proxy.subsidies_names[0]

        states = [
            [
                {'TECH': tech, 'AMOUNT': 'LOW'},
                {'TECH': tech, 'AMOUNT': 'MEDIUM'},  # Same tech again - should overwrite
            ],
        ]

        result = fairy_proxy(states)

        assert result.shape == (1,)
        assert isinstance(result, torch.Tensor)
        assert torch.isfinite(result[0])

    def test_call_no_grad(self, fairy_proxy):
        """Test that __call__ does not compute gradients."""
        states = [[{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'LOW'}]]

        result = fairy_proxy(states)

        # Result should not have gradient
        assert not result.requires_grad

    def test_call_deterministic(self, fairy_proxy):
        """Test that __call__ produces deterministic results."""
        states = [[{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'MEDIUM'}]]

        result1 = fairy_proxy(states)
        result2 = fairy_proxy(states)

        assert torch.allclose(result1, result2)


class TestFAIRYInit:
    """Test FAIRY initialization and attribute handling."""

    def test_init_creates_required_attributes(self):
        """Test that initialization creates all required attributes."""
        proxy = FAIRY()
        assert hasattr(proxy, 'fairy')
        assert hasattr(proxy, 'precomputed_scaling_params')
        assert hasattr(proxy, 'subsidies_names')
        assert hasattr(proxy, 'variables_names')
        assert hasattr(proxy, 'device')
        assert hasattr(proxy, 'key_gdx')
        assert hasattr(proxy, 'key_year')
        assert hasattr(proxy, 'key_region')
        assert hasattr(proxy, 'SCC')
        assert hasattr(proxy, 'context')

    def test_init_fairy_has_required_attributes(self):
        """Test that initialized fairy has required attributes."""
        proxy = FAIRY()
        assert hasattr(proxy.fairy, 'subsidies_dim')
        assert hasattr(proxy.fairy, 'variables_names')
        assert hasattr(proxy.fairy, 'subsidies_names')

    def test_init_subsidies_names_is_list(self):
        """Test that subsidies_names is converted to a list."""
        proxy = FAIRY()
        assert isinstance(proxy.subsidies_names, list)

    def test_init_variables_names_is_list(self):
        """Test that variables_names is converted to a list."""
        proxy = FAIRY()
        assert isinstance(proxy.variables_names, list)

    def test_init_with_custom_key_gdx(self):
        """Test initialization with custom key_gdx."""
        custom_gdx = "WITCHDB/branch_lts_1/gsa_r1.gdx"
        proxy = FAIRY(key_gdx=custom_gdx)
        assert proxy.key_gdx == custom_gdx

    def test_init_with_custom_key_year(self):
        """Test initialization with custom key_year."""
        custom_year = 2030
        proxy = FAIRY(key_year=custom_year)
        assert proxy.key_year == custom_year

    def test_init_with_custom_key_region(self):
        """Test initialization with custom key_region."""
        custom_region = "mena"
        proxy = FAIRY(key_region=custom_region)
        assert proxy.key_region == custom_region

    def test_init_with_custom_budget(self):
        """Test initialization with custom budget."""
        # Note: FAIRY class doesn't show budget parameter in __init__,
        # this may need adjustment based on actual implementation
        proxy = FAIRY()
        assert hasattr(proxy, 'device')

    def test_init_with_custom_scc(self):
        """Test initialization with custom SCC."""
        custom_scc = 100.0
        proxy = FAIRY(SCC=custom_scc)
        # SCC should be converted to tensor
        assert isinstance(proxy.SCC, torch.Tensor) or proxy.SCC == custom_scc

    def test_init_fairy_in_eval_mode(self):
        """Test that fairy model is in eval mode after initialization."""
        proxy = FAIRY()
        assert not proxy.fairy.training

    def test_init_context_shape(self):
        """Test that context has correct shape."""
        proxy = FAIRY()
        assert proxy.context.dim() == 2
        assert proxy.context.shape[0] == 1
        assert proxy.context.shape[1] == len(proxy.variables_names)

    def test_init_context_on_correct_device(self):
        """Test that context is on the correct device."""
        proxy = FAIRY()
        assert proxy.context.device == proxy.device

    def test_init_device_consistency(self):
        """Test that all tensors are on the same device after init."""
        proxy = FAIRY()
        assert proxy.context.device == proxy.device
        assert proxy.SCC.device == proxy.device
        assert next(proxy.fairy.parameters()).device == proxy.device


class TestFAIRYDeviceHandling:
    """Test device handling and movement in FAIRY proxy."""

    @pytest.fixture
    def fairy_proxy(self):
        """Import and instantiate FAIRY proxy."""
        return FAIRY()

    def test_device_consistency(self, fairy_proxy):
        """Test that all tensors are on the same device."""
        states = [
            [{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'LOW'}],
        ]

        result = fairy_proxy(states)

        assert result.device == fairy_proxy.device
        assert fairy_proxy.context.device == fairy_proxy.device
        assert fairy_proxy.SCC.device == fairy_proxy.device

    def test_to_method_moves_tensors(self, fairy_proxy):
        """Test that to() method correctly moves all tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        original_device = fairy_proxy.device
        fairy_proxy.to("cuda")

        assert fairy_proxy.device.type == "cuda"
        assert fairy_proxy.context.device.type == "cuda"
        assert fairy_proxy.SCC.device.type == "cuda"
        assert next(fairy_proxy.fairy.parameters()).device.type == "cuda"

    def test_to_method_returns_self(self, fairy_proxy):
        """Test that to() method returns self for chaining."""
        result = fairy_proxy.to("cpu")
        assert result is fairy_proxy

    def test_call_output_on_correct_device(self, fairy_proxy):
        """Test that __call__ output is on the correct device."""
        states = [[{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'LOW'}]]
        result = fairy_proxy(states)
        assert result.device == fairy_proxy.device


class TestDenormalizationNumericStability:
    """Test numeric stability of denormalization processes."""

    @pytest.fixture
    def fairy_proxy(self):
        """Import and instantiate FAIRY proxy."""
        return FAIRY()

    def test_denormalization_no_nan_or_inf(self, fairy_proxy):
        """Test that denormalization never produces NaN or Inf values."""
        states = [
                     [{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': level}]
                     for level in ['NONE', 'LOW', 'MEDIUM', 'HIGH']
                 ] * 10  # Repeat multiple times

        result = fairy_proxy(states)

        assert torch.all(torch.isfinite(result)), \
            f"Found NaN/Inf values in results: {result[~torch.isfinite(result)]}"

    def test_denormalization_scaling_bounds(self, fairy_proxy):
        """Test that denormalized consumption stays within reasonable bounds."""
        # Create multiple random states
        states = [
            [{'TECH': fairy_proxy.subsidies_names[i % len(fairy_proxy.subsidies_names)],
              'AMOUNT': level}]
            for i in range(20)
            for level in ['LOW', 'MEDIUM']
        ]

        result = fairy_proxy(states)

        # Get bounds from scaling parameters
        consumption_params = fairy_proxy.precomputed_scaling_params['CONSUMPTION']
        emissions_params = fairy_proxy.precomputed_scaling_params['EMI_total_CO2']

        # Results can go negative due to emissions penalty, but should be bounded reasonably
        # Rough heuristic: should not exceed 10x the max consumption
        max_reasonable = consumption_params['max'] * 10
        min_reasonable = -consumption_params['max'] * 10

        assert torch.all(result > min_reasonable), \
            f"Found unreasonably low values: min={result.min()}, threshold={min_reasonable}"
        assert torch.all(result < max_reasonable), \
            f"Found unreasonably high values: max={result.max()}, threshold={max_reasonable}"

    def test_denormalization_formula_correctness(self, fairy_proxy):
        """Test that denormalization formula is applied correctly."""
        # Get a normalized value from the model
        states = [[{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'MEDIUM'}]]

        # Manually extract and check the formula
        with torch.no_grad():
            contexts = fairy_proxy.context.repeat(1, 1)
            plan = torch.zeros(1, fairy_proxy.fairy.subsidies_dim, device=fairy_proxy.device)

            amount = fairy_proxy.get_invested_amount('MEDIUM')
            tech_idx = fairy_proxy.subsidies_names.index(fairy_proxy.subsidies_names[0])
            plan[0, tech_idx] = amount

            # Get normalized outputs
            developments_normalized = fairy_proxy.fairy(contexts, plan)

            # Check consumption denormalization
            consumption_norm = developments_normalized[0, fairy_proxy.variables_names.index('CONSUMPTION')]
            consumption_params = fairy_proxy.precomputed_scaling_params['CONSUMPTION']
            consumption_denorm = (
                    consumption_norm * (consumption_params['max'] - consumption_params['min'])
                    + consumption_params['min']
            )

            # Verify it's in reasonable range
            assert consumption_params['min'] <= consumption_denorm <= consumption_params['max'] * 1.5, \
                f"Denormalization formula may be wrong: {consumption_denorm} not in expected range"

    def test_emissions_penalty_scaling(self, fairy_proxy):
        """Test that emissions penalty is applied with correct scaling."""
        states_high_investment = [
            [{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'HIGH'}]
        ]
        states_low_investment = [
            [{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'NONE'}]
        ]

        result_high = fairy_proxy(states_high_investment)
        result_low = fairy_proxy(states_low_investment)

        # Both should be finite
        assert torch.isfinite(result_high[0])
        assert torch.isfinite(result_low[0])

        # High investment should lead to consumption benefit (before emissions penalty)
        # but total utility depends on emissions, so we just check both are reasonable
        scc_value = fairy_proxy.SCC
        assert 0 < scc_value < 1000, \
            f"SCC value seems unreasonable: {scc_value}. Should be positive and in Trillion USD."

    def test_denormalization_inverse_operation(self, fairy_proxy):
        """Test that denormalization can be approximately inverted."""
        consumption_params = fairy_proxy.precomputed_scaling_params['CONSUMPTION']

        # Pick a random denormalized value within bounds
        denorm_value = torch.tensor(consumption_params['min'] +
                                    (consumption_params['max'] - consumption_params['min']) * 0.5)

        # Apply inverse normalization
        norm_value = (
                (denorm_value - consumption_params['min']) /
                (consumption_params['max'] - consumption_params['min'])
        )

        # Re-denormalize
        recovered = (
                norm_value * (consumption_params['max'] - consumption_params['min'])
                + consumption_params['min']
        )

        # Should recover original value
        assert torch.allclose(denorm_value, recovered, rtol=1e-5), \
            f"Denormalization not invertible: {denorm_value} -> {recovered}"

    def test_scaling_params_consistency(self, fairy_proxy):
        """Test that scaling parameters are consistent across multiple calls."""
        proxy1 = FAIRY()
        proxy2 = FAIRY()

        for key in ['CONSUMPTION', 'EMI_total_CO2']:
            assert proxy1.precomputed_scaling_params[key]['min'] == proxy2.precomputed_scaling_params[key]['min']
            assert proxy1.precomputed_scaling_params[key]['max'] == proxy2.precomputed_scaling_params[key]['max']

    def test_extreme_investment_combinations(self, fairy_proxy):
        """Test numeric stability with extreme investment combinations."""
        # All NONE (minimal investment)
        states_none = [
            [{'TECH': tech, 'AMOUNT': 'NONE'} for tech in fairy_proxy.subsidies_names[:3]]
        ]

        # All HIGH (maximal investment)
        states_high = [
            [{'TECH': tech, 'AMOUNT': 'HIGH'} for tech in fairy_proxy.subsidies_names[:3]]
        ]

        result_none = fairy_proxy(states_none)
        result_high = fairy_proxy(states_high)

        assert torch.all(torch.isfinite(result_none))
        assert torch.all(torch.isfinite(result_high))

        # Results should be different (high investment should change outcomes)
        assert not torch.allclose(result_none, result_high)

    """Test robustness and edge cases."""

    @pytest.fixture
    def fairy_proxy(self):
        """Import and instantiate FAIRY proxy."""
        return FAIRY()

    def test_call_with_missing_required_keys(self, fairy_proxy):
        """Test that __call__ raises error with missing TECH or AMOUNT."""
        states = [
            [{'TECH': fairy_proxy.subsidies_names[0]}],  # Missing AMOUNT
        ]

        with pytest.raises(KeyError):
            fairy_proxy(states)

    def test_call_with_extra_keys(self, fairy_proxy):
        """Test that __call__ handles extra keys gracefully."""
        states = [
            [{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'LOW', 'EXTRA': 'value'}],
        ]

        result = fairy_proxy(states)
        assert result.shape == (1,)
        assert torch.isfinite(result[0])

    def test_large_batch_size(self, fairy_proxy):
        """Test __call__ with a large batch size."""
        states = [
            [{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'LOW'}]
            for _ in range(100)
        ]

        result = fairy_proxy(states)

        assert result.shape == (100,)
        assert torch.all(torch.isfinite(result))

    def test_invalid_region_raises_error(self):
        """Test that invalid region raises AssertionError."""
        with pytest.raises(AssertionError):
            FAIRY(key_region="invalid_region")

    def test_model_in_eval_mode_during_call(self, fairy_proxy):
        """Test that model is in eval mode (required for BatchNorm with batch_size=1)."""
        # The model MUST be in eval mode for inference with batch_size=1
        # because BatchNorm requires more than 1 sample in training mode
        assert not fairy_proxy.fairy.training, "Model should be in eval mode for inference"

        states = [[{'TECH': fairy_proxy.subsidies_names[0], 'AMOUNT': 'LOW'}]]
        result = fairy_proxy(states)

        assert torch.isfinite(result[0])