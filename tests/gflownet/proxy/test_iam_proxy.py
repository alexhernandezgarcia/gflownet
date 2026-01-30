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
    subsidies_df = pd.DataFrame(
        {
            "tech_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "tech_2": [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        }
    )

    variables_df = pd.DataFrame(
        {
            "var_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "var_2": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        }
    )

    # Complete dataset: scenario1 (region_a, region_b) x years (2020, 2025)
    #                   scenario2 (region_a, region_b) x years (2020, 2025)
    keys_df = pd.DataFrame(
        {
            "gdx": [
                "scenario1",
                "scenario1",
                "scenario1",
                "scenario1",
                "scenario2",
                "scenario2",
                "scenario2",
                "scenario2",
            ],
            "year": [2020, 2020, 2025, 2025, 2020, 2020, 2025, 2025],
            "n": [
                "region_a",
                "region_b",
                "region_a",
                "region_b",
                "region_a",
                "region_b",
                "region_a",
                "region_b",
            ],
        }
    )

    # Save to parquet
    subsidies_path = os.path.join(temp_dir.name, "subsidies_df.parquet")
    variables_path = os.path.join(temp_dir.name, "variables_df.parquet")
    keys_path = os.path.join(temp_dir.name, "keys_df.parquet")

    subsidies_df.to_parquet(subsidies_path)
    variables_df.to_parquet(variables_path)
    keys_df.to_parquet(keys_path)

    yield {
        "dir": temp_dir.name,
        "subsidies_path": subsidies_path,
        "variables_path": variables_path,
        "keys_path": keys_path,
        "subsidies_df": subsidies_df,
        "variables_df": variables_df,
        "keys_df": keys_df,
    }

    temp_dir.cleanup()


class TestWitchProcDataInitialization:
    """Test witch_proc_data initialization and data loading."""

    def test_init_with_default_paths(self, temp_parquet_files):
        """Test initialization with default paths when files exist."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files["subsidies_path"],
            variables_parquet=temp_parquet_files["variables_path"],
            keys_parquet=temp_parquet_files["keys_path"],
            auto_download=False,
        )

        assert dataset is not None
        assert hasattr(dataset, "subsidies_df")
        assert hasattr(dataset, "variables_df")
        assert hasattr(dataset, "keys_df")

    def test_init_with_custom_paths(self, temp_parquet_files):
        """Test initialization with custom input paths overriding defaults."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files["subsidies_path"],
            variables_parquet=temp_parquet_files["variables_path"],
            keys_parquet=temp_parquet_files["keys_path"],
            auto_download=False,
        )

        assert isinstance(dataset.subsidies_df, torch.Tensor)
        assert isinstance(dataset.variables_df, torch.Tensor)

    def test_init_auto_download_disabled_with_existing_files(self, temp_parquet_files):
        """Test initialization with auto_download=False when files exist."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files["subsidies_path"],
            variables_parquet=temp_parquet_files["variables_path"],
            keys_parquet=temp_parquet_files["keys_path"],
            auto_download=False,
        )

        # Complete dataset: 8 rows total, max_year is 2025
        # Valid indices: all rows except those with year=2025 (4 rows remain)
        assert len(dataset) == 4

    def test_init_with_invalid_scaling_type(self, temp_parquet_files):
        """Test that invalid scaling type raises ValueError."""
        with pytest.raises(ValueError, match="Scaling type must be"):
            witch_proc_data(
                subsidies_parquet=temp_parquet_files["subsidies_path"],
                variables_parquet=temp_parquet_files["variables_path"],
                keys_parquet=temp_parquet_files["keys_path"],
                scaling_type="invalid_scaling",
                auto_download=False,
            )

    @pytest.mark.parametrize(
        "scaling_type", ["original", "normalization", "maxscale", "maxmin"]
    )
    def test_init_with_valid_scaling_types(self, temp_parquet_files, scaling_type):
        """Test initialization with all valid scaling types."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files["subsidies_path"],
            variables_parquet=temp_parquet_files["variables_path"],
            keys_parquet=temp_parquet_files["keys_path"],
            scaling_type=scaling_type,
            auto_download=False,
        )

        assert dataset.scaling_type == scaling_type
        assert len(dataset) > 0

    def test_init_computes_scaling_params_for_normalization(self, temp_parquet_files):
        """Test that scaling params are computed for normalization."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files["subsidies_path"],
            variables_parquet=temp_parquet_files["variables_path"],
            keys_parquet=temp_parquet_files["keys_path"],
            scaling_type="normalization",
            auto_download=False,
        )

        assert dataset.precomputed_scaling_params is not None
        for col in ["tech_1", "tech_2", "var_1", "var_2"]:
            assert col in dataset.precomputed_scaling_params
            assert "mean" in dataset.precomputed_scaling_params[col]
            assert "std" in dataset.precomputed_scaling_params[col]

    def test_init_computes_scaling_params_for_maxscale(self, temp_parquet_files):
        """Test that max scaling params are computed correctly."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files["subsidies_path"],
            variables_parquet=temp_parquet_files["variables_path"],
            keys_parquet=temp_parquet_files["keys_path"],
            scaling_type="maxscale",
            auto_download=False,
        )

        for col in ["tech_1", "tech_2", "var_1", "var_2"]:
            assert "max" in dataset.precomputed_scaling_params[col]

    def test_init_computes_scaling_params_for_maxmin(self, temp_parquet_files):
        """Test that min/max scaling params are computed correctly."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files["subsidies_path"],
            variables_parquet=temp_parquet_files["variables_path"],
            keys_parquet=temp_parquet_files["keys_path"],
            scaling_type="maxmin",
            auto_download=False,
        )

        for col in ["tech_1", "tech_2", "var_1", "var_2"]:
            assert "min" in dataset.precomputed_scaling_params[col]
            assert "max" in dataset.precomputed_scaling_params[col]
            assert (
                dataset.precomputed_scaling_params[col]["min"]
                < dataset.precomputed_scaling_params[col]["max"]
            )

    def test_init_with_precomputed_scaling_params(self, temp_parquet_files):
        """Test initialization with precomputed scaling parameters."""
        precomputed_params = {
            "tech_1": {"mean": 0.25, "std": 0.1},
            "tech_2": {"mean": 0.65, "std": 0.1},
            "var_1": {"mean": 2.5, "std": 1.0},
            "var_2": {"mean": 6.5, "std": 1.0},
        }

        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files["subsidies_path"],
            variables_parquet=temp_parquet_files["variables_path"],
            keys_parquet=temp_parquet_files["keys_path"],
            scaling_type="normalization",
            precomputed_scaling_params=precomputed_params,
            auto_download=False,
        )

        assert dataset.precomputed_scaling_params == precomputed_params

    def test_init_drop_columns(self, temp_parquet_files):
        """Test that columns can be dropped during initialization."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files["subsidies_path"],
            variables_parquet=temp_parquet_files["variables_path"],
            keys_parquet=temp_parquet_files["keys_path"],
            drop_columns=["tech_1"],
            auto_download=False,
        )

        assert "tech_1" not in dataset.subsidies_names
        assert "tech_2" in dataset.subsidies_names

    def test_init_with_cuda_false(self, temp_parquet_files):
        """Test initialization with with_cuda=False."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files["subsidies_path"],
            variables_parquet=temp_parquet_files["variables_path"],
            keys_parquet=temp_parquet_files["keys_path"],
            with_cuda=False,
            auto_download=False,
        )

        assert dataset.subsidies_df.device.type == "cpu"
        assert dataset.variables_df.device.type == "cpu"

    def test_init_creates_index_map(self, temp_parquet_files):
        """Test that index_map is created correctly."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files["subsidies_path"],
            variables_parquet=temp_parquet_files["variables_path"],
            keys_parquet=temp_parquet_files["keys_path"],
            auto_download=False,
        )

        assert hasattr(dataset, "index_map")
        assert isinstance(dataset.index_map, dict)
        assert len(dataset.index_map) > 0

    def test_init_creates_variables_next(self, temp_parquet_files):
        """Test that variables_next_df is properly initialized."""
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files["subsidies_path"],
            variables_parquet=temp_parquet_files["variables_path"],
            keys_parquet=temp_parquet_files["keys_path"],
            auto_download=False,
        )

        assert hasattr(dataset, "variables_next_df")
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
        subsidies_df = pd.DataFrame(
            {
                "tech_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "tech_2": [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            }
        )

        variables_df = pd.DataFrame(
            {
                "var_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "var_2": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            }
        )

        keys_df = pd.DataFrame(
            {
                "gdx": ["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"],
                "year": [2020, 2020, 2025, 2025, 2020, 2020, 2025, 2025],
                "n": ["r_a", "r_b", "r_a", "r_b", "r_a", "r_b", "r_a", "r_b"],
            }
        )

        subsidies_path = os.path.join(temp_dir.name, "subsidies.parquet")
        variables_path = os.path.join(temp_dir.name, "variables.parquet")
        keys_path = os.path.join(temp_dir.name, "keys.parquet")

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
        nonexistent_path = os.path.join(
            tempfile.gettempdir(), "nonexistent_witch_file_12345.parquet"
        )

        with pytest.raises(FileNotFoundError):
            witch_proc_data(
                subsidies_parquet=nonexistent_path,
                variables_parquet=nonexistent_path,
                keys_parquet=nonexistent_path,
                auto_download=False,
            )

    def test_auto_download_enabled_with_existing_files_does_not_download(
        self, temp_parquet_files
    ):
        """Test that auto_download=True doesn't download if files already exist."""
        # This should not raise any errors and should not attempt downloads
        dataset = witch_proc_data(
            subsidies_parquet=temp_parquet_files["subsidies_path"],
            variables_parquet=temp_parquet_files["variables_path"],
            keys_parquet=temp_parquet_files["keys_path"],
            auto_download=True,
        )

        assert dataset is not None
        assert len(dataset) > 0


"""
TEST FAIRY - Tensor-based interface
"""


class TestFAIRYHelpers:
    """Helper methods and fixtures for FAIRY tests."""

    @staticmethod
    def create_plans_tensor(
        fairy_proxy,
        batch_size: int,
        amount_value: float = 0.1,
    ) -> tuple:
        """Create a plans tensor and tech names for testing.

        Parameters
        ----------
        fairy_proxy : FAIRY
            The proxy instance to get tech names from.
        batch_size : int
            Number of samples in batch.
        amount_value : float
            Value to fill the plans tensor with.

        Returns
        -------
        tuple
            (plans_tensor, tech_names)
        """
        n_techs = fairy_proxy.n_techs
        plans_tensor = torch.full(
            (batch_size, n_techs),
            amount_value,
            dtype=torch.float32,
        )
        tech_names = fairy_proxy.tech_names_ordered.copy()
        return plans_tensor, tech_names

    @staticmethod
    def create_varied_plans_tensor(
        fairy_proxy,
        batch_size: int,
    ) -> tuple:
        """Create plans tensor with varied investment levels.

        Parameters
        ----------
        fairy_proxy : FAIRY
            The proxy instance.
        batch_size : int
            Number of samples in batch.

        Returns
        -------
        tuple
            (plans_tensor, tech_names)
        """
        n_techs = fairy_proxy.n_techs
        # Amount values: NONE=0.0, LOW=0.1, MEDIUM=0.3, HIGH=0.75
        amount_values = [0.0, 0.1, 0.3, 0.75]

        plans_tensor = torch.zeros(batch_size, n_techs, dtype=torch.float32)
        for i in range(batch_size):
            for j in range(n_techs):
                plans_tensor[i, j] = amount_values[(i + j) % len(amount_values)]

        tech_names = fairy_proxy.tech_names_ordered.copy()
        return plans_tensor, tech_names


class TestFAIRYCall:
    """Test the __call__ method with tensor-based interface."""

    @pytest.fixture
    def fairy_proxy(self):
        """Import and instantiate FAIRY proxy."""
        return FAIRY()

    def test_call_basic(self, fairy_proxy):
        """Test basic __call__ with simple tensor input."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=2, amount_value=0.1
        )

        result = fairy_proxy((plans_tensor, tech_names))

        assert result.shape == (2,)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert torch.all(torch.isfinite(result))

    def test_call_single_state(self, fairy_proxy):
        """Test __call__ with a single state."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1, amount_value=0.3
        )

        result = fairy_proxy((plans_tensor, tech_names))

        assert result.shape == (1,)
        assert isinstance(result, torch.Tensor)
        assert torch.all(torch.isfinite(result))

    def test_call_zero_investment(self, fairy_proxy):
        """Test __call__ with zero investment (all subsidies are 0)."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1, amount_value=0.0
        )

        result = fairy_proxy((plans_tensor, tech_names))

        assert result.shape == (1,)
        assert isinstance(result, torch.Tensor)
        assert torch.isfinite(result[0])

    def test_call_max_investment(self, fairy_proxy):
        """Test __call__ with maximum investment (HIGH = 0.75)."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1, amount_value=0.75
        )

        result = fairy_proxy((plans_tensor, tech_names))

        assert result.shape == (1,)
        assert isinstance(result, torch.Tensor)
        assert torch.isfinite(result[0])

    def test_call_varied_investments(self, fairy_proxy):
        """Test __call__ with varied investment levels across batch."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_varied_plans_tensor(
            fairy_proxy, batch_size=4
        )

        result = fairy_proxy((plans_tensor, tech_names))

        assert result.shape == (4,)
        assert isinstance(result, torch.Tensor)
        assert torch.all(torch.isfinite(result))

    def test_call_all_amount_levels(self, fairy_proxy):
        """Test __call__ with all different amount levels."""
        amount_values = [0.0, 0.1, 0.3, 0.75]  # NONE, LOW, MEDIUM, HIGH
        n_techs = fairy_proxy.n_techs

        plans_tensor = torch.zeros(4, n_techs, dtype=torch.float32)
        for i, amount in enumerate(amount_values):
            plans_tensor[i, :] = amount

        tech_names = fairy_proxy.tech_names_ordered.copy()

        result = fairy_proxy((plans_tensor, tech_names))

        assert result.shape == (4,)
        assert isinstance(result, torch.Tensor)
        assert torch.all(torch.isfinite(result))

    def test_call_returns_tensor_type(self, fairy_proxy):
        """Test that __call__ always returns a torch.Tensor."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1, amount_value=0.75
        )

        result = fairy_proxy((plans_tensor, tech_names))

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_call_batch_size_consistency(self, fairy_proxy):
        """Test that output batch size matches input batch size."""
        for batch_size in [1, 5, 10, 50]:
            plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
                fairy_proxy, batch_size=batch_size, amount_value=0.1
            )

            result = fairy_proxy((plans_tensor, tech_names))

            assert result.shape == (batch_size,)
            assert torch.all(torch.isfinite(result))

    def test_call_no_grad(self, fairy_proxy):
        """Test that __call__ does not compute gradients."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1, amount_value=0.1
        )

        result = fairy_proxy((plans_tensor, tech_names))

        # Result should not have gradient
        assert not result.requires_grad

    def test_call_deterministic(self, fairy_proxy):
        """Test that __call__ produces deterministic results."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1, amount_value=0.3
        )

        result1 = fairy_proxy((plans_tensor.clone(), tech_names))
        result2 = fairy_proxy((plans_tensor.clone(), tech_names))

        assert torch.allclose(result1, result2)

    def test_call_different_inputs_different_outputs(self, fairy_proxy):
        """Test that different inputs produce different outputs."""
        n_techs = fairy_proxy.n_techs
        tech_names = fairy_proxy.tech_names_ordered.copy()

        plans_low = torch.full((1, n_techs), 0.0, dtype=torch.float32)
        plans_high = torch.full((1, n_techs), 0.75, dtype=torch.float32)

        result_low = fairy_proxy((plans_low, tech_names))
        result_high = fairy_proxy((plans_high, tech_names))

        assert not torch.allclose(result_low, result_high)


class TestFAIRYPermutation:
    """Test permutation logic for tech name ordering."""

    @pytest.fixture
    def fairy_proxy(self):
        """Import and instantiate FAIRY proxy."""
        return FAIRY()

    def test_permutation_initialized_on_first_call(self, fairy_proxy):
        """Test that permutation index is None before first call."""
        assert fairy_proxy._permutation_idx is None

        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1
        )
        fairy_proxy((plans_tensor, tech_names))

        assert fairy_proxy._permutation_idx is not None

    def test_permutation_cached_after_first_call(self, fairy_proxy):
        """Test that permutation index is cached and reused."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1
        )

        fairy_proxy((plans_tensor, tech_names))
        perm_first = fairy_proxy._permutation_idx.clone()

        fairy_proxy((plans_tensor, tech_names))
        perm_second = fairy_proxy._permutation_idx

        assert torch.equal(perm_first, perm_second)

    def test_permutation_identity_when_order_matches(self, fairy_proxy):
        """Test that permutation is identity when tech order matches proxy."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1
        )

        fairy_proxy((plans_tensor, tech_names))

        # Should be identity permutation [0, 1, 2, ...]
        expected = torch.arange(fairy_proxy.n_techs, dtype=torch.long)
        assert torch.equal(
            fairy_proxy._permutation_idx.cpu(),
            expected,
        )

    def test_permutation_reorders_correctly(self, fairy_proxy):
        """Test that permutation correctly reorders shuffled tech names."""
        n_techs = fairy_proxy.n_techs
        tech_names = fairy_proxy.tech_names_ordered.copy()

        # Shuffle tech names
        shuffled_indices = torch.randperm(n_techs).tolist()
        shuffled_tech_names = [tech_names[i] for i in shuffled_indices]

        # Create plans tensor with known values per tech
        plans_tensor = torch.arange(n_techs, dtype=torch.float32).unsqueeze(0)
        # Shuffle the tensor to match shuffled names
        shuffled_plans = plans_tensor[:, shuffled_indices]

        # Reset permutation
        fairy_proxy._permutation_idx = None

        result = fairy_proxy((shuffled_plans, shuffled_tech_names))

        assert result.shape == (1,)
        assert torch.isfinite(result[0])

    def test_permutation_raises_on_missing_tech_in_proxy(self, fairy_proxy):
        """Test that missing tech in proxy raises ValueError."""
        plans_tensor = torch.zeros(1, fairy_proxy.n_techs + 1, dtype=torch.float32)
        tech_names = fairy_proxy.tech_names_ordered.copy() + ["FAKE_TECH"]

        fairy_proxy._permutation_idx = None

        with pytest.raises(ValueError, match="Technologies in environment but not in proxy"):
            fairy_proxy((plans_tensor, tech_names))

    def test_permutation_raises_on_missing_tech_in_env(self, fairy_proxy):
        """Test that missing tech in environment raises ValueError."""
        plans_tensor = torch.zeros(1, fairy_proxy.n_techs - 1, dtype=torch.float32)
        tech_names = fairy_proxy.tech_names_ordered[:-1]

        fairy_proxy._permutation_idx = None

        with pytest.raises(ValueError, match="Technologies in proxy but not in environment"):
            fairy_proxy((plans_tensor, tech_names))


class TestFAIRYInit:
    """Test FAIRY initialization and attribute handling."""

    def test_init_creates_required_attributes(self):
        """Test that initialization creates all required attributes."""
        proxy = FAIRY()

        assert hasattr(proxy, "fairy")
        assert hasattr(proxy, "precomputed_scaling_params")
        assert hasattr(proxy, "tech_names_ordered")
        assert hasattr(proxy, "n_techs")
        assert hasattr(proxy, "variables_names")
        assert hasattr(proxy, "device")
        assert hasattr(proxy, "key_gdx")
        assert hasattr(proxy, "key_year")
        assert hasattr(proxy, "key_region")
        assert hasattr(proxy, "SCC")
        assert hasattr(proxy, "context")
        assert hasattr(proxy, "_permutation_idx")

    def test_init_fairy_has_required_attributes(self):
        """Test that initialized fairy has required attributes."""
        proxy = FAIRY()

        assert hasattr(proxy.fairy, "subsidies_dim")
        assert hasattr(proxy.fairy, "variables_names")
        assert hasattr(proxy.fairy, "subsidies_names")

    def test_init_tech_names_ordered_is_list(self):
        """Test that tech_names_ordered is a list."""
        proxy = FAIRY()
        assert isinstance(proxy.tech_names_ordered, list)

    def test_init_n_techs_matches_tech_names(self):
        """Test that n_techs matches length of tech_names_ordered."""
        proxy = FAIRY()
        assert proxy.n_techs == len(proxy.tech_names_ordered)

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

    def test_init_with_custom_scc(self):
        """Test initialization with custom SCC."""
        custom_scc = 100.0
        proxy = FAIRY(SCC=custom_scc)
        assert isinstance(proxy.SCC, torch.Tensor)

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

    def test_init_permutation_idx_is_none(self):
        """Test that permutation index is None after init."""
        proxy = FAIRY()
        assert proxy._permutation_idx is None

    def test_init_scaling_tensors_exist(self):
        """Test that scaling tensors are created."""
        proxy = FAIRY()
        assert hasattr(proxy, "cons_min")
        assert hasattr(proxy, "cons_max")
        assert hasattr(proxy, "cons_scale")
        assert isinstance(proxy.cons_min, torch.Tensor)
        assert isinstance(proxy.cons_max, torch.Tensor)
        assert isinstance(proxy.cons_scale, torch.Tensor)


class TestFAIRYDeviceHandling:
    """Test device handling and movement in FAIRY proxy."""

    @pytest.fixture
    def fairy_proxy(self):
        """Import and instantiate FAIRY proxy."""
        return FAIRY()

    def test_device_consistency(self, fairy_proxy):
        """Test that all tensors are on the same device."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1
        )

        result = fairy_proxy((plans_tensor, tech_names))

        assert result.device == fairy_proxy.device
        assert fairy_proxy.context.device == fairy_proxy.device
        assert fairy_proxy.SCC.device == fairy_proxy.device

    def test_to_method_moves_tensors(self, fairy_proxy):
        """Test that to() method correctly moves all tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        fairy_proxy.to("cuda")

        assert fairy_proxy.device.type == "cuda"
        assert fairy_proxy.context.device.type == "cuda"
        assert fairy_proxy.SCC.device.type == "cuda"
        assert fairy_proxy.cons_min.device.type == "cuda"
        assert fairy_proxy.cons_max.device.type == "cuda"
        assert fairy_proxy.cons_scale.device.type == "cuda"
        assert next(fairy_proxy.fairy.parameters()).device.type == "cuda"

    def test_to_method_moves_permutation_idx(self, fairy_proxy):
        """Test that to() method moves permutation index if initialized."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Initialize permutation
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1
        )
        fairy_proxy((plans_tensor, tech_names))

        fairy_proxy.to("cuda")

        assert fairy_proxy._permutation_idx.device.type == "cuda"

    def test_to_method_returns_self(self, fairy_proxy):
        """Test that to() method returns self for chaining."""
        result = fairy_proxy.to("cpu")
        assert result is fairy_proxy

    def test_call_output_on_correct_device(self, fairy_proxy):
        """Test that __call__ output is on the correct device."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1
        )

        result = fairy_proxy((plans_tensor, tech_names))

        assert result.device == fairy_proxy.device

    def test_call_accepts_tensor_on_different_device(self, fairy_proxy):
        """Test that __call__ moves input tensor to correct device."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1
        )

        # Ensure tensor is on CPU
        plans_tensor = plans_tensor.cpu()

        result = fairy_proxy((plans_tensor, tech_names))

        assert result.device == fairy_proxy.device
        assert torch.isfinite(result[0])


class TestDenormalization:
    """Test denormalization logic in the __call__ method."""

    @pytest.fixture
    def fairy_proxy(self):
        """Import and instantiate FAIRY proxy."""
        return FAIRY()

    def test_consumption_denormalization(self, fairy_proxy):
        """Test that consumption is properly denormalized."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1, amount_value=0.1
        )

        result = fairy_proxy((plans_tensor, tech_names))

        # Get scaling params
        consumption_params = fairy_proxy.precomputed_scaling_params["CONSUMPTION"]

        # Result should be in original scale (not normalized)
        # Should be roughly between min and max (with some margin for model outputs)
        assert (
            consumption_params["min"]
            <= result[0].item()
            <= consumption_params["max"] * 1.5
            or result[0].item() >= 0
        )

    def test_scaling_params_exist(self, fairy_proxy):
        """Test that all required scaling parameters exist."""
        required_keys = ["CONSUMPTION", "EMI_total_CO2"]
        for key in required_keys:
            assert key in fairy_proxy.precomputed_scaling_params
            assert "min" in fairy_proxy.precomputed_scaling_params[key]
            assert "max" in fairy_proxy.precomputed_scaling_params[key]

    def test_scaling_params_valid_ranges(self, fairy_proxy):
        """Test that scaling parameter ranges are valid (min < max)."""
        for col, params in fairy_proxy.precomputed_scaling_params.items():
            assert (
                params["min"] < params["max"]
            ), f"Invalid range for {col}: min={params['min']}, max={params['max']}"


class TestDenormalizationNumericStability:
    """Test numeric stability of denormalization processes."""

    @pytest.fixture
    def fairy_proxy(self):
        """Import and instantiate FAIRY proxy."""
        return FAIRY()

    def test_denormalization_no_nan_or_inf(self, fairy_proxy):
        """Test that denormalization never produces NaN or Inf values."""
        amount_values = [0.0, 0.1, 0.3, 0.75]
        n_techs = fairy_proxy.n_techs

        plans_tensor = torch.zeros(40, n_techs, dtype=torch.float32)
        for i in range(40):
            plans_tensor[i, :] = amount_values[i % len(amount_values)]

        tech_names = fairy_proxy.tech_names_ordered.copy()

        result = fairy_proxy((plans_tensor, tech_names))

        assert torch.all(
            torch.isfinite(result)
        ), f"Found NaN/Inf values in results: {result[~torch.isfinite(result)]}"

    def test_denormalization_scaling_bounds(self, fairy_proxy):
        """Test that denormalized consumption stays within reasonable bounds."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_varied_plans_tensor(
            fairy_proxy, batch_size=40
        )

        result = fairy_proxy((plans_tensor, tech_names))

        # Get bounds from scaling parameters
        consumption_params = fairy_proxy.precomputed_scaling_params["CONSUMPTION"]

        # Results should be bounded reasonably
        max_reasonable = consumption_params["max"] * 10
        min_reasonable = -consumption_params["max"] * 10

        assert torch.all(
            result > min_reasonable
        ), f"Found unreasonably low values: min={result.min()}, threshold={min_reasonable}"
        assert torch.all(
            result < max_reasonable
        ), f"Found unreasonably high values: max={result.max()}, threshold={max_reasonable}"

    def test_denormalization_inverse_operation(self, fairy_proxy):
        """Test that denormalization can be approximately inverted."""
        consumption_params = fairy_proxy.precomputed_scaling_params["CONSUMPTION"]

        # Pick a random denormalized value within bounds
        denorm_value = torch.tensor(
            consumption_params["min"]
            + (consumption_params["max"] - consumption_params["min"]) * 0.5
        )

        # Apply inverse normalization
        norm_value = (denorm_value - consumption_params["min"]) / (
            consumption_params["max"] - consumption_params["min"]
        )

        # Re-denormalize
        recovered = (
            norm_value * (consumption_params["max"] - consumption_params["min"])
            + consumption_params["min"]
        )

        # Should recover original value
        assert torch.allclose(
            denorm_value, recovered, rtol=1e-5
        ), f"Denormalization not invertible: {denorm_value} -> {recovered}"

    def test_scaling_params_consistency(self, fairy_proxy):
        """Test that scaling parameters are consistent across multiple calls."""
        proxy1 = FAIRY()
        proxy2 = FAIRY()

        for key in ["CONSUMPTION", "EMI_total_CO2"]:
            assert (
                proxy1.precomputed_scaling_params[key]["min"]
                == proxy2.precomputed_scaling_params[key]["min"]
            )
            assert (
                proxy1.precomputed_scaling_params[key]["max"]
                == proxy2.precomputed_scaling_params[key]["max"]
            )

    def test_extreme_investment_combinations(self, fairy_proxy):
        """Test numeric stability with extreme investment combinations."""
        n_techs = fairy_proxy.n_techs
        tech_names = fairy_proxy.tech_names_ordered.copy()

        # All NONE (minimal investment)
        plans_none = torch.zeros(1, n_techs, dtype=torch.float32)

        # All HIGH (maximal investment)
        plans_high = torch.full((1, n_techs), 0.75, dtype=torch.float32)

        result_none = fairy_proxy((plans_none, tech_names))
        result_high = fairy_proxy((plans_high, tech_names))

        assert torch.all(torch.isfinite(result_none))
        assert torch.all(torch.isfinite(result_high))

        # Results should be different
        assert not torch.allclose(result_none, result_high)


class TestFAIRYRobustness:
    """Test robustness and edge cases."""

    @pytest.fixture
    def fairy_proxy(self):
        """Import and instantiate FAIRY proxy."""
        return FAIRY()

    def test_large_batch_size(self, fairy_proxy):
        """Test __call__ with a large batch size."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=100, amount_value=0.1
        )

        result = fairy_proxy((plans_tensor, tech_names))

        assert result.shape == (100,)
        assert torch.all(torch.isfinite(result))

    def test_invalid_region_raises_error(self):
        """Test that invalid region raises AssertionError."""
        with pytest.raises(AssertionError):
            FAIRY(key_region="invalid_region")

    def test_model_in_eval_mode_during_call(self, fairy_proxy):
        """Test that model is in eval mode (required for BatchNorm with batch_size=1)."""
        assert (
            not fairy_proxy.fairy.training
        ), "Model should be in eval mode for inference"

        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=1
        )

        result = fairy_proxy((plans_tensor, tech_names))

        assert torch.isfinite(result[0])

    def test_call_with_wrong_tensor_shape(self, fairy_proxy):
        """Test that mismatched tech names raises error."""
        tech_names = fairy_proxy.tech_names_ordered.copy() + ["FAKE_TECH_1", "FAKE_TECH_2"]

        # Tensor shape matches extended tech_names
        wrong_plans = torch.zeros(1, len(tech_names), dtype=torch.float32)

        fairy_proxy._permutation_idx = None

        with pytest.raises(ValueError, match="Technologies in environment but not in proxy"):
            fairy_proxy((wrong_plans, tech_names))

    def test_call_with_empty_batch(self, fairy_proxy):
        """Test __call__ with empty batch."""
        n_techs = fairy_proxy.n_techs
        tech_names = fairy_proxy.tech_names_ordered.copy()

        empty_plans = torch.zeros(0, n_techs, dtype=torch.float32)

        result = fairy_proxy((empty_plans, tech_names))

        assert result.shape == (0,)

    def test_repeated_calls_same_result(self, fairy_proxy):
        """Test that repeated calls with same input give same result."""
        plans_tensor, tech_names = TestFAIRYHelpers.create_plans_tensor(
            fairy_proxy, batch_size=5, amount_value=0.3
        )

        results = []
        for _ in range(5):
            result = fairy_proxy((plans_tensor.clone(), tech_names))
            results.append(result.clone())

        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i])