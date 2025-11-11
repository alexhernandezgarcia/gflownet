import pytest
import torch
from typing import List, Dict

from gflownet.proxy.iam.iam_proxies import FAIRY


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
        """Test that 'MEDIUM' returns 0.5"""
        assert fairy_proxy.get_invested_amount('MEDIUM') == 0.5

    def test_get_invested_amount_high(self, fairy_proxy):
        """Test that 'HIGH' returns 1.0"""
        assert fairy_proxy.get_invested_amount('HIGH') == 1.0

    def test_get_invested_amount_invalid(self, fairy_proxy):
        """Test that invalid amount raises ValueError"""
        with pytest.raises(ValueError, match="Invalid amount"):
            fairy_proxy.get_invested_amount('INVALID')


class TestFAIRYCall:
    """Test the __call__ method with various input configurations."""

    @pytest.fixture
    def fairy_proxy(self):
        """Import and instantiate FAIRY proxy."""
        return FAIRY()

    def test_call_basic(self, fairy_proxy):
        """Test basic __call__ with simple states."""
        states = [
            [{'TECH': 'SUBS_power_COAL_noccs', 'AMOUNT': 'LOW'}],
            [{'TECH': 'SUBS_power_NUCLEAR', 'AMOUNT': 'HIGH'}],
        ]

        result = fairy_proxy(states)

        assert result.shape == (2,)
        assert isinstance(result, torch.Tensor)
        assert torch.all(torch.isfinite(result))

    def test_call_single_state(self, fairy_proxy):
        """Test __call__ with a single state."""
        states = [
            [{'TECH': 'SUBS_power_COAL_noccs', 'AMOUNT': 'MEDIUM'}],
        ]

        result = fairy_proxy(states)

        assert result.shape == (1,)
        assert isinstance(result, torch.Tensor)
        assert torch.all(torch.isfinite(result))

    def test_call_empty_state(self, fairy_proxy):
        """Test __call__ with empty investment (no subsidies)."""
        states = [
            [],  # Empty plan
        ]

        result = fairy_proxy(states)

        assert result.shape == (1,)
        assert isinstance(result, torch.Tensor)

    def test_call_none_amount(self, fairy_proxy):
        """Test __call__ with NONE investment amount."""
        states = [
            [{'TECH': 'SUBS_power_COAL_noccs', 'AMOUNT': 'NONE'}],
            [{'TECH': 'SUBS_power_NUCLEAR', 'AMOUNT': 'MEDIUM'}],
        ]

        result = fairy_proxy(states)

        assert result.shape == (2,)
        assert isinstance(result, torch.Tensor)

    @pytest.mark.parametrize(
        "num_investments",
        [1, 2, 3],
    )
    def test_call_multiple_investments_per_state(self, fairy_proxy, num_investments):
        """Test __call__ with multiple investments in a single state."""
        # Use only valid tech names from the proxy
        available_techs = fairy_proxy.subsidies_names[:num_investments]

        investments = [
            {'TECH': tech, 'AMOUNT': 'LOW'}
            for tech in available_techs
        ]
        states = [investments]

        result = fairy_proxy(states)

        assert result.shape == (1,)
        assert isinstance(result, torch.Tensor)

    def test_call_multiple_states_multiple_investments(self, fairy_proxy):
        """Test __call__ with multiple states, each having multiple investments."""
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
        # Results should generally increase with investment amount (though not guaranteed)
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

    def test_call_invalid_tech_raises_error(self, fairy_proxy):
        """Test that invalid tech name raises an error."""
        states = [
            [{'TECH': 'INVALID_TECH_NAME', 'AMOUNT': 'LOW'}],
        ]

        with pytest.raises(ValueError):
            fairy_proxy(states)

    def test_call_same_tech_multiple_times(self, fairy_proxy):
        """Test behavior when same tech appears multiple times in a state."""
        tech = fairy_proxy.subsidies_names[0]

        states = [
            [
                {'TECH': tech, 'AMOUNT': 'LOW'},
                {'TECH': tech, 'AMOUNT': 'MEDIUM'},  # Same tech again
            ],
        ]

        # This should overwrite the first value
        result = fairy_proxy(states)

        assert result.shape == (1,)
        assert isinstance(result, torch.Tensor)


class TestFAIRYInit:
    """Test FAIRY initialization."""

    def test_init_creates_fairy_attribute(self):
        """Test that initialization creates fairy attribute."""
        proxy = FAIRY()
        assert hasattr(proxy, 'fairy')

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
        assert hasattr(proxy, 'budget')
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
        custom_budget = 500.0
        proxy = FAIRY(budget=custom_budget)
        assert torch.allclose(proxy.budget, torch.tensor(custom_budget))

    def test_init_with_custom_scc(self):
        """Test initialization with custom SCC."""
        custom_scc = 100.0
        proxy = FAIRY(SCC=custom_scc)
        assert proxy.SCC == custom_scc

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

    def test_init_budget_is_tensor(self):
        """Test that budget is converted to tensor."""
        proxy = FAIRY()
        assert isinstance(proxy.budget, torch.Tensor)


class TestFAIRYDeviceHandling:
    """Test device handling in FAIRY proxy."""

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
        assert fairy_proxy.budget.device == fairy_proxy.device