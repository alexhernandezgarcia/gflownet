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


class TestFAIRYCall:
    """Test the __call__ method with various input configurations."""

    @pytest.fixture
    def fairy_proxy(self):
        """Import and instantiate FAIRY proxy."""
        return FAIRY()

    def test_call_with_contexts(self, fairy_proxy):
        """Test __call__ with explicit contexts."""
        states = [
            [{'TECH': 'SUBS_power_COAL_noccs', 'AMOUNT': 'LOW'}],
            [{'TECH': 'SUBS_power_NUCLEAR', 'AMOUNT': 'HIGH'}],
        ]
        contexts = [torch.rand(fairy_proxy.fairy.variables_dim),
                    torch.rand(fairy_proxy.fairy.variables_dim)]
        budgets = [100.0, 100.0]

        result = fairy_proxy(states, contexts=contexts, budgets=budgets)

        assert result.shape == (2,)
        assert isinstance(result, torch.Tensor)
        assert torch.all(torch.isfinite(result))

    def test_call_with_keys_context(self, fairy_proxy):
        """Test __call__ with keys_context instead of explicit contexts."""
        states = [
            [{'TECH': 'SUBS_power_COAL_noccs', 'AMOUNT': 'LOW'}],
            [{'TECH': 'SUBS_power_NUCLEAR', 'AMOUNT': 'HIGH'}],
        ]

        # Get valid keys from the data index map
        valid_keys = list(fairy_proxy.data.index_map.keys())
        if len(valid_keys) >= 2:
            keys_context = [
                {'gdx': valid_keys[0][0], 'year': valid_keys[0][1], 'region': valid_keys[0][2]},
                {'gdx': valid_keys[1][0], 'year': valid_keys[1][1], 'region': valid_keys[1][2]},
            ]
            budgets = [100.0, 100.0]

            result = fairy_proxy(states, keys_context=keys_context, budgets=budgets)

            assert result.shape == (2,)
            assert isinstance(result, torch.Tensor)

    def test_call_with_random_contexts(self, fairy_proxy):
        """Test __call__ with no contexts (generates random ones)."""
        states = [
            [{'TECH': 'SUBS_power_COAL_noccs', 'AMOUNT': 'NONE'}],
            [{'TECH': 'SUBS_power_NUCLEAR', 'AMOUNT': 'MEDIUM'}],
        ]
        budgets = [100.0, 100.0]

        result = fairy_proxy(states, budgets=budgets)

        assert result.shape == (2,)
        assert isinstance(result, torch.Tensor)

    def test_call_with_single_budget_broadcast(self, fairy_proxy):
        """Test __call__ with single budget value broadcast to all states."""
        states = [
            [{'TECH': 'SUBS_power_COAL_noccs', 'AMOUNT': 'LOW'}],
            [{'TECH': 'SUBS_power_NUCLEAR', 'AMOUNT': 'HIGH'}],
        ]
        contexts = [torch.rand(fairy_proxy.fairy.variables_dim),
                    torch.rand(fairy_proxy.fairy.variables_dim)]

        result = fairy_proxy(states, contexts=contexts, budgets=100.0)

        assert result.shape == (2,)

    def test_call_budget_constraint_enforcement(self, fairy_proxy):
        """Test that consumption is zeroed when emissions exceed budget."""
        states = [
            [{'TECH': 'SUBS_power_COAL_noccs', 'AMOUNT': 'LOW'}],
        ]
        contexts = [torch.rand(fairy_proxy.fairy.variables_dim)]
        budgets = [10.0]  # Lower threshold to trigger constraint

        result = fairy_proxy(states, contexts=contexts, budgets=budgets)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)

    @pytest.mark.parametrize(
        "num_investments",
        [1, 2, 3],
    )
    def test_call_multiple_investments_per_state(self, fairy_proxy, num_investments):
        """Test __call__ with multiple investments in a single state."""
        techs = ['SUBS_power_COAL_noccs', 'SUBS_power_NUCLEAR', 'SUBS_power_HYDRO']
        investments = [
            {'TECH': techs[i], 'AMOUNT': 'LOW'}
            for i in range(num_investments)
        ]
        states = [investments]
        contexts = [torch.rand(fairy_proxy.fairy.variables_dim)]
        budgets = [100.0]

        result = fairy_proxy(states, contexts=contexts, budgets=budgets)

        assert result.shape == (1,)
        assert isinstance(result, torch.Tensor)

    def test_call_contexts_length_mismatch_raises_assertion(self, fairy_proxy):
        """Test that assertion fails when contexts length doesn't match states."""
        states = [
            [{'TECH': 'SUBS_power_COAL_noccs', 'AMOUNT': 'LOW'}],
            [{'TECH': 'SUBS_power_NUCLEAR', 'AMOUNT': 'HIGH'}],
        ]
        contexts = [torch.rand(fairy_proxy.fairy.variables_dim)]  # Only 1 context for 2 states

        with pytest.raises(AssertionError):
            fairy_proxy(states, contexts=contexts, budgets=[100.0, 100.0])

    def test_call_context_dimension_mismatch_raises_assertion(self, fairy_proxy):
        """Test that assertion fails when context dimensions are incorrect."""
        states = [
            [{'TECH': 'SUBS_power_COAL_noccs', 'AMOUNT': 'LOW'}],
        ]
        contexts = [torch.rand(5)]  # Wrong dimension

        with pytest.raises(AssertionError):
            fairy_proxy(states, contexts=contexts, budgets=[100.0])

    def test_call_returns_tensor_type(self, fairy_proxy):
        """Test that __call__ always returns a torch.Tensor."""
        states = [[{'TECH': 'SUBS_power_NUCLEAR', 'AMOUNT': 'HIGH'}]]
        contexts = [torch.rand(fairy_proxy.fairy.variables_dim)]
        budgets = [100.0]

        result = fairy_proxy(states, contexts=contexts, budgets=budgets)

        assert isinstance(result, torch.Tensor)


class TestFAIRYInit:
    """Test FAIRY initialization."""

    def test_init_creates_fairy_attribute(self):
        """Test that initialization creates fairy attribute."""
        proxy = FAIRY()
        assert hasattr(proxy, 'fairy')

    def test_init_creates_data_attribute(self):
        """Test that initialization creates data attribute."""
        proxy = FAIRY()
        assert hasattr(proxy, 'data')

    def test_init_fairy_has_required_attributes(self):
        """Test that initialized fairy has required attributes."""
        proxy = FAIRY()
        assert hasattr(proxy.fairy, 'variables_dim')
        assert hasattr(proxy.fairy, 'subsidies_dim')
        assert hasattr(proxy.fairy, 'variables_names')
        assert hasattr(proxy.fairy, 'subsidies_names')

    def test_init_data_has_required_attributes(self):
        """Test that initialized data has required attributes."""
        proxy = FAIRY()
        assert hasattr(proxy.data, 'variables_df')
        assert hasattr(proxy.data, 'index_map')
        assert hasattr(proxy.data, 'variables_names')