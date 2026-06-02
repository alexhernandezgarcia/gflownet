"""
Node of a decision tree with a discrete threshold.

A DecisionTreeNodeDiscrete represents a single decision node consisting of two
stages:
1. Choosing a feature (via a Choice sub-environment).
2. Choosing a threshold from a finite, equally-spaced grid of values
   (via a 1D Grid sub-environment).

The environment is built as a Stack of these two sub-environments.

This is a drop-in alternative to DecisionTreeNode where the threshold is taken
from ``n_thresholds`` evenly-spaced values in ``[cell_min, cell_max]`` rather
than from a continuous interval.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from gflownet.envs.choice import Choice
from gflownet.envs.composite.stack import Stack
from gflownet.envs.grid import Grid
from gflownet.utils.common import copy


class DecisionTreeNodeDiscrete(Stack):

    def __init__(
        self,
        features: Sequence,
        n_thresholds: int = 9,
        cell_min: float = 0.0,
        cell_max: float = 1.0,
        grid_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        features : Sequence
            The set of features.
        n_thresholds : int
            Number of discrete threshold values (grid cells along the single
            threshold dimension). Must be >= 2.
        cell_min : float
            Lower bound of the threshold grid (typically 0.0 if the data is
            min-max scaled).
        cell_max : float
            Upper bound of the threshold grid (typically 1.0 if the data is
            min-max scaled).
        grid_kwargs : dict, optional
            Extra keyword arguments forwarded to the Grid sub-environment
            (e.g. ``max_increment``, ``max_dim_per_action``).
        """
        assert n_thresholds >= 2, "n_thresholds must be >= 2."

        self.features = features
        self.n_features = len(features)
        self.n_thresholds = n_thresholds
        self.cell_min = cell_min
        self.cell_max = cell_max
        self.grid_kwargs = grid_kwargs or {}

        self.stage_feature = 0
        self.feature = Choice(self.features, **kwargs)
        self.stage_threshold = 1
        self.threshold = Grid(
            n_dim=1,
            length=self.n_thresholds,
            cell_min=self.cell_min,
            cell_max=self.cell_max,
            **self.grid_kwargs,
            **kwargs,
        )
        # Active threshold bounds (set by the parent Tree before sampling each
        # node). Initially equal to the full range so masking is a no-op.
        self._tb_lower: float = float(self.cell_min)
        self._tb_upper: float = float(self.cell_max)
        super().__init__(subenvs=tuple([self.feature, self.threshold]), **kwargs)

    # Symmetric attribute names shared with DecisionTreeNode so that the
    # Tree environment can interact with both nodes through a single interface.
    @property
    def threshold_min(self) -> float:
        return float(self.cell_min)

    @property
    def threshold_max(self) -> float:
        return float(self.cell_max)

    @property
    def feature_env(self) -> Choice:
        """Returns the Choice sub-environment used for feature selection."""
        return self.subenvs[self.stage_feature]

    @property
    def threshold_env(self) -> Grid:
        """Returns the Grid sub-environment used for threshold selection."""
        return self.subenvs[self.stage_threshold]

    @property
    def threshold_values(self) -> np.ndarray:
        """The vector of valid threshold values (cell centers of the grid)."""
        return self.threshold_env.cells

    # State helpers
    def get_feature(self, state: Optional[Dict] = None) -> Optional[int]:
        """
        Returns the selected feature index from the state, or None if no feature
        has been selected yet.
        """
        state = self._get_state(state)
        feature_state = self._get_substate(state, self.stage_feature)
        if self.feature_env.is_source(feature_state):
            return None
        return feature_state[0]

    def get_threshold(self, state: Optional[Dict] = None) -> Optional[float]:
        """
        Returns the threshold value (a float in [cell_min, cell_max]) from the
        state, or None if the threshold has not been set yet.

        Note: unlike the continuous variant, the Grid's source [0] is itself a
        valid cell, so we cannot rely on ``threshold_env.is_source`` alone to
        decide whether the threshold has been set. We use the Stack's active
        stage to disambiguate: the threshold is only considered "set" once the
        threshold sub-environment has been activated *and* either an action has
        been taken (state != source) or the stage has been left (i.e. the node
        is fully built).
        """
        state = self._get_state(state)
        active = state.get("_active", 0)
        if active < self.stage_threshold:
            return None
        threshold_state = self._get_substate(state, self.stage_threshold)
        # Threshold sub-environment is currently active but no action has been
        # taken yet -> threshold is still pending.
        if active == self.stage_threshold and self.threshold_env.is_source(
            threshold_state
        ):
            return None
        cell_idx = int(threshold_state[0])
        return float(self.threshold_values[cell_idx])

    def threshold_to_cell_idx(self, value: float) -> int:
        """
        Maps a threshold value in [cell_min, cell_max] to the index of the
        nearest cell on the threshold grid.
        """
        return int(np.argmin(np.abs(self.threshold_values - value)))

    # State Conversion

    def state2readable(self, state: Optional[Dict] = None) -> str:
        """
        Converts a state into a human-readable string.

        Format: ``"Feature: <name>; Threshold: <value>"``. If either has not
        been selected yet, ``"<pending>"`` is shown.
        """
        state = self._get_state(state)
        feature_idx = self.get_feature(state)
        threshold = self.get_threshold(state)
        if feature_idx is not None:
            feature_name = str(self.features[feature_idx - 1])
        else:
            feature_name = "<pending>"
        threshold_str = f"{threshold:.4f}" if threshold is not None else "<pending>"
        return f"Feature: {feature_name}; Threshold: {threshold_str}"

    def readable2state(self, readable: str) -> Dict:
        """
        Converts a human-readable representation of a state back into
        environment format.
        """
        parts = readable.split("; ")
        feature_part = parts[0].split(": ", 1)[1]
        threshold_part = parts[1].split(": ", 1)[1]

        # Parse feature
        if feature_part == "<pending>":
            feature_state = copy(self.feature_env.source)
        else:
            feature_idx = self.features.index(feature_part) + 1
            feature_state = [feature_idx]

        # Parse threshold (stored as a cell index in the Grid sub-environment)
        if threshold_part == "<pending>":
            threshold_state = copy(self.threshold_env.source)
        else:
            cell_idx = self.threshold_to_cell_idx(float(threshold_part))
            threshold_state = [cell_idx]

        # Determine active subenv
        if feature_part == "<pending>":
            active = 0
        elif threshold_part == "<pending>":
            active = 1
        else:
            active = self.n_subenvs

        state = {"_active": active}
        state[self.stage_feature] = feature_state
        state[self.stage_threshold] = threshold_state
        return state

    # =========================================================================
    # Threshold-constraint interface
    #
    # Called by the parent Tree environment to inform the node about the valid
    # threshold range imposed by ancestor decisions. For the discrete node, the
    # bounds are enforced directly during sampling by masking out invalid
    # threshold cells in :py:meth:`get_mask_invalid_actions_forward`. No
    # post-done rescaling is needed because every sampled threshold is already
    # in ``[lower, upper]`` by construction.
    # =========================================================================

    def set_threshold_bounds(self, lower: float, upper: float) -> None:
        """Stores the threshold bounds used for the next mask computation."""
        self._tb_lower = float(lower)
        self._tb_upper = float(upper)

    def clear_threshold_bounds(self) -> None:
        """Resets the threshold bounds to the full ``[cell_min, cell_max]`` range."""
        self._tb_lower = float(self.cell_min)
        self._tb_upper = float(self.cell_max)

    def apply_threshold_rescale(
        self, substate: Dict, lower: float, upper: float
    ) -> Dict:
        """No-op for the discrete node: thresholds are already in range."""
        return substate

    def unapply_threshold_rescale(
        self, substate: Dict, lower: float, upper: float
    ) -> Dict:
        """No-op for the discrete node: thresholds are already in range."""
        return substate

    # =========================================================================
    # Mask override: apply threshold-bound masking on the Grid sub-environment
    # =========================================================================

    def _threshold_subenv_bounds_mask(self, threshold_state: List[int]) -> List[bool]:
        """
        Returns an "out-of-bounds" mask over the Grid sub-environment's action
        space given the currently active threshold bounds.

        For a Grid in 1D with ``cells = linspace(cell_min, cell_max, length)``,
        a forward action ``(k,)`` is invalid when:

        - ``k > 0``: it would land on a cell whose value exceeds ``self._tb_upper``.
        - ``k == 0`` (Grid EOS): the current cell's value is below ``self._tb_lower``.

        The mask is ANDed with the Grid's intrinsic forward mask by the caller.
        """
        grid = self.threshold_env
        current_idx = int(threshold_state[0])
        cells = grid.cells
        lower = self._tb_lower
        upper = self._tb_upper
        mask: List[bool] = []
        for action in grid.action_space:
            if action == grid.eos:
                # EOS: terminating with a threshold strictly below the lower
                # bound would produce an invalid split (degenerate routing).
                mask.append(bool(cells[current_idx] < lower))
            else:
                k = int(action[0])
                new_idx = current_idx + k
                if new_idx >= grid.length:
                    # Already invalid via the Grid's intrinsic mask, repeat
                    # here for safety so an OR with this mask remains correct.
                    mask.append(True)
                else:
                    mask.append(bool(cells[new_idx] > upper))
        return mask

    def get_mask_invalid_actions_forward(
        self, state: Optional[Dict] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the forward mask for the Stack and additionally invalidates
        actions that would lead to a threshold outside the active bounds.

        The base Stack mask is computed first; if the active sub-environment is
        the threshold (Grid), the threshold-bounds mask is OR-ed into the
        relevant slice of the flat mask.
        """
        state = self._get_state(state)
        mask = list(super().get_mask_invalid_actions_forward(state, done))

        # Only apply bounds when the threshold stage is currently active.
        active_subenv = self._get_active_subenv(state)
        if active_subenv != self.stage_threshold:
            return mask

        threshold_state = self._get_substate(state, self.stage_threshold)
        bounds_mask = self._threshold_subenv_bounds_mask(threshold_state)

        # The Stack mask format is:
        #   [one-hot subenv idx (n_subenvs)] + [active subenv mask] + [padding].
        # The threshold sub-env mask starts right after the one-hot prefix.
        offset = self.n_subenvs
        for i, invalid in enumerate(bounds_mask):
            if invalid:
                mask[offset + i] = True
        return mask

    def get_valid_actions(
        self,
        mask: Optional[List[bool]] = None,
        state: Optional[Dict] = None,
        done: Optional[bool] = None,
        backward: Optional[bool] = False,
    ):
        """
        Returns the list of valid actions, ensuring that the customized forward
        mask (which incorporates threshold-bound constraints) is used when no
        mask is supplied. Without this override, ``Stack.get_valid_actions``
        would compute the Grid sub-env's mask directly and miss the bounds.
        """
        if mask is None and not backward:
            mask = self.get_mask_invalid_actions_forward(state, done)
        return super().get_valid_actions(
            mask=mask, state=state, done=done, backward=backward
        )
