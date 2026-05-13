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

from typing import Dict, Optional, Sequence

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
        super().__init__(subenvs=tuple([self.feature, self.threshold]), **kwargs)

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
