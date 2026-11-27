"""
Node of a decision tree with a discrete threshold modeled as a Choice.

A DecisionTreeNodeDiscreteChoice represents a single decision node consisting of
two stages:
1. Choosing a feature (via a Choice sub-environment).
2. Choosing a threshold value directly from a finite set of equally-spaced
   options (via a second Choice sub-environment).

This is an alternative to ``DecisionTreeNodeDiscrete`` where the threshold is
selected in a single action (plus EOS) rather than incrementally on a 1D Grid.
For ``n_thresholds=9`` with ``cell_min=0``, ``cell_max=1`` the available
threshold values are ``0.1, 0.2, ..., 0.9`` (interior points of the unit
interval, so degenerate boundary splits cannot occur).
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from gflownet.envs.choice import Choice
from gflownet.envs.composite.stack import Stack
from gflownet.utils.common import copy


class DecisionTreeNodeDiscreteChoice(Stack):

    def __init__(
        self,
        features: Sequence,
        n_thresholds: int = 9,
        cell_min: float = 0.0,
        cell_max: float = 1.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        features : Sequence
            The set of features.
        n_thresholds : int
            Number of discrete threshold values. Must be >= 2. The values are
            placed at the interior points of ``[cell_min, cell_max]`` spaced by
            ``(cell_max - cell_min) / (n_thresholds + 1)``.
        cell_min : float
            Lower bound of the threshold support (typically 0.0 if the data is
            min-max scaled).
        cell_max : float
            Upper bound of the threshold support (typically 1.0 if the data is
            min-max scaled).
        """
        assert n_thresholds >= 2, "n_thresholds must be >= 2."

        self.features = features
        self.n_features = len(features)
        self.n_thresholds = n_thresholds
        self.cell_min = float(cell_min)
        self.cell_max = float(cell_max)

        step = (self.cell_max - self.cell_min) / (self.n_thresholds + 1)
        self._threshold_values = self.cell_min + step * np.arange(
            1, self.n_thresholds + 1
        )

        self.stage_feature = 0
        self.feature = Choice(self.features, **kwargs)
        self.stage_threshold = 1
        self.threshold = Choice(
            options=[f"{v:.6g}" for v in self._threshold_values], **kwargs
        )
        # Active threshold bounds (set by the parent Tree before sampling each
        # node). Initially equal to the full range so masking is a no-op.
        self._tb_lower: float = self.cell_min
        self._tb_upper: float = self.cell_max
        super().__init__(subenvs=tuple([self.feature, self.threshold]), **kwargs)

    # Symmetric attribute names shared with DecisionTreeNode so that the
    # Tree environment can interact with both nodes through a single interface.
    @property
    def threshold_min(self) -> float:
        return self.cell_min

    @property
    def threshold_max(self) -> float:
        return self.cell_max

    @property
    def feature_env(self) -> Choice:
        """Returns the Choice sub-environment used for feature selection."""
        return self.subenvs[self.stage_feature]

    @property
    def threshold_env(self) -> Choice:
        """Returns the Choice sub-environment used for threshold selection."""
        return self.subenvs[self.stage_threshold]

    @property
    def threshold_values(self) -> np.ndarray:
        """The vector of valid threshold values."""
        return self._threshold_values

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
        Returns the threshold value from the state, or None if the threshold
        has not been set yet.

        The threshold sub-environment is a Choice whose source is ``[0]`` and
        whose selected state is ``[i]`` with ``1 <= i <= n_thresholds``. The
        returned value is ``threshold_values[i - 1]``.
        """
        state = self._get_state(state)
        active = state.get("_active", 0)
        if active < self.stage_threshold:
            return None
        threshold_state = self._get_substate(state, self.stage_threshold)
        if self.threshold_env.is_source(threshold_state):
            return None
        return float(self._threshold_values[threshold_state[0] - 1])

    def threshold_to_choice_idx(self, value: float) -> int:
        """
        Maps a threshold value in [cell_min, cell_max] to the 1-based Choice
        index of the nearest discrete threshold option.
        """
        return int(np.argmin(np.abs(self._threshold_values - value))) + 1

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

        # Parse threshold (stored as a 1-based index in the Choice sub-env)
        if threshold_part == "<pending>":
            threshold_state = copy(self.threshold_env.source)
        else:
            choice_idx = self.threshold_to_choice_idx(float(threshold_part))
            threshold_state = [choice_idx]

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
    # threshold range imposed by ancestor decisions. The bounds are enforced
    # directly during sampling by masking out invalid threshold options in
    # :py:meth:`get_mask_invalid_actions_forward`. No post-done rescaling is
    # needed because every sampled threshold is already in ``[lower, upper]``
    # by construction.
    # =========================================================================

    def set_threshold_bounds(self, lower: float, upper: float) -> None:
        """Stores the threshold bounds used for the next mask computation."""
        self._tb_lower = float(lower)
        self._tb_upper = float(upper)

    def clear_threshold_bounds(self) -> None:
        """Resets the threshold bounds to the full ``[cell_min, cell_max]`` range."""
        self._tb_lower = self.cell_min
        self._tb_upper = self.cell_max

    def apply_threshold_rescale(
        self, substate: Dict, lower: float, upper: float
    ) -> Dict:
        """No-op for the discrete-choice node: thresholds are already in range."""
        return substate

    def unapply_threshold_rescale(
        self, substate: Dict, lower: float, upper: float
    ) -> Dict:
        """No-op for the discrete-choice node: thresholds are already in range."""
        return substate

    # =========================================================================
    # Mask override: apply threshold-bound masking on the Choice sub-environment
    # =========================================================================

    def _threshold_subenv_bounds_mask(self, threshold_state: List[int]) -> List[bool]:
        """
        Returns an "out-of-bounds" mask over the threshold Choice sub-env's
        action space given the currently active threshold bounds.

        The Choice action space is
        ``[(1,), (2,), ..., (n_thresholds,), (-1,)]``. For each selection
        action ``(i,)``, the entry is True (invalid) when
        ``threshold_values[i - 1]`` is outside ``[lower, upper]``. The EOS
        entry is always False here (the Choice's intrinsic mask already
        invalidates EOS at the source state).

        The returned mask is OR-ed with the threshold Choice's intrinsic
        forward mask by the caller.
        """
        choice = self.threshold_env
        lower = self._tb_lower
        upper = self._tb_upper
        mask: List[bool] = []
        for action in choice.action_space:
            if action == choice.eos:
                mask.append(False)
            else:
                idx = int(action[0])
                value = float(self._threshold_values[idx - 1])
                mask.append(bool(value < lower or value > upper))
        return mask

    def get_mask_invalid_actions_forward(
        self, state: Optional[Dict] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the forward mask for the Stack and additionally invalidates
        actions that would lead to a threshold outside the active bounds.
        """
        state = self._get_state(state)
        mask = list(super().get_mask_invalid_actions_forward(state, done))

        # Only apply bounds when the threshold stage is currently active.
        active_subenv = self._get_active_subenv(state)
        if active_subenv != self.stage_threshold:
            return mask

        bounds_mask = self._threshold_subenv_bounds_mask(
            self._get_substate(state, self.stage_threshold)
        )

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
        mask is supplied.
        """
        if mask is None and not backward:
            mask = self.get_mask_invalid_actions_forward(state, done)
        return super().get_valid_actions(
            mask=mask, state=state, done=done, backward=backward
        )
