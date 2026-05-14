"""
Node of a decision tree.

A DecisionTreeNode represents a single decision node consisting of two stages:
1. Choosing a feature (via a Choice sub-environment).
2. Choosing a threshold (via a ContinuousCube sub-environment).

The environment is built as a Stack of these two sub-environments.
"""

from typing import Dict, List, Optional, Sequence, Union

import torch
from torchtyping import TensorType

from gflownet.envs.choice import Choice
from gflownet.envs.composite.stack import Stack
from gflownet.envs.cube import ContinuousCube
from gflownet.utils.common import copy, tfloat


class DecisionTreeNode(Stack):

    # Lower/upper bounds for the threshold sub-environment. Stored on the
    # instance and re-set per active node by the parent Tree environment via the
    # threshold-constraint interface (see ``set_threshold_bounds``).
    threshold_min: float = 0.0
    threshold_max: float = 1.0

    def __init__(
        self,
        features: Sequence,
        cube_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        features : Sequence
            The set of features.
        cube_kwargs : dict, optional
            Additional keyword arguments passed to the ContinuousCube sub-environment
            for the threshold.
        """
        self.features = features
        self.n_features = len(features)
        self.cube_kwargs = cube_kwargs or {}

        self.stage_feature = 0
        self.feature = Choice(self.features, **kwargs)
        self.stage_threshold = 1
        self.threshold = ContinuousCube(n_dim=1, **self.cube_kwargs)
        # Active threshold bounds (set by the parent Tree before sampling /
        # finishing a node). The ContinuousCube always samples in [0, 1] so the
        # bounds are only consumed at done-time by ``apply_threshold_rescale``.
        self._tb_lower: float = self.threshold_min
        self._tb_upper: float = self.threshold_max
        super().__init__(subenvs=tuple([self.feature, self.threshold]), **kwargs)

    # Properties for convenient access to sub-environments
    @property
    def feature_env(self) -> Choice:
        """Returns the Choice sub-environment used for feature selection."""
        return self.subenvs[self.stage_feature]

    @property
    def threshold_env(self) -> ContinuousCube:
        """Returns the ContinuousCube sub-environment used for threshold selection."""
        return self.subenvs[self.stage_threshold]

    # State helpers
    def get_feature(self, state: Optional[Dict] = None) -> Optional[int]:
        """
        Returns the selected feature index from the state, or None if no feature has
        been selected yet.

        Parameters
        ----------
        state : dict, optional
            A state. If None, self.state is used.

        Returns
        -------
        int or None
            The selected feature index (1-based from Choice), or None if still at source.
        """
        state = self._get_state(state)
        feature_state = self._get_substate(state, self.stage_feature)
        if self.feature_env.is_source(feature_state):
            return None
        return feature_state[0]

    def get_threshold(self, state: Optional[Dict] = None) -> Optional[float]:
        """
        Returns the threshold value from the state, or None if the threshold has not
        been set yet.

        Parameters
        ----------
        state : dict, optional
            A state. If None, self.state is used.

        Returns
        -------
        float or None
            The threshold value, or None if the threshold sub-environment is still at
            its source.
        """
        state = self._get_state(state)
        threshold_state = self._get_substate(state, self.stage_threshold)
        if self.threshold_env.is_source(threshold_state):
            return None
        return threshold_state[0]

    # State Conversion

    def state2readable(self, state: Optional[Dict] = None) -> str:
        """
        Converts a state into a human-readable string.

        Format: ``"Feature: <name>; Threshold: <value>"`` where <name> is the feature
        name from ``self.features`` and <value> is the threshold. If either is not yet
        selected, ``"<pending>"`` is shown.

        Parameters
        ----------
        state : dict, optional
            A state. If None, self.state is used.

        Returns
        -------
        str
            Human-readable representation.
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
        Converts a human-readable representation of a state back into environment
        format.

        Parameters
        ----------
        readable : str
            A string in the format ``"Feature: <name>; Threshold: <value>"``.

        Returns
        -------
        dict
            A state in environment format.
        """
        parts = readable.split("; ")
        feature_part = parts[0].split(": ", 1)[1]
        threshold_part = parts[1].split(": ", 1)[1]

        # Parse feature
        if feature_part == "<pending>":
            feature_state = self.feature_env.source
        else:
            feature_idx = self.features.index(feature_part) + 1
            feature_state = [feature_idx]

        # Parse threshold
        if threshold_part == "<pending>":
            threshold_state = copy(self.threshold_env.source)
        else:
            threshold_state = [float(threshold_part)]

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
    # threshold range imposed by ancestor decisions. For the continuous node,
    # the ContinuousCube always samples in ``[0, 1]``, so the bounds cannot be
    # honored during sampling and are instead applied as a rescaling at
    # done-time. The bounds are stored on the instance for symmetry with the
    # discrete node, but are not consumed by the mask methods.
    # =========================================================================

    def set_threshold_bounds(self, lower: float, upper: float) -> None:
        """Stores the threshold bounds for the currently active node."""
        self._tb_lower = float(lower)
        self._tb_upper = float(upper)

    def clear_threshold_bounds(self) -> None:
        """Resets the threshold bounds to the full ``[threshold_min, threshold_max]`` range."""
        self._tb_lower = self.threshold_min
        self._tb_upper = self.threshold_max

    def apply_threshold_rescale(
        self, substate: Dict, lower: float, upper: float
    ) -> Dict:
        """
        Rescales the threshold from the raw ``[0, 1]`` cube range to the valid
        ``[lower, upper]`` range determined by ancestor constraints.

        Returns the same ``substate`` reference, mutated in place. If the bounds
        are the trivial ``[0, 1]`` or the threshold is not yet set, returns the
        substate unchanged.
        """
        if lower == self.threshold_min and upper == self.threshold_max:
            return substate
        raw = self.get_threshold(substate)
        if raw is None:
            return substate
        rescaled = lower + raw * (upper - lower)
        substate[self.stage_threshold] = [rescaled]
        return substate

    def unapply_threshold_rescale(
        self, substate: Dict, lower: float, upper: float
    ) -> Dict:
        """
        Reverses :py:meth:`apply_threshold_rescale`: maps a threshold stored in
        ``[lower, upper]`` back to the raw ``[0, 1]`` cube range.

        Returns the same ``substate`` reference, mutated in place.
        """
        if lower == self.threshold_min and upper == self.threshold_max:
            return substate
        actual = self.get_threshold(substate)
        if actual is None:
            return substate
        span = upper - lower
        raw = (actual - lower) / span if span > 0 else 0.0
        substate[self.stage_threshold] = [raw]
        return substate
