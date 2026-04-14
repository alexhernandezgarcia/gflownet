"""
Useless playground script: DecisionTreeNodeSet

Reimplements DecisionTreeNode (gflownet/envs/tree/node.py) using SetFix instead of
Stack. The original uses Stack, which enforces a fixed order: first pick the feature
(Choice), then pick the threshold (ContinuousCube). This version uses SetFix, so the
two sub-environments can be completed in any order.
"""

from typing import Dict, Optional, Sequence

from gflownet.envs.choice import Choice
from gflownet.envs.composite.setfix import SetFix
from gflownet.envs.cube import ContinuousCube


class DecisionTreeNodeSet(SetFix):

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
            The set of features available for splitting.
        cube_kwargs : dict, optional
            Extra keyword arguments forwarded to ContinuousCube (the threshold env).
        """
        self.features = features
        self.cube_kwargs = cube_kwargs or {}

        self.feature = Choice(self.features, **kwargs)
        self.threshold = ContinuousCube(n_dim=1, **self.cube_kwargs)

        super().__init__(subenvs=[self.feature, self.threshold], **kwargs)
