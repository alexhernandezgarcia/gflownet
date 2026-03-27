"""
Node of a decision tree.
"""

from typing import Dict, Optional, Sequence

from gflownet.envs.choice import Choice
from gflownet.envs.composite.stack import Stack
from gflownet.envs.cube import ContinuousCube


class DecisionTreeNode(Stack):

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
        """
        self.features = features
        self.cube_kwargs = cube_kwargs or {}

        self.stage_feature = 0
        self.feature = Choice(self.features, **kwargs)
        self.stage_threshold = 1
        self.threshold = ContinuousCube(n_dim=1, **self.cube_kwargs)
        super().__init__(subenvs=tuple([self.feature, self.threshold]), **kwargs)
