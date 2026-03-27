"""
Node of a decision tree.
"""

from typing import Sequence

from gflownet.envs.choice import Choice
from gflownet.envs.composite.stack import Stack
from gflownet.envs.cube import ContinuousCube


class DecisionTreeNode(Stack):

    def __init__(
        self,
        features: Sequence,
        **kwargs,
    ):
        """
        Parameters
        ----------
        features : Sequence
            The set of features.
        """
        self.features = features
        self.stage_feature = 0
        self.feature = Choice(self.features, **kwargs)
        self.stage_threshold = 1
        self.threshold = ContinuousCube(n_dim=1, **kwargs)
        super().__init__(subenvs=tuple([self.feature, self.threshold]), **kwargs)
