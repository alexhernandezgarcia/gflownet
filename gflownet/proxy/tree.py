from typing import Optional

import numpy as np
import torch
from torchtyping import TensorType

from gflownet.envs.tree import Tree as TreeEnvironment
from gflownet.proxy.base import Proxy


class Tree(Proxy):
    """
    Simple decision tree proxy that uses empirical frequency of correct predictions for
    computing likelihood, and the number of leafs in the tree for computing the prior.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.X = None
        self.y = None
        self.max_depth = None

    def setup(self, env: Optional[TreeEnvironment] = None):
        self.X = env.X
        self.y = env.y
        self.max_depth = env.max_depth

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        tree = TreeEnvironment.__new__(TreeEnvironment)
        energies = []

        for state in states:
            tree.state = state

            predictions = []
            for x in self.X:
                predictions.append(tree.predict(x))

            likelihood = (np.array(predictions) == self.y).mean()
            prior = 1 - np.log2(len(TreeEnvironment._find_leafs(state))) / (
                self.max_depth - 1
            )
            energies.append(-likelihood * prior)

        return torch.Tensor(energies)
