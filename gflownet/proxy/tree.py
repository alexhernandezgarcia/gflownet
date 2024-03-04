from typing import Optional

import numpy as np
import torch
from torchtyping import TensorType

from gflownet.envs.tree import Tree
from gflownet.proxy.base import Proxy


class TreeProxy(Proxy):
    """
    Simple decision tree proxy that uses empirical frequency of correct predictions for
    computing likelihood, and the number of nodes in the tree for computing the prior.
    """

    def __init__(self, use_prior: bool = True, beta: float = 1.0, **kwargs):
        """
        Parameters
        ----------
        use_prior : bool
            Whether to use -likelihood * prior for energy computation or just the
            -likelihood.
        beta : float
            Beta coefficient in ``prior = np.exp(-self.beta * n_nodes)``. Note that
            this is temporary prior implementation that was used for debugging,
            in combination with reward_func="boltzmann" it doesn't make much sense.
        """
        super().__init__(**kwargs)

        self.use_prior = use_prior
        self.beta = beta
        self.X = None
        self.y = None

    def setup(self, env: Optional[Tree] = None):
        self.X = env.X_train
        self.y = env.y_train

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        energies = []

        for state in states:
            predictions = []
            for x in self.X:
                predictions.append(Tree.predict(state, x))
            likelihood = (np.array(predictions) == self.y).mean()

            if self.use_prior:
                n_nodes = Tree.get_n_nodes(state)
                prior = np.exp(-self.beta * n_nodes)
            else:
                prior = 1

            energies.append(-likelihood * prior)

        return torch.Tensor(energies)
