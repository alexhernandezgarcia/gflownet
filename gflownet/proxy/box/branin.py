"""
Branin objective function, relying on the botorch implementation.

This code is based on the implementation by Nikita Saxena (nikita-0209) in 
https://github.com/alexhernandezgarcia/activelearning

The implementation assumes by default that the inputs will be on [0, 1] x [0, 1] and
will be mapped to the standard domain of the Branin function (see X1_DOMAIN and
X2_DOMAIN). Setting do_domain_map to False will prevent the mapping.

Branin function is typically used as a minimization problem, with the minima around
zero but positive. In order to map the range into the convential negative range, an
upper bound of of Branin in the standard domain (UPPER_BOUND_IN_DOMAIN) is subtracted.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.test_functions.multi_fidelity import AugmentedBranin
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchtyping import TensorType

from gflownet.proxy.base import Proxy

X1_DOMAIN = [-5, 10]
X1_LENGTH = X1_DOMAIN[1] - X1_DOMAIN[0]
X2_DOMAIN = [0, 15]
X2_LENGTH = X2_DOMAIN[1] - X2_DOMAIN[0]
UPPER_BOUND_IN_DOMAIN = 309


class Branin(Proxy):
    def __init__(self, fidelity=1.0, do_domain_map=True, **kwargs):
        """
        fidelity : float
            Fidelity of the Branin oracle. 1.0 corresponds to the original Branin.
            Smaller values (up to 0.0) reduce the fidelity of the oracle.

        See: https://botorch.org/api/test_functions.html
        """
        super().__init__(**kwargs)
        self.fidelity = fidelity
        self.do_domain_map = do_domain_map
        self.function_mf_botorch = AugmentedBranin(negate=False)
        # Modes and extremum compatible with 100x100 grid
        self.modes = [
            [12.4, 81.833],
            [54.266, 15.16],
            [94.98, 16.5],
        ]
        self.extremum = 0.397887

    def __call__(self, states: TensorType["batch", "2"]) -> TensorType["batch"]:
        if states.shape[1] != 2:
            raise ValueError(
                """
            Inputs to the Branin function must be 2-dimensional, but inputs with
            {states.shape[1]} dimensions were passed.
            """
            )
        if self.do_domain_map:
            states = Branin.map_to_standard_domain(states)
        # Append fidelity as a new dimension of states
        states = torch.cat(
            [
                states,
                self.fidelity
                * torch.ones(
                    states.shape[0], device=self.device, dtype=self.float
                ).unsqueeze(-1),
            ],
            dim=1,
        )
        return Branin.map_to_negative_range(self.function_mf_botorch(states))

    @property
    def min(self):
        if not hasattr(self, "_min"):
            self._min = torch.tensor(
                -UPPER_BOUND_IN_DOMAIN, device=self.device, dtype=self.float
            )
        return self._min

    @staticmethod
    def map_to_standard_domain(
        states: TensorType["batch", "2"]
    ) -> TensorType["batch", "2"]:
        """
        Maps a batch of input states onto the domain typically used to evaluate the
        Branin function. See X1_DOMAIN and X2_DOMAIN. It assumes that the inputs are on
        [0, 1] x [0, 1].
        """
        states[:, 0] = X1_DOMAIN[0] + states[:, 0] * X1_LENGTH
        states[:, 1] = X2_DOMAIN[0] + states[:, 1] * X2_LENGTH
        return states

    @staticmethod
    def map_to_negative_range(values: TensorType["batch"]) -> TensorType["batch"]:
        """
        Maps a batch of function values onto a negative range by substracting an upper
        bound of the Branin function in the standard domain (UPPER_BOUND_IN_DOMAIN).
        """
        return values - UPPER_BOUND_IN_DOMAIN

    @staticmethod
    def map_to_standard_range(values: TensorType["batch"]) -> TensorType["batch"]:
        """
        Maps a batch of function values in a negative range back onto the standard
        range by adding an upper bound of the Branin function in the standard domain
        (UPPER_BOUND_IN_DOMAIN).
        """
        return values + UPPER_BOUND_IN_DOMAIN

    def plot_true_rewards(self, env, ax, rescale):
        states = torch.FloatTensor(env.get_all_terminating_states()).to(self.device)
        states_oracle = states.clone()
        grid_size = env.length
        states_oracle = states_oracle / (grid_size - 1)
        states_oracle[:, 0] = states_oracle[:, 0] * rescale - 5
        states_oracle[:, 1] = states_oracle[:, 1] * rescale
        scores = self(states_oracle).detach().cpu().numpy()
        if hasattr(self, "fid"):
            title = "Oracle Energy (TrainY) with fid {}".format(self.fid)
        else:
            title = "Oracle Energy (TrainY)"
        # what the GP is trained on
        #         if self.maximize == False:
        #             scores = scores * (-1)
        index = states.long().detach().cpu().numpy()
        grid_scores = np.zeros((env.length, env.length))
        grid_scores[index[:, 0], index[:, 1]] = scores
        ax.set_xticks(
            np.arange(start=0, stop=env.length, step=int(env.length / rescale))
        )
        ax.set_yticks(
            np.arange(start=0, stop=env.length, step=int(env.length / rescale))
        )
        ax.imshow(grid_scores)
        ax.set_title(title)
        im = ax.imshow(grid_scores)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
        return ax
