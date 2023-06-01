"""
Branin objective function, relying on the botorch implementation.

This code is based on the implementation by Nikita Saxena (nikita-0209) in 
https://github.com/alexhernandezgarcia/activelearning
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.test_functions.multi_fidelity import AugmentedBranin
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchtyping import TensorType


class Branin(Proxy):
    def __init__(self, fidelity=1.0, **kwargs):
        """
        fidelity : float
            Fidelity of the Branin oracle. 1.0 corresponds to the original Branin.
            Smaller values (up to 0.0) reduce the fidelity of the oracle.

        See: https://botorch.org/api/test_functions.html
        """
        super().__init__(**kwargs)
        self.fidelity = fidelity
        self.function_mf_botorch = AugmentedBranin(negate=True)
        # Modes and extremum compatible with 100x100 grid
        self.modes = [
            [12.4, 81.833],
            [54.266, 15.16],
            [94.98, 16.5],
        ]
        self.extremum = 0.397887

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        if states.shape[1] != 2:
            raise ValueError(
                """
            Inputs to the Branin function must be 2-dimensional, but inputs with
            {states.shape[1]} dimensions were passed.
            """
            )
        # TODO: need to map states onto [-5, 10] x [0, 15]?
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
        return self.function_mf_botorch(states)

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
        if self.maximize == False:
            scores = scores * (-1)
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
