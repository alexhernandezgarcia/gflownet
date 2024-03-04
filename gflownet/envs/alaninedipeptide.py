from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from torchtyping import TensorType

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.utils.molecule import constants
from gflownet.utils.molecule.atom_positions_dataset import AtomPositionsDataset
from gflownet.utils.molecule.conformer_base import ConformerBase


class AlanineDipeptide(ContinuousTorus):
    """Simple extension of 2d continuous torus where reward function is defined by the
    energy of the alanine dipeptide molecule"""

    def __init__(
        self,
        path_to_dataset,
        url_to_dataset,
        **kwargs,
    ):
        self.atom_positions_dataset = AtomPositionsDataset(
            path_to_dataset, url_to_dataset
        )
        atom_positions = self.atom_positions_dataset.sample()
        self.conformer = ConformerBase(
            atom_positions, constants.ad_smiles, constants.ad_free_tas
        )
        n_dim = len(self.conformer.freely_rotatable_tas)
        super().__init__(**kwargs)
        self.sync_conformer_with_state()

    def sync_conformer_with_state(self, state: List = None):
        if state is None:
            state = self.state
        for idx, ta in enumerate(self.conformer.freely_rotatable_tas):
            self.conformer.set_torsion_angle(ta, state[idx])
        return self.conformer

    # TODO: are the conversions to oracle relevant?
    def states2proxy(
        self, states: Union[List[List], TensorType["batch", "state_dim"]]
    ) -> npt.NDArray:
        """
        Prepares a batch of states in "environment format" for the proxy: each state is
        a vector of length n_dim where each value is an angle in radians. The n_actions
        item is removed. This transformation is obtained from the parent states2proxy.

        Important: this method returns a numpy array, unlike in most other
        environments.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A numpy array containing all the states in the batch.
        """
        return super().states2proxy(states).numpy()


if __name__ == "__main__":
    import sys

    path_to_data = sys.argv[1]
    print(path_to_data)
    env = AlanineDipeptide(path_to_data)
    print("initial state:", env.state)
    conf = env.sync_conformer_with_state()
    tas = conf.get_freely_rotatable_tas_values()
    print("initial conf torsion angles:", tas)

    # apply simple action
    new_state, action, done = env.step((0, 1.2))
    print("action:", action)
    print("new_state:", new_state)
    conf = env.sync_conformer_with_state()
    tas = conf.get_freely_rotatable_tas_values()
    print("new conf torsion angles:", tas)
    print("done:", done)
