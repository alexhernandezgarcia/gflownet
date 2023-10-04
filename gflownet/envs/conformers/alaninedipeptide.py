from copy import deepcopy
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torchtyping import TensorType

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.utils.molecule import constants
from gflownet.utils.molecule.datasets import AtomPositionsDataset
from gflownet.utils.molecule.rdkit_conformer import RDKitConformer


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
        self.conformer = RDKitConformer(
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

    def statetorch2proxy(self, states: TensorType["batch", "state_dim"]) -> npt.NDArray:
        """
        Prepares a batch of states in torch "GFlowNet format" for the oracle.
        """
        device = states.device
        if device == torch.device("cpu"):
            np_states = states.numpy()
        else:
            np_states = states.cpu().numpy()
        return np_states[:, :-1]

    def statebatch2proxy(self, states: List[List]) -> npt.NDArray:
        """
        Prepares a batch of states in "GFlowNet format" for the proxy: a tensor where
        each state is a row of length n_dim with an angle in radians. The n_actions
        item is removed.
        """
        return np.array(states)[:, :-1]

    def statetorch2oracle(
        self, states: TensorType["batch", "state_dim"]
    ) -> List[Tuple[npt.NDArray, npt.NDArray]]:
        """
        Prepares a batch of states in torch "GFlowNet format" for the oracle.
        """
        device = states.device
        if device == torch.device("cpu"):
            np_states = states.numpy()
        else:
            np_states = states.cpu().numpy()
        result = self.statebatch2oracle(np_states)
        return result

    def statebatch2oracle(
        self, states: List[List]
    ) -> List[Tuple[npt.NDArray, npt.NDArray]]:
        """
        Prepares a batch of states in "GFlowNet format" for the oracle: a list of
        tuples, where first element in the tuple is numpy array of atom positions of
        shape [num_atoms, 3] and the second element is numpy array of atomic numbers of
        shape [num_atoms, ]
        """
        states_oracle = []
        for st in states:
            conf = self.sync_conformer_with_state(st)
            states_oracle.append((conf.get_atom_positions(), conf.get_atomic_numbers()))
        return states_oracle


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
