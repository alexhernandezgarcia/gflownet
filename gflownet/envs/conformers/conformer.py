import copy
from typing import List

import numpy as np
import numpy.typing as npt
import ray
from torchtyping import TensorType

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.proxy.conformers.xtb import get_energy
from gflownet.utils.molecule.datasets import AtomPositionsDataset
from gflownet.utils.molecule.rdkit_conformer import RDKitConformer


class Conformer(ContinuousTorus):
    """
    Extension of continuous torus to conformer generation. Based on AlanineDipeptide,
    but accepts any molecule (defined by SMILES, freely rotatable torsion angles, and
    path to dataset containing sample conformers).
    """

    def __init__(
        self,
        smiles: str,
        path_to_dataset: str,
        url_to_dataset: str,
        **kwargs,
    ):
        atom_positions_dataset = AtomPositionsDataset(
            smiles, path_to_dataset, url_to_dataset
        )
        atom_positions = atom_positions_dataset.first()
        torsion_angles = atom_positions_dataset.torsion_angles
        self.conformer = RDKitConformer(atom_positions, smiles, torsion_angles)

        tasks = []
        for positions in atom_positions_dataset.positions:
            tasks.append(
                get_energy.remote(self.conformer.get_atomic_numbers(), positions)
            )
        energies = ray.get(tasks)
        self.max_energy = max(energies)
        self.min_energy = min(energies)

        # Conversions
        self.statebatch2oracle = self.statebatch2proxy
        self.statetorch2oracle = self.statetorch2proxy

        super().__init__(n_dim=len(self.conformer.freely_rotatable_tas), **kwargs)

        self.sync_conformer_with_state()

    def sync_conformer_with_state(self, state: List = None):
        if state is None:
            state = self.state
        for idx, ta in enumerate(self.conformer.freely_rotatable_tas):
            self.conformer.set_torsion_angle(ta, state[idx])
        return self.conformer

    def statebatch2proxy(self, states: List[List]) -> List[npt.NDArray]:
        """
        Returns a list of proxy states, each being a numpy array with dimensionality
        (n_atoms, 4), in which the first column encodes atomic number, and the last
        three columns encode atom positions.
        """
        states_proxy = []
        for st in states:
            conf = self.sync_conformer_with_state(st)
            states_proxy.append(
                np.concatenate(
                    [
                        conf.get_atomic_numbers()[..., np.newaxis],
                        conf.get_atom_positions(),
                    ],
                    axis=1,
                )
            )
        return states_proxy

    def statetorch2proxy(
        self, states: TensorType["batch", "state_dim"]
    ) -> List[npt.NDArray]:
        return self.statebatch2proxy(states.cpu().numpy())

    def statebatch2kde(self, states: List[List]) -> npt.NDArray[np.float32]:
        return np.array(states)[:, :-1]

    def statetorch2kde(
        self, states: TensorType["batch_size", "state_dim"]
    ) -> TensorType["batch_size", "state_proxy_dim"]:
        return states.cpu().numpy()[:, :-1]

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_instance = cls.__new__(cls)

        for attr_name, attr_value in self.__dict__.items():
            if attr_name != "conformer":
                setattr(new_instance, attr_name, copy.copy(attr_value))

        new_instance.conformer = self.conformer

        return new_instance
