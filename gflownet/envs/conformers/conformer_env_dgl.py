import copy
import pickle
import os
import numpy as np
import numpy.typing as npt

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem
from torchtyping import TensorType

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.utils.common import tfloat
from gflownet.utils.molecule.constants import ad_atom_types
from gflownet.utils.molecule.featurizer import MolDGLFeaturizer
from gflownet.utils.molecule.dgl_conformer import DGLConformer
from gflownet.utils.molecule.rdkit_utils import get_rdkit_atom_positions
# from gflownet.utils.molecule.rotatable_bonds import find_rotor_from_smile


class ConformerDGLEnv(ContinuousTorus):
    def __init__(
        self,
        smiles: str,
        torsion_indices: Optional[List[int]] = None,
        policy_type: str = "mlp",
        extra_opt: bool = False,
        save_init_pos = False,
        **kwargs,
    ):
        # changed previous default: 
        # if torsion_indices is None it means we take all rotatable torsion angles 
        # any other desirable behaviour should be specified with a list of torsion_indices

        # maybe change extra_opt to true
        # test saving initial conformer file to the disci
        atom_positions = get_rdkit_atom_positions(smiles, extra_opt=extra_opt)
        self.conformer = DGLConformer(atom_positions, smiles, torsion_indices)

        # Conversions
        self.statebatch2oracle = self.statebatch2proxy
        self.statetorch2oracle = self.statetorch2proxy
        if policy_type == "gnn":
            self.statebatch2policy = self.statebatch2policy_gnn
        elif policy_type != "mlp":
            raise ValueError(
                f"Unrecognized policy_type = {policy_type}, expected either 'mlp' or 'gnn'."
            )

        super().__init__(n_dim=self.conformer.n_rotatable_bonds, **kwargs)
        if save_init_pos:
            self.save_initial_positions(atom_positions)

    def save_initial_positions(self, positions):
        cwd = Path(os.getcwd())
        now = datetime.now()
        time_format = now.strftime("%Y%m%d_%H%M%S")
        file_name = f"initial_pos_env_id_{self.id}_time_{time_format}.pkl"
        file_path = cwd / file_name
        with open(file_path, 'wb') as file:
            pickle.dump(positions, file)
        print(f'Saved initial atom positions at {file_path}')
        return file_path

    def get_conformer_synced_with_state(self, state: List = None):
        if state is None:
            state = self.state
        # cut off step, convert to tensor
        angles = tfloat(state[:-1], device=self.device, float_type=self.float)
        self.conformer.set_rotatable_torsion_angles(angles)
        # deepcopy maybe slow, figure out a better way
        return copy.deepcopy(self.conformer)

    def statebatch2proxy(self, states: List[List]) -> npt.NDArray:
        """
        Returns a list of proxy states, each being a numpy array with dimensionality
        (n_atoms, 4), in which the first column encodes atomic number, and the last
        three columns encode atom positions.
        """
        states_proxy = []
        for st in states:
            conf = self.get_conformer_synced_with_state(st)
            states_proxy.append(
                np.concatenate(
                    [
                        self._tonp(conf.get_atomic_numbers())[..., np.newaxis],
                        self._tonp(conf.get_atom_positions()),
                    ],
                    axis=1,
                )
            )
        return np.array(states_proxy)

    def statetorch2proxy(self, states: TensorType["batch", "state_dim"]) -> npt.NDArray:
        return self.statebatch2proxy(self._tonp(states))

    def statebatch2policy_gnn(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Returns an array of GNN-format policy inputs with dimensionality
        (n_states, n_atoms, 4), in which the first three columns encode atom positions,
        and the last column encodes current timestep.
        """
        policy_input = []
        for state in states:
            conformer = self.get_conformer_synced_with_state(state)
            positions = conformer.get_atom_positions()
            policy_input.append(
                np.concatenate(
                    [positions, np.full((positions.shape[0], 1), state[-1])],
                    axis=1,
                )
            )
        return np.array(policy_input)
    
    def _tonp(tensor):
        return np.array(tensor.tolist())

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
