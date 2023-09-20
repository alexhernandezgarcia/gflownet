import copy
from typing import List, Optional, Tuple

import dgl
import numpy as np
import numpy.typing as npt
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torchtyping import TensorType

from gflownet.envs.ctorus import ContinuousTorus
from gflownet.utils.molecule.constants import ad_atom_types
from gflownet.utils.molecule.featurizer import MolDGLFeaturizer
from gflownet.utils.molecule.rdkit_conformer import RDKitConformer
from gflownet.utils.molecule.rotatable_bonds import find_rotor_from_smiles


class Conformer(ContinuousTorus):
    """
    Extension of continuous torus to conformer generation. Based on AlanineDipeptide,
    but accepts any molecule (defined by SMILES and freely rotatable torsion angles).
    """

    def __init__(
        self,
        smiles: str,
        n_torsion_angles: Optional[int] = 2,
        torsion_indices: Optional[List[int]] = None,
        policy_type: str = "mlp",
        remove_hs: bool = True,
        **kwargs,
    ):
        if torsion_indices is None:
            # We hard code default torsion indices for Alanine Dipeptide to preserve
            # backward compatibility.
            if smiles == "CC(C(=O)NC)NC(=O)C" and n_torsion_angles == 2:
                torsion_indices = [2, 1]
            elif n_torsion_angles == -1:
                torsion_indices = None
            else:
                torsion_indices = list(range(n_torsion_angles))

        self.smiles = smiles
        self.torsion_indices = torsion_indices
        self.atom_positions = Conformer._get_positions(self.smiles)
        self.torsion_angles = Conformer._get_torsion_angles(
            self.smiles, self.torsion_indices
        )
        self.set_conformer()

        # Conversions
        self.statebatch2oracle = self.statebatch2proxy
        self.statetorch2oracle = self.statetorch2proxy
        if policy_type == "gnn":
            self.statebatch2policy = self.statebatch2policy_gnn
        elif policy_type != "mlp":
            raise ValueError(
                f"Unrecognized policy_type = {policy_type}, expected either 'mlp' or 'gnn'."
            )

        self.graph = MolDGLFeaturizer(ad_atom_types).mol2dgl(self.conformer.rdk_mol)
        # TODO: use DGL conformer instead
        rotatable_edges = [ta[1:3] for ta in self.torsion_angles]
        for i in range(self.graph.num_edges()):
            if (
                self.graph.edges()[0][i].item(),
                self.graph.edges()[1][i].item(),
            ) not in rotatable_edges:
                self.graph.edata["rotatable_edges"][i] = False

        # Hydrogen removal
        self.remove_hs = remove_hs
        self.hs = torch.where(self.graph.ndata["atom_features"][:, 0] == 1)[0]
        self.non_hs = torch.where(self.graph.ndata["atom_features"][:, 0] != 1)[0]
        if remove_hs:
            self.graph = dgl.remove_nodes(self.graph, self.hs)

        super().__init__(n_dim=len(self.conformer.freely_rotatable_tas), **kwargs)

        self.sync_conformer_with_state()

    def set_conformer(self, state: Optional[List] = None) -> RDKitConformer:
        self.conformer = RDKitConformer(
            self.atom_positions, self.smiles, self.torsion_angles
        )

        if state is not None:
            self.sync_conformer_with_state(state)

        return self.conformer

    @staticmethod
    def _get_positions(smiles: str) -> npt.NDArray:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=0)
        return mol.GetConformer().GetPositions()

    @staticmethod
    def _get_torsion_angles(
        smiles: str, indices: Optional[List[int]]
    ) -> List[Tuple[int]]:
        torsion_angles = find_rotor_from_smiles(smiles)
        if indices is not None:
            torsion_angles = [torsion_angles[i] for i in indices]
        return torsion_angles

    def sync_conformer_with_state(self, state: List = None):
        if state is None:
            state = self.state
        for idx, ta in enumerate(self.conformer.freely_rotatable_tas):
            self.conformer.set_torsion_angle(ta, state[idx])
        return self.conformer

    def statebatch2proxy(self, states: List[List]) -> npt.NDArray:
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
        return np.array(states_proxy)

    def statetorch2proxy(self, states: TensorType["batch", "state_dim"]) -> npt.NDArray:
        return self.statebatch2proxy(states.cpu().numpy())

    def statebatch2policy_gnn(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Returns an array of GNN-format policy inputs with dimensionality
        (n_states, n_atoms, 4), in which the first three columns encode atom positions,
        and the last column encodes current timestep.
        """
        policy_input = []
        for state in states:
            conformer = self.sync_conformer_with_state(state)
            positions = conformer.get_atom_positions()
            if self.remove_hs:
                positions = positions[self.non_hs]
            policy_input.append(
                np.concatenate(
                    [positions, np.full((positions.shape[0], 1), state[-1])],
                    axis=1,
                )
            )
        return np.array(policy_input)

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
