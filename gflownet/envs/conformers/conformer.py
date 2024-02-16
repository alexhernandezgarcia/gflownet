import copy
from typing import List, Optional, Tuple, Union

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

PREDEFINED_SMILES = [
    "O=C(c1ccccc1)c1ccc2c(c1)OCCOCCOCCOCCO2",
    "O=S(=O)(NN=C1CCCCCC1)c1ccc(Cl)cc1",
    "O=C(NC1CCCCC1)N1CCN(C2CCCCC2)CC1",
    "O=C(COc1ccc(Cl)cc1[N+](=O)[O-])N1CCCCCC1",
    "O=C(Nc1ccc(N2CCN(C(=O)c3ccccc3)CC2)cc1)c1cccs1",
    "O=[N+]([O-])/C(C(=C(Cl)Cl)N1CCN(Cc2ccccc2)CC1)=C1\\NCCN1Cc1ccc(Cl)nc1",
    "O=C(CSc1nnc(C2CC2)n1-c1ccccc1)Nc1ccc(N2CCOCC2)cc1",
    "O=C(Nc1ccccn1)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)Nc1ccccn1",
    "O=C(NCCc1nnc2ccc(NCCCN3CCOCC3)nn12)c1ccccc1F",
    "O=C(CSc1cn(CCNC(=O)c2cccs2)c2ccccc12)NCc1ccccc1",
    "O=C(CCC(=O)N(CC(=O)NC1CCCCC1)Cc1cccs1)Nc1ccccn1",
    "S=C(c1ccc2c(c1)OCO2)N1CCOCC1",
    "O=Cc1cnc(N2CCN(c3ccccc3)CC2)s1",
    "O=[N+]([O-])c1ccc(NCc2ccc(Cl)cc2)nc1",
    "O=C(Nc1cc(C(F)(F)F)ccc1N1CCCCC1)c1ccncc1",
    "O=C(Nc1nnc(-c2ccccc2Cl)s1)C1CCN(S(=O)(=O)c2ccc(Cl)cc2)CC1",
    "O=C(CCNC(=O)c1ccccc1Cl)Nc1nc2ccccc2s1",
    "O=C(CCC(=O)Nc1ccccc1Cl)N/N=C/c1ccccc1",
    "O=C(COc1ccccc1)Nc1ccccc1C(=O)NCc1ccco1",
    "O=C(CNC(=O)c1cccs1)NCC(=O)OCc1ccc(Cl)cc1Cl",
    "O=C(NCc1ccccc1)c1onc(CSc2ccc(Cl)cc2)c1C(=O)NCC1CC1",
    "O=C(CN(C(=O)CCC(=O)Nc1ccccn1)c1ccc2c(c1)OCO2)NCc1ccco1",
    "O=[N+]([O-])c1ccc(N2CCNCC2)c(Cl)c1",
    "O=[N+]([O-])c1ccccc1S(=O)(=O)N1CCCCC1",
    "N#C/C(=C\\N1CCN(Cc2ccc3c(c2)OCO3)CC1)c1nc2ccccc2s1",
    "O=C(NNc1ccc([N+](=O)[O-])cc1)c1ccccc1Cl",
    "O=C(OCc1ccccc1Cl)c1ccccc1C(=O)c1ccccc1",
    "O=C(CN(c1cccc(C(F)(F)F)c1)S(=O)(=O)c1ccccc1)N1CCOCC1",
    "O=c1[nH]c2cc3c(cc2cc1CN(Cc1cccnc1)Cc1nnnn1Cc1ccco1)OCCO3",
    "O=C(CCNC(=O)C1CCN(S(=O)(=O)c2ccccc2)CC1)NC1CC1",
    "O=C(CN(c1ccc(F)cc1)S(=O)(=O)c1ccccc1)NCCSC1CCCCC1",
    "C=CCn1c(CSCc2ccccc2)nnc1SCC(=O)N1CCN(c2ccccc2)CC1",
    "C=COCCNC(=S)N1CCOCCOCCN(C(=S)NCCOC=C)CCOCC1",
    "O=S(=O)(c1cccc(Cl)c1Cl)N1CCCCC1",
    "O=C(Cn1ccccc1=O)c1cccs1",
    "FC(F)Sc1ccc(Nc2ncnc3nc[nH]c23)cc1",
    "O=c1[nH]c(SCCOc2ccccc2F)nc2ccccc12",
    "O=C(/C=C/c1ccccc1Cl)NCCN1CCOCC1",
    "N#CCCN1CCN(S(=O)(=O)c2ccc(S(=O)(=O)NC3CC3)cc2)CC1",
    "O=C(CSc1nnc(Cc2cccs2)n1-c1ccccc1)Nc1ccc2c(c1)OCCO2",
    "O=C(CNC(=O)c1ccc(N2C(=O)c3ccccc3C2=O)cc1)OCC(=O)c1ccccc1",
    "O=C(CSc1nnc(CNC(=O)c2c(F)cccc2Cl)o1)NCc1ccc2c(c1)OCO2",
    "O=C(CNC(=S)N(Cc1ccc(F)cc1)C1CCCCC1)NCCN1CCOCC1",
    "C=CCn1c(CCNC(=O)c2cccs2)nnc1SCC(=O)Nc1cccc(F)c1",
    "Clc1cccc(CN2CCCCCC2)c1Cl",
    "O=C(Nc1cc(=O)c2ccccc2o1)N1CCCCC1",
    "O=S(=O)(c1ccccc1)c1nc(-c2ccco2)oc1N1CCOCC1",
    "O=C(CC1CCCCC1)NCc1ccco1",
    "O=C(c1cccc([N+](=O)[O-])c1)N1CCCN(C(=O)c2cccc([N+](=O)[O-])c2)CC1",
    "Fc1ccccc1OCCCCCN1CCCC1",
    "O=C(c1ccc(S(=O)(=O)NCc2ccco2)cc1)N1CCN(Cc2ccc3c(c2)OCO3)CC1",
    "N#Cc1ccc(NC(=O)COC(=O)CNC(=O)C2CCCCC2)cc1",
    "O=C(CCC(=O)OCC(=O)c1ccc(-c2ccccc2)cc1)Nc1cccc(Cl)c1",
    "O=C(COC(=O)CCC(=O)c1cccs1)Nc1ccc(S(=O)(=O)N2CCOCC2)cc1",
    "O=C(COC(=O)CCCNC(=O)c1ccc(Cl)cc1)NCc1ccccc1Cl",
    "C1CCC([NH2+]C2=NCCC2)CC1",
    "N#Cc1ccccc1S(=O)(=O)Nc1ccc2c(c1)OCCO2",
    "O=[N+]([O-])c1ccccc1S(=O)(=O)N1CCN(c2ccccc2)CC1",
    "O=S(=O)(NCc1ccccc1Cl)c1ccc(-n2cccn2)cc1",
    "O=C(CNS(=O)(=O)c1cccc2nsnc12)NC1CCCCC1",
    "O=C(c1cccc([N+](=O)[O-])c1)n1nc(-c2ccccc2)nc1NCc1ccccc1",
    "O=C(CN1CCN(c2ccccc2)CC1)NC(=O)NCc1ccco1",
    "O=C(CCCn1c(=O)c2ccccc2n(Cc2ccccc2)c1=O)NCc1ccco1",
    "O=C(COc1ccc(Cl)cc1)NCc1nnc(SCC(=O)N2CCCCCC2)o1",
    "O=C(NCc1ccccc1)c1onc(CSc2ccccn2)c1C(=O)NCc1ccccc1",
    "C=CCN(c1cccc(C(F)(F)F)c1)S(=O)(=O)c1cccc(C(=O)OCC(=O)Nc2ccccc2)c1",
    "O=S(=O)(N1CCCCCC1)N1CC[NH2+]CC1",
    "O=C1c2ccccc2C(=O)N1Cc1nn2c(-c3ccc(Cl)cc3)nnc2s1",
    "O=C(CN1C(=O)NC2(CCCC2)C1=O)Nc1ccc(F)c(F)c1F",
    "O=C(Cc1n[nH]c(=O)[nH]c1=O)N/N=C/c1ccccc1",
    "O=C(NCCSc1ccc(Cl)cc1)c1ccco1",
    "O=C(CN1CCN(Cc2ccccc2Cl)CC1)N/N=C/c1ccco1",
    "O=C(Nc1ccc(-c2csc(Nc3cccc(C(F)(F)F)c3)n2)cc1)c1cccc(C(F)(F)F)c1",
    "O=C(CCCCCN1C(=O)c2cccc3cccc(c23)C1=O)NCc1ccco1",
    "O=C(NCCCn1ccnc1)/C(=C\\c1cccs1)NC(=O)c1cccs1",
    "O=C(COC(=O)COc1ccccc1[N+](=O)[O-])Nc1ccc(S(=O)(=O)N2CCCCC2)cc1",
    "O=C(NCCCN1CCCC1=O)c1cc(NS(=O)(=O)c2ccc(F)cc2)cc(NS(=O)(=O)c2ccc(F)cc2)c1",
    "Clc1ccc(N2CCN(c3ncnc4c3oc3ccccc34)CC2)cc1",
    "O=C(Nc1c(Cl)ccc2nsnc12)c1cccnc1",
    "c1nc(COc2nsnc2N2CCOCC2)cs1",
    "O=C(C1CC1)N1CCN=C1SCc1ccccc1",
    "O=C(Cc1ccccc1)OCC[NH+]1CCOCC1",
    "O=C(CCSc1ccccc1)NCc1cccnc1",
    "O=C(CNC(=O)c1ccc(F)cc1)N/N=C/c1cn[nH]c1-c1ccccc1",
    "O=C1CCN(CCc2ccccc2)CCN1[C@H](CSc1ccccc1)Cc1ccccc1",
    "O=C(CCCCCn1c(=S)[nH]c2ccc(N3CCOCC3)cc2c1=O)NCc1ccc(Cl)cc1",
    "O=C(CCC(=O)OCCCC(F)(F)C(F)(F)F)NC1CCCCC1",
    "O=C(CN(Cc1ccco1)C(=O)CNS(=O)(=O)c1ccc(F)cc1)NCc1ccco1",
    "O=c1[nH]c(N2CCN(c3ccccc3)CC2)nc2c1CCC2",
    "O=C(Nc1ccc2c(c1)OCO2)c1cccs1",
    "O=C(Nc1cccc2ccccc12)N1CCN(c2ccccc2)CC1",
    "O=C(NC(=S)Nc1ccccn1)c1ccccc1",
    "O=C(Nc1ccc(Cl)cc1)c1cccc(S(=O)(=O)Nc2ccccn2)c1",
    "O=C(COc1cnc2ccccc2n1)NCCC1=CCCCC1",
    "O=C(NCCN1CCOCC1)c1ccc(/C=C2\\Sc3ccccc3N(Cc3ccc(F)cc3)C2=O)cc1",
    "O=C(NCCc1cccc(Cl)c1)c1ccc(OC2CCN(Cc3ccccn3)CC2)cc1",
    "C=CC[NH2+]CCOCCOc1ccccc1-c1ccccc1",
    "O=C(COC(=O)c1ccccc1NC(=O)c1ccco1)NCCC1=CCCCC1",
    "O=C(CNC(=S)N(Cc1ccccc1Cl)C1CCCC1)NCCCN1CCOCC1",
    "O=c1c2ccccc2nnn1Cc1ccccc1Cl",
    "S=C(Nc1ccccc1)N1CCCCCCC1",
    "O=C(Cn1ccc([N+](=O)[O-])n1)N1CCCc2ccccc21",
    "O=C(NS(=O)(=O)N1CCOCC1)C1=C(N2CCCC2)COC1=O",
    "O=C(CCCn1c(=O)[nH]c2ccsc2c1=O)NC1CCCCC1",
    "O=C(Cc1ccc(Cl)cc1)Nc1ccc(S(=O)(=O)Nc2ncccn2)cc1",
    "O=C1COc2ccc(C(=O)COC(=O)CCSc3ccccc3)cc2N1",
    "O=C(Nc1cc(F)cc(F)c1)c1ccc(NCCC[NH+]2CCCCCC2)c([N+](=O)[O-])c1",
    "O=C(CCCn1c(=O)[nH]c2cc(Cl)ccc2c1=O)NCCCN1CCN(c2ccc(F)cc2)CC1",
    "O=C(NCCN1CCN(C(=O)C(c2ccccc2)c2ccccc2)CC1)C(=O)Nc1ccccc1",
    "O=C(NCCCN1CCN(CCCNC(=O)c2ccc3c(c2)OCO3)CC1)c1ccc2c(c1)OCO2",
]


class Conformer(ContinuousTorus):
    """
    Extension of continuous torus to conformer generation. Based on AlanineDipeptide,
    but accepts any molecule (defined by SMILES and freely rotatable torsion angles).
    """

    def __init__(
        self,
        smiles: Union[str, int],
        n_torsion_angles: Optional[int] = 2,
        torsion_indices: Optional[List[int]] = None,
        policy_type: str = "mlp",
        remove_hs: bool = True,
        **kwargs,
    ):
        if torsion_indices is None:
            # We hard code default torsion indices to preserve backward compatibility.
            if smiles == "CC(C(=O)NC)NC(=O)C" and n_torsion_angles == 2:
                torsion_indices = [0, 1]
            elif smiles == "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O" and n_torsion_angles == 2:
                torsion_indices = [1, 2]
            elif smiles == "O=C(c1ccc2n1CCC2C(=O)O)c3ccccc3" and n_torsion_angles == 2:
                torsion_indices = [0, 1]
            elif n_torsion_angles == -1:
                torsion_indices = None
            else:
                torsion_indices = list(range(n_torsion_angles))

        if isinstance(smiles, int):
            smiles = PREDEFINED_SMILES[smiles]

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
