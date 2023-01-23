import numpy as np
import torch

from collections import defaultdict
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import TorsionFingerprints
from rdkit.Geometry.rdGeometry import Point3D

from gflownet.utils.molecule import constants
from gflownet.utils.molecule.featurizer import MolDGLFeaturizer
from gflownet.utils.molecule.conformer_base import ConformerBase


class Conformer(ConformerBase):
    def __init__(self, atom_positions, smiles, atom_types, freely_rotatable_tas=None):
        """
        :param atom_positions: numpy.ndarray of shape [num_atoms, 3] of dtype float64
        """
        super(Conformer, self).__init__(atom_positions, smiles, freely_rotatable_tas)

        self.featuraiser = MolDGLFeaturizer(atom_types)
        # dgl graph is not supposed to be consistent with rdk_conf untill it is returned via .dgl_graph
        self._dgl_graph = self.featuraiser.mol2dgl(self.rdk_mol)
        self.set_atom_positions_dgl(atom_positions)
        self.ta_to_index = defaultdict(lambda: None)

    @property
    def dgl_graph(self):
        pos = self.get_atom_positions()
        self.set_atom_positions_dgl(pos)
        return self._dgl_graph

    def set_atom_positions_dgl(self, atom_positions):
        """Set atom positions of the self.dgl_graph to the input atom_positions values
        :param atom_positions: 2d numpy array of shape [num atoms, 3] with new atom positions"""
        self._dgl_graph.ndata[constants.atom_position_name] = torch.Tensor(
            atom_positions
        )

    def apply_actions(self, actions):
        """
        Apply torsion angles updates defined by agent's actions
        :param actions: a sequence of torsion angle updates of length = number of bonds in the molecule.
        The order corresponds to the order of edges in self.dgl_graph, such that action[i] is
        an update for the torsion angle corresponding to the edge[2i]
        """
        for torsion_angle in self.freely_rotatable_tas:
            idx = self.get_ta_index_in_dgl_graph(torsion_angle)
            assert idx % 2 == 0
            # actions tensor is 2 times shorter that edges tensor (w/o reversed edges)
            idx = int(idx // 2)
            increment = actions[idx]
            self.increment_torsion_angle(torsion_angle, increment)

    def get_ta_index_in_dgl_graph(self, torsion_angle):
        """
        Get an index in the dgl graph of the first edge corresponding to the input torsion_angle
        :param torsion_angle: tuple of 4 integers defining torsion angle
        (these integers are indexes of the atoms in both self.rdk_mol and self.dgl_graph)
        :returns: int, index of the torsion_angle's edge in self.dgl_graph
        """
        if self.ta_to_index[torsion_angle] is None:
            for idx, (s, d) in enumerate(zip(*self._dgl_graph.edges())):
                if torsion_angle[1:3] == (s, d):
                    self.ta_to_index[torsion_angle] = idx
        if self.ta_to_index[torsion_angle] is None:
            raise Exception("Cannot find torsion angle {}".format(torsion_angle))
        return self.ta_to_index[torsion_angle]


if __name__ == "__main__":
    from tabulate import tabulate
    from gflownet.utils.molecule.conformer_base import get_all_torsion_angles

    rmol = Chem.MolFromSmiles(constants.ad_smiles)
    rmol = Chem.AddHs(rmol)
    AllChem.EmbedMolecule(rmol)
    rconf = rmol.GetConformer()
    test_pos = rconf.GetPositions()
    initial_tas = get_all_torsion_angles(rmol, rconf)

    conf = Conformer(
        test_pos, constants.ad_smiles, constants.ad_atom_types, constants.ad_free_tas
    )
    # check torsion angles randomisation
    conf.randomize_freely_rotatable_tas()
    conf_tas = conf.get_all_torsion_angles()
    for k, v in conf_tas.items():
        if k in conf.freely_rotatable_tas:
            assert not np.isclose(v, initial_tas[k])
        else:
            assert np.isclose(v, initial_tas[k])

    data = [[k, v1, v2] for (k, v1), v2 in zip(initial_tas.items(), conf_tas.values())]
    print(tabulate(data, headers=["torsion angle", "initial value", "conf value"]))

    # check actions are applied
    actions = (
        np.random.uniform(-1, 1, size=len(conf._dgl_graph.edges()[0]) // 2) * np.pi
    )
    conf.apply_actions(actions)
    new_conf_tas = conf.get_all_torsion_angles()
    data = [[k, v1, v2] for (k, v1), v2 in zip(conf_tas.items(), new_conf_tas.values())]
    print(tabulate(data, headers=["torsion angle", "before action", "after action"]))
    actions_dict = {
        ta: actions[conf.get_ta_index_in_dgl_graph(ta) // 2]
        for ta in conf.freely_rotatable_tas
    }
    data = [[k, a, (conf_tas[k] + a), new_conf_tas[k]] for k, a in actions_dict.items()]
    print(
        tabulate(
            data, headers=["torsion angle", "action", "init + action", "after action"]
        )
    )

    # check dgl_graph
    conf.randomize_freely_rotatable_tas()
    print("rdk pos", conf.get_atom_positions()[3])
    print(
        "_dgl pos (should differ from rdk)",
        conf._dgl_graph.ndata[constants.atom_position_name][3],
    )
    print(
        "dgl pos (should be the same as rdk)",
        conf.dgl_graph.ndata[constants.atom_position_name][3],
    )
