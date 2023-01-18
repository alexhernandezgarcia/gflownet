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


def get_torsion_angles_atoms_list(mol):
    return [x[0][0] for x in TorsionFingerprints.CalculateTorsionLists(mol)[0]]


def get_torsion_angles_values(conf, torsion_angles_atoms_list):
    return [rdMolTransforms.GetDihedralRad(conf, *ta) for ta in torsion_angles_atoms_list]


def get_all_torsion_angles(mol, conf):
    ta_atoms = get_torsion_angles_atoms_list(mol)
    ta_values = get_torsion_angles_values(conf, ta_atoms)
    return {k:v for k,v in zip(ta_atoms, ta_values)}


class Conformer():
    def __init__(self, atom_positions, smiles, atom_types):
        """
        :param atom_positions: numpy.ndarray of shape [num_atoms, 3] of dtype float64
        """
        self.rdk_mol = self.get_mol_from_smiles(smiles)
        self.rdk_conf = self.embed_mol_and_get_conformer(self.rdk_mol)
        self.featuraiser = MolDGLFeaturizer(atom_types)
        self.dgl_graph = self.featuraiser.mol2dgl(self.rdk_mol)
        self.set_atom_positions_rdk(atom_positions)
        self.set_atom_positions_dgl(atom_positions)
        self.ta_to_index = defaultdict(lambda: None)
        self.freely_rotatable_tas = ((0, 1, 2, 3), (0, 1, 6, 7))
        # self.randomize_freely_rotatable_ta()
        
    def get_mol_from_smiles(self, smiles):
        """Create RDKit molecule from SMILES string
        :param smiles: python string
        :returns: rdkit.Chem.rdchem.Mol object"""
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        return mol
    
    def embed_mol_and_get_conformer(self, mol, extra_opt=False):
        """Embed RDkit mol with a conformer and return the RDKit conformer object 
        (which is synchronized with the RDKit molecule object)
        :param mol: rdkit.Chem.rdchem.Mol object defining the molecule
        :param extre_opt: bool, if True, an additional optimisation of the conformer will be performed"""
        AllChem.EmbedMolecule(mol)
        if extra_opt:
            AllChem.MMFFOptimizeMolecule(mol, confId=0, maxIters=1000)
        return mol.GetConformer()
    
    def set_atom_positions_rdk(self, atom_positions):
        """Set atom positions of the self.rdk_conf to the input atom_positions values
        :param atom_positions: 2d numpy array of shape [num atoms, 3] with new atom positions"""
        for idx, pos in enumerate(atom_positions):
            self.rdk_conf.SetAtomPosition(idx, Point3D(*pos))
    
    def set_atom_positions_dgl(self, atom_positions):
        """Set atom positions of the self.dgl_graph to the input atom_positions values
        :param atom_positions: 2d numpy array of shape [num atoms, 3] with new atom positions"""
        self.dgl_graph.ndata[constants.atom_position_name] = torch.Tensor(atom_positions)

    def set_atom_positions(self, atom_positions):
        """
        :param atom_positions: 2d numpy array of shape [num atoms, 3] with new atom positions
        """
        self.set_atom_positions_rdk(atom_positions)
        self.set_atom_positions_dgl(atom_positions)

    def get_atom_positions(self):
        """
        :returns: numpy array of atom positions of shape [num_atoms, 3]
        """
        assert self.dgl_and_rdkit_pos_are_quial()
        return self.rdk_conf.GetPositions()
        
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
            idx = int(idx //2)
            increment = actions[idx]
            self.increment_torsion_angle(torsion_angle, increment)
        assert self.dgl_and_rdkit_pos_are_quial
    
    def randomize_freely_rotatable_ta(self):
        """
        Uniformly randomize torsion angles defined by self.freely_rotatable_tas
        """
        for torsion_angle in self.freely_rotatable_tas:
            increment = np.random.uniform(0, 2*np.pi)
            self.increment_torsion_angle(torsion_angle, increment)
        assert self.dgl_and_rdkit_pos_are_quial
    
    def increment_torsion_angle(self, torsion_angle, increment):
        """
        :param torsion_angle: tuple of 4 integers defining the torsion angle
        :param increment: a float value of the increment of the angle (in radians)
        """
        initial_value = rdMolTransforms.GetDihedralRad(self.rdk_conf, *torsion_angle)
        self.set_torsion_angle(torsion_angle, initial_value + increment)

    def set_torsion_angle(self, torsion_angle, value):
        rdMolTransforms.SetDihedralRad(self.rdk_conf, *torsion_angle, value)
        new_pos = self.rdk_conf.GetPositions()
        self.set_atom_positions_dgl(new_pos)
    
    def get_ta_index_in_dgl_graph(self, torsion_angle):
        """
        Get an index in the dgl graph of the first edge corresponding to the input torsion_angle
        :param torsion_angle: tuple of 4 integers defining torsion angle 
        (these integers are indexes of the atoms in both self.rdk_mol and self.dgl_graph)
        :returns: int, index of the torsion_angle's edge in self.dgl_graph
        """
        if self.ta_to_index[torsion_angle] is None:
            for idx, (s,d) in enumerate(zip(*self.dgl_graph.edges())):
                if torsion_angle[1:3] == (s,d):
                    self.ta_to_index[torsion_angle] = idx
        if self.ta_to_index[torsion_angle] is None:
            raise Exception("Cannot find torsion angle {}".format(torsion_angle))
        return self.ta_to_index[torsion_angle]
    
    def get_all_torsion_angles(self):
        """
        :returns: a dict of all tostion angles in the molecule with their values
        """
        return get_all_torsion_angles(self.rdk_mol, self.rdk_conf)

    def get_freely_rotatable_ta_values(self):
        """
         :returns: a list of values of self.freely_rotatable_tas
        """
        return get_torsion_angles_values(self.rdk_conf, self.freely_rotatable_tas)
    
    @property
    def dgl_and_rdkit_pos_are_quial(self):
        """
        indicator of whether self.rdk_conf and self.dgl_graph have the same atom positions
        """
        rdk_pos = torch.tensor(self.rdk_conf.GetPositions(), dtype=torch.float32)
        return self.dgl_graph.ndata[constants.atom_position_name].allclose(rdk_pos)

    def get_n_atoms(self):
        return self.rdk_mol.GetNumAtoms()


if __name__ == '__main__':
    from tabulate import tabulate

    rmol = Chem.MolFromSmiles(constants.ad_smiles)
    rmol = Chem.AddHs(rmol)
    AllChem.EmbedMolecule(rmol)
    rconf = rmol.GetConformer()
    test_pos = rconf.GetPositions()
    initial_tas = get_all_torsion_angles(rmol, rconf)

    conf = Conformer(test_pos, constants.ad_smiles, constants.ad_atom_types)
    # check torsion angles randomisation
    conf_tas = conf.get_all_torsion_angles()
    conf.randomize_freely_rotatable_ta()
    for k, v in conf_tas.items():
        if k in conf.freely_rotatable_tas:
           assert not np.isclose(v, initial_tas[k])
        else:
            assert np.isclose(v,initial_tas[k])
    
    data = [[k, v1, v2] for (k, v1), v2 in zip(initial_tas.items(), conf_tas.values())]
    print(tabulate(data, headers=["torsion angle", 'initial value', 'conf value']))
    
    # check actions are applied
    actions = np.random.uniform(-1, 1, size=len(conf.dgl_graph.edges()[0]) // 2)*np.pi
    print('actions', {ta: actions[conf.get_ta_index_in_dgl_graph(ta)//2] for ta in conf.freely_rotatable_tas}, sep='\n')
    conf.apply_actions(actions)
    new_conf_tas = conf.get_all_torsion_angles()
    data = [[k, v1, v2] for (k, v1), v2 in zip(conf_tas.items(), new_conf_tas.values())]
    print(tabulate(data, headers=["torsion angle", 'before action', 'after action']))
    
   
