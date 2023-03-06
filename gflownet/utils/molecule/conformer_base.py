import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import TorsionFingerprints
from rdkit.Geometry.rdGeometry import Point3D

from gflownet.utils.molecule import constants


def get_torsion_angles_atoms_list(mol):
    return [x[0][0] for x in TorsionFingerprints.CalculateTorsionLists(mol)[0]]


def get_torsion_angles_values(conf, torsion_angles_atoms_list):
    return [
        np.float32(rdMolTransforms.GetDihedralRad(conf, *ta))
        for ta in torsion_angles_atoms_list
    ]


def get_all_torsion_angles(mol, conf):
    ta_atoms = get_torsion_angles_atoms_list(mol)
    ta_values = get_torsion_angles_values(conf, ta_atoms)
    return {k: v for k, v in zip(ta_atoms, ta_values)}


def get_dummy_ad_atom_positions():
    rmol = Chem.MolFromSmiles(constants.ad_smiles)
    rmol = Chem.AddHs(rmol)
    AllChem.EmbedMolecule(rmol)
    rconf = rmol.GetConformer()
    return rconf.GetPositions()


def get_dummy_ad_conf_base():
    pos = get_dummy_ad_atom_positions()
    conf = ConformerBase(pos, constants.ad_smiles, constants.ad_free_tas)
    return conf


class ConformerBase:
    def __init__(self, atom_positions, smiles, freely_rotatable_tas=None):
        """
        :param atom_positions: numpy.ndarray of shape [num_atoms, 3] of dtype float64
        """
        self.smiles = smiles
        self.rdk_mol = self.get_mol_from_smiles(smiles)
        self.rdk_conf = self.embed_mol_and_get_conformer(self.rdk_mol)

        self.set_atom_positions(atom_positions)

        self.freely_rotatable_tas = freely_rotatable_tas

    def __deepcopy__(self, memo):
        atom_positions = self.get_atom_positions()
        cls = self.__class__
        new_obj = cls.__new__(
            cls, atom_positions, self.smiles, self.freely_rotatable_tas
        )
        return new_obj

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

    def set_atom_positions(self, atom_positions):
        """Set atom positions of the self.rdk_conf to the input atom_positions values
        :param atom_positions: 2d numpy array of shape [num atoms, 3] with new atom positions"""
        for idx, pos in enumerate(atom_positions):
            self.rdk_conf.SetAtomPosition(idx, Point3D(*pos))

    def get_atom_positions(self):
        """
        :returns: numpy array of atom positions of shape [num_atoms, 3]
        """
        return self.rdk_conf.GetPositions()

    def get_atomic_numbers(self):
        """Get atomic numbers of the atoms as 1d numpy array
        :returns: numpy array of atomic numbers of shape [num_atoms,]"""
        atomic_numbers = [atom.GetAtomicNum() for atom in self.rdk_mol.GetAtoms()]
        return np.array(atomic_numbers)

    def get_n_atoms(self):
        return self.rdk_mol.GetNumAtoms()

    def set_torsion_angle(self, torsion_angle, value):
        rdMolTransforms.SetDihedralRad(self.rdk_conf, *torsion_angle, float(value))

    def get_all_torsion_angles(self):
        """
        :returns: a dict of all tostion angles in the molecule with their values
        """
        return get_all_torsion_angles(self.rdk_mol, self.rdk_conf)

    def get_freely_rotatable_tas_values(self):
        """
        :returns: a list of values of self.freely_rotatable_tas
        """
        return get_torsion_angles_values(self.rdk_conf, self.freely_rotatable_tas)

    def randomize_freely_rotatable_tas(self):
        """
        Uniformly randomize torsion angles defined by self.freely_rotatable_tas
        """
        for torsion_angle in self.freely_rotatable_tas:
            increment = np.random.uniform(0, 2 * np.pi)
            self.increment_torsion_angle(torsion_angle, increment)

    def increment_torsion_angle(self, torsion_angle, increment):
        """
        :param torsion_angle: tuple of 4 integers defining the torsion angle
        :param increment: a float value of the increment of the angle (in radians)
        """
        initial_value = rdMolTransforms.GetDihedralRad(self.rdk_conf, *torsion_angle)
        self.set_torsion_angle(torsion_angle, initial_value + increment)


if __name__ == "__main__":
    from tabulate import tabulate

    rmol = Chem.MolFromSmiles(constants.ad_smiles)
    rmol = Chem.AddHs(rmol)
    AllChem.EmbedMolecule(rmol)
    rconf = rmol.GetConformer()
    test_pos = rconf.GetPositions()
    initial_tas = get_all_torsion_angles(rmol, rconf)

    conf = ConformerBase(
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
    print(
        tabulate(data, headers=["torsion angle", "initial value", "randomized value"])
    )
