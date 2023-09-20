import pytest
from rdkit import Chem

from gflownet.utils.molecule import constants
from gflownet.utils.molecule.rotatable_bonds import find_rotor_from_smiles, is_hydrogen_ta


def test_simple_ad():
    tas = find_rotor_from_smiles(constants.ad_smiles)
    assert len(tas) == 7
    expected = [[0, 1, 2, 3], [0, 1, 6, 7], [1, 2, 4, 5], [1, 6, 7, 8], 
                [10, 0, 1, 2], [2, 4, 5, 15], [6, 7, 9, 19]]
    assert tas == expected

@pytest.mark.parametrize(
        'ta, expected_flag',
        [
           ([0, 1, 2, 3], False),
           ([0, 1, 6, 7], False),
           ([1, 2, 4, 5], False),
           ([1, 6, 7, 8], False),
           ([2, 4, 5, 15], True),
           ([6, 7, 9, 19], True),
           ([10, 0, 1, 2], True) 
        ]
)
def test_is_hydrogen_ta(ta, expected_flag):
    mol = Chem.MolFromSmiles(constants.ad_smiles)
    mol = Chem.AddHs(mol)
    assert is_hydrogen_ta(mol, ta) == expected_flag

def test_number_tas():
    smiles = 'CCCc1nnc(NC(=O)COc2ccc3c(c2)OCO3)s1'
    expected = 8
    tas = find_rotor_from_smiles(smiles)
    assert len(tas) == expected 


