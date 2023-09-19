from rdkit import Chem

from gflownet.utils.molecule import constants
from gflownet.utils.molecule.rotatable_bonds import find_rotor_from_smiles


def test_simple_ad():
    tas = find_rotor_from_smiles(constants.ad_smiles)
    assert len(tas) == 7
    expected = [[0, 1, 2, 3], [0, 1, 6, 7], [1, 2, 4, 5], [1, 6, 7, 8], 
                [2, 4, 5, 15], [6, 7, 9, 19], [10, 0, 1, 13]]
    assert tas == expected
