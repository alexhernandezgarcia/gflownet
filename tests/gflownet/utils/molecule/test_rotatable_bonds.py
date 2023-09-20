from rdkit import Chem

from gflownet.utils.molecule import constants
from gflownet.utils.molecule.rotatable_bonds import find_rotor_from_smile


def test_simple_ad():
    tas = find_rotor_from_smile(constants.ad_smiles)
    assert len(tas) == 4
    expected = [[0, 1, 2, 3], [0, 1, 6, 7], [1, 2, 4, 5], [1, 6, 7, 8]]
    assert tas == expected
