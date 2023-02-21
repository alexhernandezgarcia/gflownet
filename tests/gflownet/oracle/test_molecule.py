from gflownet.oracle.molecule import MoleculeEnergy
from gflownet.utils.molecule.conformer_base import get_dummy_ad_conf_base


def test__get_energy__computes_an_energy_for_a_dummy():
    conf = get_dummy_ad_conf_base()
    proxy = MoleculeEnergy(device="cpu", float_precision=16)

    proxy.get_energy(conf.get_atom_positions(), conf.get_atomic_numbers())
