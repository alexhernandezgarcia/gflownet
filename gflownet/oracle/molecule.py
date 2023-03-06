import numpy as np
import numpy.typing as npt
import torch

from xtb.interface import Calculator, Param, XTBException
from xtb.libxtb import VERBOSITY_MUTED

from gflownet.proxy.base import Proxy


class XTBMoleculeEnergy(Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, states_proxy):
        # todo: probably make it parallel with mpi
        return torch.tensor(
            [self.get_energy(*st) for st in states_proxy],
            dtype=self.float,
            device=self.device,
        )

    def get_energy(
        self,
        atom_positions: npt.NDArray[np.float32],
        atomic_numbers: npt.NDArray[np.int64],
    ) -> float:
        """
        Compute energy of a molecule defined by atom_positions and atomic_numbers
        """
        calc = Calculator(Param.GFN2xTB, atomic_numbers, atom_positions)
        calc.set_verbosity(VERBOSITY_MUTED)
        try:
            return calc.singlepoint().get_energy()
        except XTBException:
            return np.nan


if __name__ == "__main__":
    from gflownet.utils.molecule.conformer_base import get_dummy_ad_conf_base

    conf = get_dummy_ad_conf_base()
    proxy = XTBMoleculeEnergy()
    energy = proxy.get_energy(conf.get_atom_positions(), conf.get_atomic_numbers())
    print("energy", energy)
