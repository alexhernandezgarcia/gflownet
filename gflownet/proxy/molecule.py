import numpy as np
import numpy.typing as npt

from xtb.interface import Calculator, Param
from xtb.libxtb import VERBOSITY_MUTED


from gflownet.proxy.base import Proxy

class MoleculeEnergy(Proxy):
    def __init__(self):
        super().__init__()
        
    def __call__(self, states):
        # todo: probably make it parallel with mpi
        pass

    def get_energy(atom_positions: npt.NDArray[np.float32], atomic_numbers: npt.NDArray[np.int64]) -> float:
        """
        Compute energy of a molecule defined by atom_positions and atomic_numbers
        """
        calc = Calculator(Param.GFN2xTB, atomic_numbers, atom_positions)
        calc.set_verbosity(VERBOSITY_MUTED)
        res = calc.singlepoint()
        return res.get_energy()

    
