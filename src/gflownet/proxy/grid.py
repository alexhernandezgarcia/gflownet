from src.gflownet.proxy.base import Proxy
import numpy as np


class GridCorners(Proxy):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        """
        args:
            inputs: list of arrays in desired format interpretable by oracle
        returns:
            list of scores
        technically an oracle, hence used variable name energies
        """

        def _func_corners(x):
            ax = abs(x)
            energies = -1.0 * (
                (ax > 0.5).prod(-1) * 0.5
                + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2
                + 1e-1
            )
            return energies

        return np.asarray([_func_corners(x) for x in inputs])
