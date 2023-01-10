from gflownet.proxy.base import Proxy
import numpy as np


class GridCorners(Proxy):
    def __init__(self, function):
        super().__init__()
        if function.lower() == "corners":
            self.function = self._func_corners
        elif function.lower() == "corners_floor_B":
            self.function = self._func_corners_floor_B
        elif function.lower() == "corners_floor_A":
            self.function = self._func_corners_floor_A
        elif function.lower() == "cos_n":
            self.function = self._func_cos_N
        else:
            raise NotImplementedError

    def __call__(self, inputs):
        """
        args:
            inputs: list of arrays in desired format interpretable by oracle
        returns:
            array of scores
        technically an oracle, hence used variable name energies
        """

        return np.asarray([self.function(x) for x in inputs])

    def _func_corners(self, x):
        ax = abs(x)
        return -1.0 * (
            (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + 1e-1
        )

    def _func_corners_floor_A(self, x):
        ax = abs(x)
        return -1.0 * (
            (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + 1e-3
        )

    def _func_corners_floor_B(self, x):
        ax = abs(x)
        return -1.0 * (
            (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + 1e-2
        )

    def _func_cos_N(self, x):
        ax = abs(x)
        # TODO: find out what norm is
        return -1.0 * (((np.cos(x * 50) + 1) * norm.pdf(x * 5)).prod(-1) + 0.01)
