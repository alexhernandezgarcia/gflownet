from src.gflownet.proxy.base import Proxy
import numpy as np


class Torus2D(Proxy):
    def __init__(self, normalize):
        super().__init__()
        self.normalize = normalize

    def __call__(self, x_list):
        """
        args:
            x_list: list of arrays in desired format interpretable by oracle
        returns:
            list of scores
        technically an oracle, hence used variable name energies
        """

        def _func_sin_cos_cube(x):
            return -1.0 * ((np.sin(x[0]) + np.cos(x[1]) + 2) ** 3)

        def _func_sin_cos_cube_norm(x):
            return (-1.0 / 64) * ((np.sin(x[0]) + np.cos(x[1]) + 2) ** 3)

        if self.normalize:
            _func = _func_sin_cos_cube_norm
        else:
            _func = _func_sin_cos_cube

        return np.asarray([_func(x) for x in x_list])
