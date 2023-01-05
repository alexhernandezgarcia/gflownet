from gflownet.proxy.base import Proxy
import numpy as np


class Torus(Proxy):
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
            return (
                -1 * (np.sum(np.sin(x[0::2])) + np.sum(np.cos(x[1::2])) + len(x)) ** 3
            )

        def _func_sin_cos_cube_norm(x):
            norm = (len(x) * 2) ** 3
            return (-1.0 / norm) * (
                np.sum(np.sin(x[0::2])) + np.sum(np.cos(x[1::2])) + len(x)
            ) ** 3

        if self.normalize:
            _func = _func_sin_cos_cube_norm
        else:
            _func = _func_sin_cos_cube

        return np.asarray([_func(x) for x in x_list])
