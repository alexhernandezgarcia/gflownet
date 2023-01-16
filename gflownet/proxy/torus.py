from gflownet.proxy.base import Proxy
import numpy as np
import numpy.typing as npt


class Torus(Proxy):
    def __init__(self, normalize):
        super().__init__()
        self.normalize = normalize

    def __call__(self, states: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        args:
            states: ndarray
        returns:
            list of scores
        technically an oracle, hence used variable name energies
        """

        def _func_sin_cos_cube(x):
            return (
                -1
                * (
                    np.sum(np.sin(x[:, 0::2]), axis=1)
                    + np.sum(np.cos(x[:, 1::2]), axis=1)
                    + x.shape[1]
                )
                ** 3
            )

        def _func_sin_cos_cube_norm(x):
            norm = (x.shape[1] * 2) ** 3
            return (-1.0 / norm) * (
                np.sum(np.sin(x[:, 0::2]), axis=1)
                + np.sum(np.cos(x[:, 1::2]), axis=1)
                + x.shape[1]
            ) ** 3

        if self.normalize:
            return _func_sin_cos_cube_norm(states)
        else:
            return _func_sin_cos_cube(states)
