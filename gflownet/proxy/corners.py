from gflownet.proxy.base import Proxy
import numpy as np
import numpy.typing as npt


class Corners(Proxy):
    """
    It is assumed that the state values will be in the range [-1.0, 1.0].
    """
    def __init__(self, n_dim=None, mu=None, sigma=None):
        super().__init__()
        self.n_dim = n_dim
        self.mu = mu
        self.sigma = sigma
        self.mulnormal = self.setup()

    def setup(self):
        if self.sigma and self.mu and self.n_dim:
            self.mu_vec = np.repeat(self.mu, self.n_dim)
            self.cov = self.sigma * np.eye(self.n_dim)
            self.cov_det = np.linalg.det(self.cov)
            self.cov_inv = np.linalg.inv(self.cov)
            self.mulnormal_norm = 1.0 / ((2 * np.pi) ** 2 * self.cov_det) ** 0.5
            self.mulnormal = True
        else:
            self.mulnormal = False
        return self.mulnormal

    def __call__(self, states: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        args:
            states: ndarray

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

        def _mulnormal_corners(x):
            return -1.0 * self.mulnormal_norm * np.exp(
                -0.5
                * (
                    np.diag(
                        np.dot(
                            np.dot((np.abs(x) - self.mu_vec), self.cov_inv),
                            (np.abs(x) - self.mu_vec).T,
                        )
                    )
                )
            )

        if self.mulnormal:
            return _mulnormal_corners(states)
        else:
            return np.asarray([_func_corners(state) for state in states])
