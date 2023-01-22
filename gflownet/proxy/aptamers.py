from gflownet.proxy.base import Proxy
import numpy as np
import numpy.typing as npt


class Aptamers(Proxy):
    """
    DNA Aptamer oracles
    """

    def __init__(self, oracle_id, norm):
        super().__init__()
        self.type = oracle_id
        self.norm = norm

    def setup(self, max_seq_length, norm=True):
        self.max_seq_length = max_seq_length

    def __call__(self, states: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        args:
            states : ndarray
        """

        def _length(x):
            if self.norm:
                return -1.0 * np.sum(x, axis=1) / self.max_seq_length
            else:
                return -1.0 * np.sum(x, axis=1)

        if self.type == "length":
            return _length(states)
        else:
            raise NotImplementedError("self.type must be length")
