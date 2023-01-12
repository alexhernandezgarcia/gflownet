from typing import List
from gflownet.proxy.base import Proxy
import numpy as np


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

    def __call__(self, state_list: List[List]):
        """
        args:
            state_list: list of states
        """

        def _length(x):
            if self.norm:
                return -1.0 * len(x) / self.max_seq_length
            else:
                return -1.0 * len(x)

        if self.type == "length":
            return np.asarray([_length(state) for state in state_list])
        else:
            raise NotImplementedError("self.type must be length")
