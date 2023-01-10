from gflownet.proxy.base import Proxy
from clamp_common_eval.defaults import get_test_oracle

import numpy as np

class AMPOracleWrapper(Proxy):
    def __init__(self, oracle_split, oracle_type, oracle_features, medoid_oracle_norm, device):
        super().__init__()
        self.oracle = get_test_oracle(oracle_split, 
                                        model=oracle_type, 
                                        feature=oracle_features, 
                                        dist_fn="edit", 
                                        norm_constant=medoid_oracle_norm)
        self.oracle.to(device)

    def __call__(self, sequences, batch_size=256):
        scores = []
        for i in range(int(np.ceil(len(sequences) / batch_size))):
            s = self.oracle.evaluate_many(sequences[i*batch_size:(i+1)*batch_size])
            if type(s) == dict:
                scores += s["confidence"][:, 1].tolist()
            else:
                scores += s.tolist()
        return np.float32(scores)
