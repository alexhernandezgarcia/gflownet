from gflownet.proxy.base import Proxy
from clamp_common_eval.defaults import get_test_oracle
import torch
import numpy as np


class AMPOracleWrapper(Proxy):
    def __init__(
        self,
        oracle_split,
        oracle_type,
        oracle_features,
        dist_fn,
        medoid_oracle_norm,
        cost,
        **kwargs
    ):
        super().__init__(**kwargs)
        # TODO: assert oracle_split in ["D2_target", "D2_target_fid1", "D2_target_fid2"]
        # TODO: assert oracle_type in ["MLP"]
        self.oracle = get_test_oracle(
            oracle_split,
            model=oracle_type,
            feature=oracle_features,
            dist_fn=dist_fn,
            norm_constant=medoid_oracle_norm,
        )
        self.oracle.to(self.device)
        self.cost = cost

    def __call__(self, sequences, batch_size=256):
        """
        oracle.evaluate_many() returns a dictionary
            confidence: 2D array (batch, 2) where confidence[i][k] is the probability of the ith element being part of category "k"
            prediction: 1D array, essentially argmax(probability)
            entropy: 1D array, uncertainty in the prediction
        """
        scores = []
        for i in range(int(np.ceil(len(sequences) / batch_size))):
            s = self.oracle.evaluate_many(
                sequences[i * batch_size : (i + 1) * batch_size]
            )
            if type(s) == dict:
                # regressive task is score of the peptide being anti-microbial, ie, in category 1 (not category 0)
                scores += s["confidence"][:, 1].tolist()
            else:
                scores += s.tolist()
        return torch.tensor(scores, device=self.device, dtype=self.float)
