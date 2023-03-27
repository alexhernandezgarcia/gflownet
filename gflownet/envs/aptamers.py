"""
Classes to represent aptamers environments
"""
import itertools
import time
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
import time
from gflownet.utils.sequence.aptamers import NUCLEOTIDES
from gflownet.envs.sequence import Sequence


class Aptamers(Sequence):
    """
    Aptamer sequence environment
    """

    def __init__(
        self,
        **kwargs,
    ):
        special_tokens = ["[PAD]", "[EOS]"]
        self.vocab = NUCLEOTIDES + special_tokens
        super().__init__(
            **kwargs,
            special_tokens=special_tokens,
        )

        if (
            hasattr(self, "proxy")
            and self.proxy is not None
            and hasattr(self.proxy, "setup")
        ):
            self.proxy.setup(self.max_seq_length)

    def make_train_set(
        self,
        ntrain,
        oracle=None,
        seed=168,
        output_csv=None,
    ):
        """
        Constructs a randomly sampled train set.

        Args
        ----
        ntest : int
            Number of test samples.

        seed : int
            Random seed.

        output_csv: str
            Optional path to store the test set as CSV.
        """
        samples_dict = oracle.initializeDataset(
            save=False, returnData=True, customSize=ntrain, custom_seed=seed
        )
        energies = samples_dict["energies"]
        samples_mat = samples_dict["samples"]
        state_letters = oracle.numbers2letters(samples_mat)
        state_ints = [
            "".join([str(el) for el in state if el > 0]) for state in samples_mat
        ]
        if isinstance(energies, dict):
            energies.update({"samples": state_letters, "indices": state_ints})
            df_train = pd.DataFrame(energies)
        else:
            df_train = pd.DataFrame(
                {"samples": state_letters, "indices": state_ints, "energies": energies}
            )
        if output_csv:
            df_train.to_csv(output_csv)
        return df_train
