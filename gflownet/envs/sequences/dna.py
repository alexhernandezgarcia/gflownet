"""
Class to represent DNA sequences.
"""
from typing import List, Union

import numpy.typing as npt
import torch.nn.functional as F
from torchtyping import TensorType

from gflownet.envs.sequences.base import SequenceBase
from gflownet.utils.common import tlong

NUCLEOBASES = ("A", "C", "T", "G")
PAD_TOKEN = "0"


class DNA(SequenceBase):
    """
    A DNA sequence environment, where the tokens are the nucleobases A, C, T and G.

    This basic implementation fully relies on the base sequence class Sequence.
    """

    def __init__(
        self,
        proxy_fmt: str = "onehot-np",
        **kwargs,
    ):
        """
        Parameters
        ----------
        proxy_fmt : str
            Specifies the proxy format. Options:
                - onehot: One-hot encoding
                - letters: The nucleobases as a list of strings
                - np or numpy: numpy, for the onehot case
                - torch or tensor: torch tensor, for the onehot case
        """
        if "letters" in proxy_fmt:
            self.states2proxy = super().states2proxy
        elif "onehot" in proxy_fmt or "one-hot" in proxy_fmt:
            self.states2proxy = self.states2proxy_onehot
            if "np" in proxy_fmt or "numpy" in proxy_fmt:
                self.proxy_np = True
            elif "torch" in proxy_fmt or "tensor" in proxy_fmt:
                self.proxy_np = False
            else:
                raise NotImplementedError(
                    "If proxy format is one-hot, it must specify "
                    "one of the following options: np, numpy, "
                    f"torch, tensor. None found in {proxy_fmt}."
                )
        else:
            raise NotImplementedError(
                "proxy_fmt must contain either onehot, one-hot or letters. "
                f"None found in {proxy_fmt}"
            )
        super().__init__(tokens=NUCLEOBASES, pad_token=PAD_TOKEN, **kwargs)

    def states2proxy_onehot(
        self,
        states: Union[
            List[TensorType["max_length"]], TensorType["batch", "max_length"]
        ],
    ) -> Union[TensorType["batch", "policy_input_dim"], npt.NDArray]:
        """
        Prepares a batch of states in "environment format" for a proxy model: states
        are one-hot encoded. If numpy is True (default), the output is converted into a
        numpy array, otherwise it remains a torch tensor.

        Example, with max_length = 5:
          - Sequence (tokens): ACGC
          - state: [1, 2, 4, 2, 0]
          - policy format:
                [0, 1, 0, 0, 0, (A)
                 0, 0, 1, 0, 0, (C)
                 0, 0, 0, 0, 1, (G)
                 0, 0, 1, 0, 0, (C)
                 1, 0, 0, 0, 0] (PAD)

        Parameters
        ----------
        states : tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A numpy array containing the one-hot encoding of all the states in the batch.
        """
        states = tlong(states, device=self.device)
        states_proxy = (
            F.one_hot(states, self.n_tokens + 1)
            .reshape(states.shape[0], -1)
            .to(self.float)
        )
        if self.proxy_np:
            return states_proxy.numpy()
        else:
            return states_proxy
