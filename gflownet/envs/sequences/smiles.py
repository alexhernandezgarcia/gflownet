"""Class to represent SMILES molecules."""

from typing import List, Union

import numpy as np
from torchtyping import TensorType

from gflownet.envs.sequences.base import SequenceBase
from gflownet.envs.sequences.smiles_vocab import vocab as SMILES_VOCAB
from gflownet.utils.common import tlong


class Smiles(SequenceBase):
    """
    A SMILES sequence environment whose tokens are random SMILES tokens.

    Smiles may not be a valid molecule
    """

    def __init__(
        self,
        smiles_vocab: List[str] | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        smiles_vocab : List[str] | None
            The list of SMILES tokens to use as the vocabulary.
            If None (default), the small vocabulary defined in
            SMILES_VOCAB is used.
        """
        if smiles_vocab is None:
            smiles_vocab = SMILES_VOCAB
        self.smiles_vocab = smiles_vocab
        super().__init__(tokens=self.smiles_vocab, **kwargs)

    def states2proxy(
        self,
        states: Union[
            List[TensorType["max_length"]],  # noqa: F821
            TensorType["batch", "max_length"],  # noqa: F821
        ],
    ) -> List[str]:
        """
        Prepare a batch of states for a SMILES-string proxy.

        The proxy representation is the compact SMILES string obtained by
        concatenating all non-padding tokens in the sequence.

        Parameters
        ----------
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A list containing one SMILES string per state.
        """
        states = tlong(states, device=self.device).tolist()
        return [
            "".join(self.idx2token[idx] for idx in self._unpad(state))
            for state in states
        ]
