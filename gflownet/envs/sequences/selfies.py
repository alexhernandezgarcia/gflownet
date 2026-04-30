"""Class to represent SELFIES molecules."""

from typing import List, Union

from torchtyping import TensorType

from gflownet.envs.sequences.base import SequenceBase
from gflownet.utils.common import tlong

SELFIES_VOCAB_SMALL = [
    "[#Branch1]",
    "[#Branch2]",
    "[#C]",
    "[#N]",
    "[=Branch1]",
    "[=Branch2]",
    "[=C]",
    "[=N]",
    "[=O]",
    "[=Ring1]",
    "[=Ring2]",
    "[=S]",
    "[B]",
    "[Br]",
    "[Branch1]",
    "[Branch2]",
    "[C]",
    "[Cl]",
    "[F]",
    "[NH1]",
    "[N]",
    "[O]",
    "[P]",
    "[Ring1]",
    "[Ring2]",
    "[S]",
]
PAD_TOKEN = "[nop]"


class Selfies(SequenceBase):
    """
    A SELFIES sequence environment whose tokens are valid SELFIES tokens.

    Human-readable states use space-separated tokens, while proxy states are emitted
    as compact SELFIES strings without whitespace or padding.
    """

    def __init__(
        self,
        selfies_vocab: List[str] | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        selfies_vocab : List[str] | None
            The list of SELFIES tokens to use as the vocabulary.
            If None (default), the small vocabulary defined in
            SELFIES_VOCAB_SMALL is used.
        """
        if selfies_vocab is None:
            selfies_vocab = SELFIES_VOCAB_SMALL
        self.selfies_vocab = selfies_vocab
        super().__init__(tokens=self.selfies_vocab, pad_token=PAD_TOKEN, **kwargs)

    def states2proxy(
        self,
        states: Union[
            List[TensorType["max_length"]],  # noqa: F821
            TensorType["batch", "max_length"],  # noqa: F821
        ],
    ) -> List[str]:
        """
        Prepare a batch of states for a SELFIES-string proxy.

        The proxy representation is the compact SELFIES string obtained by
        concatenating all non-padding tokens in the sequence.

        Parameters
        ----------
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A list containing one SELFIES string per state.
        """
        states = tlong(states, device=self.device).tolist()
        return [
            "".join(self.idx2token[idx] for idx in self._unpad(state))
            for state in states
        ]
