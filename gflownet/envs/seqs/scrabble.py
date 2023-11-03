"""
Class to represent a Scrabble-inspired environment.
"""
from gflownet.envs.seqs.sequence import Sequence
from gflownet.utils.scrabble.utils import read_alphabet

PAD_TOKEN = "0"


class Scrabble(Sequence):
    """
    A Scrabble-inspired environment based on the Sequence, where the tokens are the
    letters in the English alphabet.

    This basic implementation fully relies on the base sequence class Sequence.
    """

    def __init__(
        self,
        **kwargs,
    ):
        alphabet_dict = read_alphabet()
        super().__init__(tokens=alphabet_dict.keys(), pad_token=PAD_TOKEN, **kwargs)

    def states2proxy(
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: the batch is
        simply converted into a tensor of indices.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        A list containing all the states in the batch, represented themselves as lists.
        """
        return tlong(states, device=self.device)
