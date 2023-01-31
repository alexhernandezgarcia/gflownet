from gflownet.proxy.base import Proxy
import numpy as np
import torch


class AMPDummy(Proxy):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

    def __call__(self, sequences):
        # scores = []
        scores = list(map(lambda x: x.count("A"), sequences))

        # for i in range(len(sequences)):
        # scores.append(sequences[i].count("A"))
        return np.float32(scores)


class AMPDummyPair(Proxy):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

    def __call__(self, sequences):
        """
        reward = #pairs(KL) + pairs(RW)
        """
        # sequences = map(lambda x: torch.tensor(x), sequences)
        # sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=-1)
        # count_pair_KL = torch.sum(sequences[:-1] == 8, axis=1) * torch.sum(sequences[1:] == 9, axis=1)
        # count_pair_RW = torch.sum(sequences[:, :-1] == 14) * torch.sum(sequences[:, 1:] == 18)
        # score = count_pair_KL + count_pair_RW
        # return score
        scores = list(map(lambda x: x.count("KL") + x.count("RW"), sequences))
        # scores = []
        # for i in range(len(sequences)):
        #     val = sequences[i].count("KL") + sequences[i].count("RW")
        #     scores.append(val)
        return np.float32(scores)
