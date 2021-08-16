from torch._C import device
import torch.nn as nn
from chemprop.args import TrainArgs


class AptamerLSTM(nn.Module):
    """
    Encodes protein amino acid sequences using multiple layers of 1D convolutions.
    1) Each character in the AA sequence is embedded resulting in a (input_length x vocab_size x embedding_depth)
    sized matrix.
    2) The matrix then goes through 3 1D convolutional layers of increasing depth.
    3) A Maxpooling Layer along the time dimension encodes the feature maps, effectively indicating
    the presence or lack thereof for each of the feature detectors in the final layer. This means
    that if a feature is found multiple times in the sequence. The only information retained is that
    it was found at least once in the sequence.


    Args:
        peptide_vocab_size (int): amino acid vocabulary size. Used for embedding layer
        embedding_depth (int): Embedding Vector depth for protein embedding layer
        hidden_depth (int): # of feature maps generated in convolutional layers
        kernel_size (int): 1D kernel window length.

    Forward Output:
        (batch x hidden_depth * 3) sized Torch Tensor

    Adapted from: https://github.com/hkmztrk/DeepDTA
    Paper: DeepDTA: deep drug--target binding affinity prediction, BMC Bioinformatics, 2018
    """

    # 5 layers, a hidden size of 32, 10 classes, and a learning rate of 3.45e–4
    def __init__(self, args: TrainArgs, vocab_size=4):
        super(AptamerLSTM, self).__init__()
        self.layers = args.lstm_layers
        self.hidden_size = args.lstm_hidden_size
        self.vocab_size = vocab_size
        self.embedding_dim = args.lstm_embedding_dim
        self.device = args.device
        self.smiles_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(
            hidden_size=self.hidden_size, num_layers=self.layers, input_size=self.embedding_dim
        )  # (seq_len, batch, input_size)

    def forward(self, x):
        smiles, lengths = x
        # lengths.to(self.device)
        embedded_padded_smiles = self.smiles_embedding(smiles.to(self.device))

        lstm_input = nn.utils.rnn.pack_padded_sequence(
            embedded_padded_smiles, lengths, batch_first=True, enforce_sorted=False
        )

        # x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(lstm_input)
        x = h_n.view(-1, self.layers * self.hidden_size)
        # seq_unpacked, lens_unpacked = nn.utils.rnn.pad_packed_sequence(lstm_output)

        return x
