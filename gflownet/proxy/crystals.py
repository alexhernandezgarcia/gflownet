from gflownet.proxy.base import Proxy
import torch
import torch.nn as nn

import os.path as osp


class SendekMLPWrapper(Proxy):
    """
    Wrapper for MLP proxy trained on Li-ion SSB ionic conductivities calculated
    from Sendek et. al's logistic regression model

    Attributes
    ----------

    feature_set: str
        supports either "comp" or "all", with the former including only
        features that denote the composition of a crystal in the form of
        [Li_content, Natoms, e1, e2, e3...]. The latter contains one-hot
        encoded space-group values as well as the crystal structure
        paramaters [a, b, c, alpha, beta, gamma]

    path_to_proxy: str
        path to saved torch checkpoint for MLP state dictionary

    scale: bool
        whether or not the input nereds to be standardised based on
        mean and standard deviation of the dataset
    """

    def __init__(self, feature_set, path_to_proxy, scale=False, **kwargs):
        super().__init__(**kwargs)
        # TODO: assert oracle_split in ["D2_target", "D2_target_fid1", "D2_target_fid2"]
        # TODO: assert oracle_type in ["MLP"]
        if feature_set == "comp":
            self.oracle = CrystalMLP(85, [256, 256])
        elif feature_set == "all":
            self.oracle = CrystalMLP(322, [512, 1024, 1024, 512])
        self.oracle.load_state_dict(
            torch.load(osp.join(path_to_proxy, feature_set + ".ckpt"))
        )
        self.oracle.to(self.device)
        if scale:
            self.scale = {
                "mean": torch.load(osp.join(path_to_proxy, feature_set + "_mean.pt")),
                "std": torch.load(osp.join(path_to_proxy, feature_set + "_std.pt")),
            }
        else:
            self.scale = None

    def __call__(self, crystals):
        """
        Returns a vector of size [batch_size] that are calculated
        ionic conductivity values between 0 and 1
        """

        if self.scale is not None:
            crystals = (crystals - self.scale["mean"]) / self.scale["std"]
        crystals = torch.nan_to_num(crystals, nan=0.0)
        with torch.no_grad():
            scaled_ionic_conductivity = self.oracle(crystals.to(torch.float32))

        return scaled_ionic_conductivity


class CrystalMLP(nn.Module):

    """
    Skeleton code for an MLP that can be used to train
    MLPs on ionic conductivity values of
    """

    def __init__(self, in_feat, hidden_layers):
        super(CrystalMLP, self).__init__()
        self.nn_layers = []
        self.modules = []

        for i in range(len(hidden_layers)):
            if i == 0:
                self.nn_layers.append(nn.Linear(in_feat, hidden_layers[i]))
            else:
                self.nn_layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.modules.append(self.nn_layers[-1])
            self.nn_layers.append(nn.BatchNorm1d(hidden_layers[i]))
        self.nn_layers.append(nn.Linear(hidden_layers[-1], 1))
        self.modules.append(self.nn_layers[-1])
        self.nn_layers = nn.ModuleList(self.nn_layers)
        self.hidden_act = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(p=0.5)
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        for l, layer in enumerate(self.nn_layers):
            if isinstance(layer, nn.BatchNorm1d):
                continue
            x = layer(x)
            if l == len(self.nn_layers) - 1:
                x = self.final_act(x)
            if l % 2 == 1:
                x = self.hidden_act(x)

        return x


if __name__ == "__main__":
    tmp = SendekMLPWrapper(
        "all",
        "/Users/divya-sh/Documents/gflownet/data/crystals",
        device="cpu",
        float_precision=32,
    )
