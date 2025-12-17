import os
import pickle
from typing import Tuple

import torch
import torch.nn as nn

from gflownet.proxy.iam.scenario_scripts.Scenario_Datasets import witch_proc_data


def load_datasets(filename="datasets.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_datasets(full_dataset, filename="datasets.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(full_dataset, f)


class residual_block(nn.Module):
    def __init__(self, width, dim_block, activation, normalization, dropout=0.0):
        super(residual_block, self).__init__()
        layers = []
        for i in range(dim_block - 1):
            layers.append(nn.Linear(width, width))
            if i == 0:
                if normalization == "batch":
                    layers.append(nn.BatchNorm1d(width))
                elif normalization == "layer":
                    layers.append(nn.LayerNorm(width))
                elif normalization == "none":
                    pass
                else:
                    raise ValueError(f"Unsupported normalization type: {normalization}")
            layers.append(activation)
            layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "selu":
        return nn.SELU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "None" or activation is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


class fairy_model(nn.Module):
    def __init__(
        self,
        variables_dim,
        variables_names,
        subsidies_dim,
        subsidies_names,
        num_blocks,
        dim_block,
        width_block,
        activation,
        normalization="batch",
        dropouts=0.0,
        probabilistic=True,
        scaling="original",
    ):
        super(fairy_model, self).__init__()

        self.variables_dim = variables_dim
        self.variables_names = variables_names
        self.subsidies_dim = subsidies_dim
        self.subsidies_names = subsidies_names
        self.dim_block = dim_block
        self.width_block = width_block
        self.activation = get_activation(activation)
        self.normalization = normalization
        self.dropouts = dropouts
        self.num_blocks = num_blocks
        self.probabilistic = probabilistic

        self.scaling = scaling
        if scaling not in ["original", "normalization", "maxscale", "maxmin"]:
            raise ValueError(
                "Scaling type must be either original, normalization, or maxscale\n Unknown scaling type: {}".format(
                    scaling
                )
            )

        blocks = []

        blocks.append(
            nn.Linear(self.variables_dim + self.subsidies_dim, self.width_block)
        )
        for _ in range(self.num_blocks):
            blocks.append(
                residual_block(
                    self.width_block,
                    self.dim_block,
                    self.activation,
                    self.normalization,
                    self.dropouts,
                )
            )

        self.net = nn.Sequential(*blocks)

        temp_variables_layers = []
        temp_variables_layers.append(nn.Linear(self.width_block, self.variables_dim))
        if self.scaling in ["original", "maxscale", "maxmin"]:
            temp_variables_layers.append(nn.ReLU())
        self.variables_layer = nn.Sequential(*temp_variables_layers)
        if self.probabilistic:
            self.confidence_layer = nn.Linear(self.width_block, self.variables_dim)

    def forward(self, variables_current, subsidies_current):
        stack = torch.cat((variables_current, subsidies_current), dim=1)
        mlp_out = self.net(stack)

        # Final output layer
        variables_next = variables_current + self.variables_layer(mlp_out)
        if self.probabilistic:
            confidence = self.confidence_layer(mlp_out)
            return variables_next, confidence
        else:
            return variables_next


def initialize_fairy() -> Tuple[fairy_model, witch_proc_data]:
    model_filename = "gflownet/proxy/iam/scenario_data/fairy_state_dict.pth"
    dataset_filename = (
        "gflownet/proxy/iam/scenario_data/witch_scenario_maxmin_dataset.pkl"
    )

    scaling_type = "maxmin"  #

    if os.path.exists(dataset_filename):
        scen_data = load_datasets(dataset_filename)
        print("Dataset loaded from file.")
    else:
        scen_data = witch_proc_data(
            subsidies_parquet="gflownet/proxy/iam/scenario_data/subsidies_df.parquet",
            variables_parquet="gflownet/proxy/iam/scenario_data/variables_df.parquet",
            keys_parquet="gflownet/proxy/iam/scenario_data/keys_df.parquet",
            scaling_type=scaling_type,
            with_cuda=False,
            drop_columns=[
                "COST_enduse_BIOMASS",
                "COST_enduse_COAL",
                "COST_enduse_COAL_abated",
                "COST_enduse_GAS",
                "COST_enduse_OIL",
                "EMI_net_emission_PERMITS",
            ],
        )
        save_datasets(scen_data, dataset_filename)
        print("Dataset created and saved to file.")

    num_blocks = 4
    dim_block = 2
    width_block = 256
    activation = "gelu"
    normalization = "batch"
    dropouts = 0.05

    variables_dim = len(scen_data.variables_names)
    subsidies_dim = len(scen_data.subsidies_names)

    fairy = fairy_model(
        variables_dim=variables_dim,
        variables_names=scen_data.variables_names,
        subsidies_dim=subsidies_dim,
        subsidies_names=scen_data.subsidies_names,
        num_blocks=num_blocks,
        dim_block=dim_block,
        width_block=width_block,
        activation=activation,
        normalization=normalization,
        dropouts=dropouts,
        probabilistic=False,
        scaling=scaling_type,
    )

    fairy.load_state_dict(torch.load(model_filename, map_location="cpu"))

    return fairy, scen_data
