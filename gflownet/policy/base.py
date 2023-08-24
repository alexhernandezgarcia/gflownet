import torch
from omegaconf import OmegaConf
from torch import nn


class Policy:
    def __init__(self, config, env, device, float_precision, base=None):
        # Device and float precision
        self.device = device
        self.float = float_precision
        # Input and output dimensions
        self.state_dim = env.policy_input_dim
        self.fixed_output = torch.tensor(env.fixed_policy_output).to(
            dtype=self.float, device=self.device
        )
        self.random_output = torch.tensor(env.random_policy_output).to(
            dtype=self.float, device=self.device
        )
        self.output_dim = len(self.fixed_output)
        # Optional base model
        self.base = base

        self.parse_config(config)
        self.instantiate()

    def parse_config(self, config):
        # If config is null, default to uniform
        if config is None:
            config = OmegaConf.create()
            config.type = "uniform"
        if "shared_weights" in config:
            self.shared_weights = config.shared_weights
        else:
            self.shared_weights = False
        if "n_hid" in config:
            self.n_hid = config.n_hid
        else:
            self.n_hid = None
        if "n_layers" in config:
            self.n_layers = config.n_layers
        else:
            self.n_layers = None
        if "tail" in config:
            self.tail = config.tail
        else:
            self.tail = []
        if "type" in config:
            self.type = config.type
        elif self.shared_weights:
            self.type = self.base.type
        else:
            raise "Policy type must be defined if shared_weights is False"

    def instantiate(self):
        if self.type == "fixed":
            self.model = self.fixed_distribution
            self.is_model = False
        elif self.type == "uniform":
            self.model = self.uniform_distribution
            self.is_model = False
        elif self.type == "mlp":
            self.model = self.make_mlp(nn.LeakyReLU()).to(self.device)
            self.is_model = True
        else:
            raise "Policy model type not defined"

    def __call__(self, states):
        return self.model(states)

    def make_mlp(self, activation):
        """
        Defines an MLP with no top layer activation
        If share_weight == True,
            baseModel (the model with which weights are to be shared) must be provided
        Args
        ----
        layers_dim : list
            Dimensionality of each layer
        activation : Activation
            Activation function
        """
        if self.shared_weights == True and self.base is not None:
            mlp = nn.Sequential(
                self.base.model[:-1],
                nn.Linear(
                    self.base.model[-1].in_features, self.base.model[-1].out_features
                ),
            )
            return mlp
        elif self.shared_weights == False:
            layers_dim = (
                [self.state_dim] + [self.n_hid] * self.n_layers + [(self.output_dim)]
            )
            mlp = nn.Sequential(
                *(
                    sum(
                        [
                            [nn.Linear(idim, odim)]
                            + ([activation] if n < len(layers_dim) - 2 else [])
                            for n, (idim, odim) in enumerate(
                                zip(layers_dim, layers_dim[1:])
                            )
                        ],
                        [],
                    )
                    + self.tail
                )
            )
            return mlp
        else:
            raise ValueError(
                "Base Model must be provided when shared_weights is set to True"
            )

    def fixed_distribution(self, states):
        """
        Returns the fixed distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.fixed_output, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )

    def random_distribution(self, states):
        """
        Returns the random distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.random_output, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )

    def uniform_distribution(self, states):
        """
        Return action logits (log probabilities) from a uniform distribution
        Args: states: tensor
        """
        return torch.ones(
            (len(states), self.output_dim), dtype=self.float, device=self.device
        )
