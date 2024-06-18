from abc import ABC, abstractmethod

import torch
from omegaconf import OmegaConf
from torch import nn

from gflownet.utils.common import set_device, set_float_precision


class ModelBase(ABC):
    def __init__(self, config, env, device, float_precision, base=None):
        # Device and float precision
        self.device = set_device(device)
        self.float = set_float_precision(float_precision)
        # Input and output dimensions
        self.state_dim = env.policy_input_dim
        self.fixed_output = env.fixed_policy_output
        self.random_output = env.random_policy_output
        self.output_dim = len(self.fixed_output)
        # Optional base model
        self.base = base

        self.parse_config(config)

    def parse_config(self, config):
        # If config is null, default to uniform
        if config is None:
            config = OmegaConf.create()
            config.type = "uniform"
        self.checkpoint = config.get("checkpoint", None)
        self.shared_weights = config.get("shared_weights", False)
        self.n_hid = config.get("n_hid", None)
        self.n_layers = config.get("n_layers", None)
        self.tail = config.get("tail", [])
        if "type" in config:
            self.type = config.type
        elif self.shared_weights:
            self.type = self.base.type
        else:
            raise "Policy type must be defined if shared_weights is False"

    @abstractmethod
    def instantiate(self):
        pass

    def __call__(self, states):
        return self.model(states)


class Policy(ModelBase):
    def __init__(self, config, env, device, float_precision, base=None):
        super().__init__(config, env, device, float_precision, base)

        self.instantiate()

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

    def instantiate(self):
        if self.type == "fixed":
            self.model = self.fixed_distribution
            self.is_model = False
        elif self.type == "uniform":
            self.model = self.uniform_distribution
            self.is_model = False
        elif self.type == "mlp":
            from policy.mlp import MLPPolicy
            mlp_policy = MLPPolicy(self.config, self.env, self.device, self.float_precision, self.base)
            self.model = mlp_policy.model
            self.is_model = mlp_policy.is_model
        else:
            raise "Policy model type not defined"

    def fixed_distribution(self, states):
        """
        Returns the fixed distribution specified by the environment.

        Parameters
        ----------
        states : tensor
            The states for which the fixed distribution is to be returned
        """
        return torch.tile(self.fixed_output, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )

    def random_distribution(self, states):
        """
        Returns the random distribution specified by the environment.

        Parameters
        ----------
        states : tensor
            The states for which the random distribution is to be returned
        """
        return torch.tile(self.random_output, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )

    def uniform_distribution(self, states):
        """
        Return action logits (log probabilities) from a uniform distribution

        Parameters
        ----------
        states : tensor
            The states for which the uniform distribution is to be returned
        """
        return torch.ones(
            (len(states), self.output_dim), dtype=self.float, device=self.device
        )
