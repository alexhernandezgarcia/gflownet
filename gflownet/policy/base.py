from abc import ABC, abstractmethod

import torch
from omegaconf import OmegaConf

from gflownet.utils.common import set_device, set_float_precision


class Policy(ABC):
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
        self.instantiate()

    def parse_config(self, config):
        # If config is null, default to uniform
        if config is None:
            config = OmegaConf.create()
            config.type = "uniform"
        self.checkpoint = config.get("checkpoint", None)
        if "type" in config:
            self.type = config.type
        else:
            self.type = "uniform"

    def instantiate(self):
        if self.type == "fixed":
            self.model = self.fixed_distribution
            self.is_model = False
        elif self.type == "uniform":
            self.model = self.uniform_distribution
            self.is_model = False
        else:
            raise "Policy model type not defined"

    def __call__(self, states):
        return self.model(states)

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
