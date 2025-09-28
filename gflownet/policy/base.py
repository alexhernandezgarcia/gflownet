"""
Base Policy class for GFlowNet policy models.
"""

from typing import Tuple, Union

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import set_device, set_float_precision


class Policy:
    def __init__(
        self,
        config: Union[dict, DictConfig],
        env: GFlowNetEnv,
        device: Union[str, torch.device],
        float_precision: [int, torch.dtype],
        base=None,
        **kwargs,
    ):
        """
        Base Policy class for a :class:`GFlowNetAgent`.

        Parameters
        ----------
        config : dict or DictConfig
            The configuration dictionary to set up the policy model.
        env : GFlowNetEnv
            The environment used to train the :class:`GFlowNetAgent`, used to extract
            needed properties.
        device : str or torch.device
            The device to be passed to torch tensors.
        float_precision : int or torch.dtype
            The floating point precision to be passed to torch tensors.
        base: Policy (optional)
            A base policy to be used as backbone for the backward policy.
        """
        config = self._get_config(config)
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
        # Policy type, defaults to uniform
        self.type = config.get("type", "uniform")
        # Checkpoint, defaults to None
        self.checkpoint = config.get("checkpoint", None)
        # Instantiate the model
        self.model, self.is_model = self.make_model()

    @staticmethod
    def _get_config(config: Union[dict, DictConfig]) -> Union[dict, DictConfig]:
        """
        Returns a configuration dictionary, even if the input is None.

        Parameters
        ----------
        config : dict or DictConfig
            The configuration dictionary to set up the policy model. It may be None, in
            which an empty config is created and the defaults will be used.

        Returns
        -------
        config : dict or DictConfig
            The configuration dictionary to set up the policy model.
        """
        if config is None:
            config = OmegaConf.create()
        return config

    def make_model(self) -> Tuple[Union[torch.Tensor, torch.nn.Module], bool]:
        """
        Instantiates the model of the policy.

        Returns
        -------
        model : torch.tensor or torch.nn.Module
            A tensor representing the output of the policy or a torch model.
        is_model : bool
            True if the policy is a model (for example, a neural network) and False if
            it is a fixed tensor (for example to make a uniform distribution).
        """
        if self.type == "fixed":
            return self.fixed_distribution, False
        elif self.type == "uniform":
            return self.uniform_distribution, False
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
