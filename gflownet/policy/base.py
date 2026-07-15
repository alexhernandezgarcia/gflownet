"""
Base Policy class for GFlowNet policy models.
"""

from abc import abstractmethod
from typing import Union

import torch

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import set_device, set_float_precision


class Policy:
    def __init__(
        self,
        env: GFlowNetEnv,
        device: Union[str, torch.device] = "cpu",
        float_precision: [int, torch.dtype] = 32,
        **kwargs,
    ):
        """
        Base Policy class for a :class:`GFlowNetAgent`.

        Parameters
        ----------
        device : str or torch.device
            The device to be passed to torch tensors.
        float_precision : int or torch.dtype
            The floating point precision to be passed to torch tensors.
        env : GFlowNetEnv
            The environment used to train the :class:`GFlowNetAgent`, used to extract
            needed properties.
        """
        # Device and float precision
        self.device = set_device(device)
        self.float = set_float_precision(float_precision)
        # Input and output dimensions
        self.state_dim = env.policy_input_dim
        self.fixed_output = env.fixed_policy_output
        self.random_output = env.random_policy_output
        self.output_dim = len(self.fixed_output)
        # By default, the policy is not a model
        self.is_model = False

    @abstractmethod
    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """
        Returns the policy outputs corresponding to a batch of states.

        Parameters
        ----------
        states : tensor
            A batch of states in policy format.

        Returns
        -------
        tensor
            The policy outputs corresponding to the input states.
        """
        pass
