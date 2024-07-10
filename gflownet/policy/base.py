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
        env: GFlowNetEnv,
        device: Union[str, torch.device],
        float_precision: [int, torch.dtype],
        base=None,
        shared_weights: bool = False,
        checkpoint: str = None,
    ):
        """
        Base Policy class for a :class:`GFlowNetAgent`.

        Parameters
        ----------
        env : GFlowNetEnv
            The environment used to train the :class:`GFlowNetAgent`, used to extract
            needed properties.
        device : str or torch.device
            The device to be passed to torch tensors.
        float_precision : int or torch.dtype
            The floating point precision to be passed to torch tensors.
        base: Policy (optional)
            A base policy to be used as backbone for the backward policy.
        shared_weights: bool (optional)
            Whether the weights of the backward policy are shared with the (base)
            forward policy model. Defaults to False.
        checkpoint: str (optional)
            The path the a checkpoint file to be reloaded as the policy model.
        """
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
        # Shared weights, defaults to False
        self.shared_weights = shared_weights
        # Checkpoint, defaults to None
        self.checkpoint = checkpoint
        # Instantiate the model
        self.model, self.is_model = self.make_model()

    @abstractmethod
    def make_model(self) -> Tuple[Union[torch.Tensor, torch.nn.Module], bool]:
        """
        Instantiates the model or fixed tensor of the policy.

        Returns
        -------
        model : torch.tensor or torch.nn.Module
            A tensor representing the output of the policy or a torch model.
        is_model : bool
            True if the policy is a model (for example, a neural network) and False if
            it is a fixed tensor (for example to make a uniform distribution).
        """
        pass

    def __call__(self, states: torch.Tensor):
        """
        Returns the outputs of the policy model on a batch of states.

        Parameters
        ----------
        states : torch.Tensor
            A batch of states in policy format.
        """
        return self.model(states)
