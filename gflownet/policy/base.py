"""
Base Policy class for GFlowNet policy models.
"""

from abc import abstractmethod
from typing import Dict, Union

import torch
from omegaconf import DictConfig, OmegaConf

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import set_device, set_float_precision


class SubPolicy:
    """
    The class to define sub-policies (forward, backward, etc.).

    Attributes
    ----------
    model : torch.nn.Module or None
        A torch model, a fixed vector or None.
    is_trainable : bool
        Whether the model has trainable weights.
    shared_weights : bool
        Whether the policy should share the weights with another policy (the forward
        policy). False by default. If True, ``model`` must be a ``torch.nn.Module``.
    checkpoint : str
        A path to a file containing a checkpoints of a model.
    """
    def __init__(
        self,
        model: Union[torch.nn.Module, None] = None,
        shared_weights: bool = False,
        checkpoint: str = None,
        **kwargs,
    ):
        """
        Base SubPolicy class for the components of the :class:`Policy`.

        Parameters
        ----------
        model : torch.nn.Module or None
            A torch model.
        """
        self.model = model
        if self.model and isinstance(self.model, torch.nn.Module):
            self.is_trainable = True
        else:
            self.is_trainable = False
        self.shared_weights = shared_weights
        self.checkpoint = checkpoint

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """
        Returns the sub-policy outputs corresponding to a batch of states.

        Parameters
        ----------
        states : tensor
            A batch of states in policy format.

        Returns
        -------
        tensor
            The policy outputs corresponding to the input states.
        """
        return self.model(states)


class Policy:
    def __init__(
        self,
        forward: Union[SubPolicy, Dict, DictConfig],
        backward: Union[SubPolicy, Dict, DictConfig, None] = None,
        stateflow: Union[SubPolicy, Dict, DictConfig, None] = None,
        logZ: Union[int, SubPolicy, Dict, DictConfig, None] = None,
        state_dim: int = None,
        action_dim: int = None,
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
        forward : SubPolicy, Dict or DictConfig
            The forward subpolicy or the parameters to instantiate it.
        backward : SubPolicy, Dict, DictConfig or None
            The backward subpolicy or the parameters to instantiate it. If None
            (default), no backward policy is used.
        logZ : int, SubPolicy, Dict, DictConfig or None
            The partition function subpolicy or the parameters to instantiate it. If it
            is an integer, it is interpreted as the dimensionality of a vector of
            learnable parameters. If None (default), the partition function is not
            learned.
        """
        # Device and float precision
        self.device = set_device(device)
        self.float = set_float_precision(float_precision)
        # State and action dimensionality
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Sub-policies
        if isinstance(forward, SubPolicy):
            self.forward = forward
        else:
            self.forward = SubPolicy(**forward)
        if backward is None or isinstance(backward, SubPolicy):
            self.backward = backward
        else:
            self.backward = SubPolicy(**backward)
        if stateflow is None or isinstance(stateflow, SubPolicy):
            self.stateflow = stateflow
        else:
            self.stateflow = SubPolicy(**stateflow)
        if isinstance(logZ, int):
            self.logZ = torch.nn.Parameter(torch.ones(logZ) * 150.0 / 64)
        elif logZ is None or isinstance(logZ, SubPolicy):
            self.logZ = logZ
        else:
            self.logZ = SubPolicy(**logZ)

    def setup(self, env: GFlowNetEnv):
        """
        Obtains the state and action dimensions from an input environment.

        Parameters
        ----------
        env : GFlowNetEnv
            A instance of an environment.
        """
        self.state_dim = env.policy_input_dim
        self.action_dim = env.policy_output_dim
