"""
Classes to represent hyper-torus environments
"""

import itertools
import re
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.neighbors import KernelDensity
from torch.distributions import Bernoulli, Categorical, Uniform, VonMises
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import tfloat, torch2np


class HybridTorus(GFlowNetEnv):
    """
    Continuous (hybrid: discrete and continuous) hyper-torus environment in which the
    action space consists of the selection of which dimension d to increment and of the
    angle of dimension d. The trajectory is of fixed length length_traj.

    The states space is the concatenation of the angle (in radians and within [0, 2 *
    pi]) at each dimension and the number of actions.

    Attributes
    ----------
    n_dim : int
        Dimensionality of the torus

    length_traj : int
       Fixed length of the trajectory.

    state_space_atol: float
        Tolerance for comparing states similarity.
    """

    def __init__(
        self,
        distr_type: str = "von_mises",
        n_dim: int = 2,
        length_traj: int = 1,
        n_comp: int = 3,
        policy_encoding_dim_per_angle: int = None,
        do_nonzero_source_prob: bool = True,
        fixed_distr_params: dict = None,
        random_distr_params: dict = None,
        reward_sampling_method="rejection",
        vonmises_min_concentration: float = 1e-3,
        exp_vonmises_concentration: bool = True,
        state_space_atol=1e-1,
        **kwargs,
    ):
        assert n_dim > 0
        assert length_traj > 0
        assert n_comp > 0
        self.n_dim = n_dim
        self.distr_type = distr_type
        self.length_traj = length_traj
        self.policy_encoding_dim_per_angle = policy_encoding_dim_per_angle
        self.state_space_atol = state_space_atol
        # Parameters of fixed policy distribution
        self.n_comp = n_comp
        if self.distr_type == "diffusion":
            self.n_params_per_dim = 1
        elif self.distr_type == "von_mises":
            self.n_params_per_dim = 3
        else:
            raise ValueError(
                f"Unknown distribution type: {self.distr_type}. Supported types are 'diffusion' and 'von_mises'."
            )
        if do_nonzero_source_prob:
            self.n_params_per_dim += 1
        # Source state: position 0 at all dimensions and number of actions 0
        self.source_angles = [0.0 for _ in range(self.n_dim)]
        self.source = self.source_angles + [0]
        # End-of-sequence action: (n_dim, 0)
        self.eos = (self.n_dim, 0)

        self.reward_sampling_method = reward_sampling_method

        # Base class init
        if self.distr_type == "diffusion":
            assert (
                self.n_comp == 1
            ), "Diffusion distribution only supports 1 component (both forward and backward policies are parametrized as **unimodal** wrapped normal distributions), with a learned mean and a fixed variance"
            self.sigma_max = np.pi
            self.sigma_min = 0.01 * np.pi
            fixed_distr_params = {"means": 0.0, "stds": 0.5}
            random_distr_params = {"means": 0.0, "stds": 2 * np.pi}
        elif self.distr_type == "von_mises":
            fixed_distr_params = {"vonmises_mean": 0.0, "vonmises_concentration": 0.5}
            random_distr_params = {
                "vonmises_mean": 0.0,
                "vonmises_concentration": 0.001,
            }
            self.vonmises_min_concentration = vonmises_min_concentration
            self.exp_vonmises_concentration = exp_vonmises_concentration

        self.fixed_distr_params = fixed_distr_params
        self.random_distr_params = random_distr_params

        super().__init__(
            fixed_distr_params=fixed_distr_params,
            random_distr_params=random_distr_params,
            **kwargs,
        )

        self.continuous = True

    def get_policy_output(self, params: dict):
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy.

        For each dimension of the hyper-torus, the output of the policy should return
        1) a logit, for the categorical distribution over dimensions and 2) the
        location and 3) the concentration of the projected normal distribution to
        sample the increment of the angle and 4) (if do_nonzero_source_prob is True)
        the logit of a Bernoulli distribution to model the (discrete) backward
        probability of returning to the value of the source node.

        Thus:
        - n_params_per_dim = 4 if do_nonzero_source_prob is True
        - n_params_per_dim = 3 if do_nonzero_source_prob is False

        Therefore, the output of the policy model has dimensionality D x
        n_params_per_dim + 1, where D is the number of dimensions, and the elements of
        the output vector are:
        - d * n_params_per_dim: logit of dimension d
        - d * n_params_per_dim + 1: location of Von Mises distribution for dimension d
        - d * n_params_per_dim + 2: log concentration of Von Mises distribution for dimension d
        - d * n_params_per_dim + 3: logit of Bernoulli distribution
        with d in [0, ..., D]
        """
        if self.distr_type == "von_mises":
            policy_output = torch.ones(
                self.n_dim * self.n_params_per_dim + 1,
                dtype=self.float,
                device=self.device,
            )
            policy_output[1 :: self.n_params_per_dim] = params["vonmises_mean"]
            policy_output[2 :: self.n_params_per_dim] = params["vonmises_concentration"]
        elif self.distr_type == "diffusion":
            policy_output = torch.ones(self.n_dim, dtype=self.float, device=self.device)
            policy_output[::1] = params["means"]
        return policy_output
