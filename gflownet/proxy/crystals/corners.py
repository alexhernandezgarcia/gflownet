"""Debugging proxy for Crystal-GFN"""

from typing import Dict, List

import numpy as np
import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.proxy.box.corners import Corners


class CrystalCorners(Proxy):
    """
    A synthetic proxy which resembles the Corners proxy in the lattice parameters
    domain. It places different corners (with varying mean and standard deviations for
    the Gaussians) depending on the space group and composition.

    Attributes
    ----------
    proxy_default : Corners
        Default proxy to be used for the states not included in the conditions
        specified in the configuration.
    """

    def __init__(
        self, mu: float = 0.75, sigma: float = 0.05, config: List[Dict] = [], **kwargs
    ):
        """
        Initializes the CrystalCorners proxy according the configuration passed as an
        argument.

        Parameters
        ----------
        mu : float
            Mean of the multivariate Gaussian distribution used to construct the
            default corners proxy, that is the proxy used for states not included in
            the conditions specified in the config. Note that mu is a single float
            because the mean is homogeneous.
        sigma : float
            Standard deviation of the multivariate Gaussian distribution used to
            default corners proxy, that is the proxy used for states not included in
            the conditions specified in the config. Note that sigma is a single float
            because the covariance matrix is diagonal.
        config : list
            A list of dictionaries specifying the parameters of the Corners
            sub-proxies. Each element in the list is a dictionary that must contain at
            least one of the following keys:
                - spacegroup: int
                - element: int
            Additionally, it must contain the following keys:
                - mu: float
                - sigma:
            mu and sigma are the mean and standard deviation for a Corners sub-proxy to
            be applied for states that have the spacegroup and element indicated in the
            configuration.
        """
        super().__init__(**kwargs)

        # Check whether the config list is invalid
        if not all([self._dict_is_valid[el] for el in config]):
            raise ValueError("Configuration is not valid")

        # Initialize default proxy
        self.proxy_default = Corners(n_dim=3, mu=mu, sigma=sigma, **kwargs)

    @torch.no_grad()
    def __call__(self, states: TensorType["batch", "102"]) -> TensorType["batch"]:
        """
        Builds the proxy values of the CornersProxy for a batch Crystal states.

        Parameters
        ----------
        states : torch.Tensor
            States to infer on. Shape: ``(batch, [6 + 1 + n_elements])``.

        Returns
        -------
        torch.Tensor
            Proxy scores. Shape: ``(batch,)``.
        """
        comp = states[:, :-7]
        sg = states[:, -7]
        lat_params = states[:, -6:]
        lp_lengths = lat_params[:3]

    @staticmethod
    def _dict_is_valid(config: Dict):
        """
        Checks whether a dictionary of configuration is valid.

        To be valid, the following conditions must be satisfied:
            - There is a key 'mu' containing a float number.
            - There is a key 'sigma' containing a float number.
            - There is at least one of the keys 'spacegroup' and 'element' containing
              an int number.
        """
        if "mu" not in config.keys() or isinstance(config["mu"], float):
            return False
        if "sigma" not in config.keys() or isinstance(config["sigma"], float):
            return False
        if "spacegroup" not in config.keys() and "element" not in config.keys():
            return False
        if "spacegroup" in config.keys() and not isinstance(config["spacegroup"], int):
            return False
        if "element" in config.keys() and not isinstance(config["spacegroup"], int):
            return False
        return True
