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

    Specifically, a different proxy than the default one can be used for states that
    contain a particular space group or a particular element.

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
            sub-proxies. Each element in the list is a dictionary that must contain one
            (and only one) of the following keys:
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
        if not all([self._dict_is_valid(el) for el in config]):
            raise ValueError("Configuration is not valid")
        self.proxies = config

        # Initialize special case proxies
        for conf in self.proxies:
            conf.update(
                {
                    "proxy": Corners(
                        n_dim=3, mu=conf["mu"], sigma=conf["sigma"], **kwargs
                    )
                }
            )
            conf["proxy"].setup()

        # Initialize default proxy
        self.proxy_default = Corners(n_dim=3, mu=mu, sigma=sigma, **kwargs)
        self.proxy_default.setup()

    def setup(self, env=None):
        """
        Sets the minimum length and maximum length of the LatticeParameters
        sub-environment.

        This is needed to be able to rescale the lattice lengths before passing them to
        the Corners proxy.

        Parameters
        ----------
        env : Crystal
            A Crystal environment.
        """
        if env:
            self.min_length = env.lattice_parameters.min_length
            self.max_length = env.lattice_parameters.max_length

    def lattice_lengths_to_corners_proxy(
        self, lp_lengths: TensorType["batch", "3"]
    ) -> TensorType["batch", "3"]:
        """
        Converts a batch of lattice lengtsh in LatticeParameters proxy format into the
        format expected by the Corners proxy.

        The lattice lengths in LatticeParameters proxy format are in angstroms.

        The corners proxy expects the states in the range [-1, 1].

        Parameters
        ----------
        lp_lengths : tensor
            Batch of lattice lengths in LatticeParameters proxy format (angstroms).

        Returns
        -------
        tensor
            Batch of re-scaled lattice lengths in the range [-1, 1].
        """
        return -1.0 + ((lp_lengths - self.min_length) * 2.0) / (
            self.max_length - self.min_length
        )

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
        lp_lengths = self.lattice_lengths_to_corners_proxy(lat_params[:, :3])

        # Apply the corresponding proxy for each state in the batch
        scores = torch.empty(states.shape[0], dtype=self.float)
        default = torch.ones(states.shape[0], dtype=torch.bool)
        for proxy in self.proxies:
            if "spacegroup" in proxy:
                indices = sg == proxy["spacegroup"]
            elif "element" in proxy:
                indices = comp[:, proxy["element"]] > 0
            else:
                raise ValueError("Configuration is not valid")
            scores[indices] = proxy["proxy"](lp_lengths[indices])
            default[indices] = False

        # Apply default proxy
        scores[default] = self.proxy_default(lp_lengths[default])

        return scores

    @staticmethod
    def _dict_is_valid(config: Dict):
        """
        Checks whether a dictionary of configuration is valid.

        To be valid, the following conditions must be satisfied:
            - There is a key 'mu' containing a float number.
            - There is a key 'sigma' containing a float number.
            - There is a key 'spacegroup' or 'element', but only one of the two,
              containing an int number.
        """
        if "mu" not in config.keys() or not isinstance(config["mu"], float):
            return False
        if "sigma" not in config.keys() or not isinstance(config["sigma"], float):
            return False
        if "spacegroup" not in config.keys() and "element" not in config.keys():
            return False
        if "spacegroup" in config.keys() and "element" in config.keys():
            return False
        if "spacegroup" in config.keys() and not isinstance(config["spacegroup"], int):
            return False
        if "element" in config.keys() and not isinstance(config["element"], int):
            return False
        return True
