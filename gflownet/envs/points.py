"""
Class to represent in Euclidean spaces.
"""

from typing import Optional

from gflownet.envs.cube import ContinuousCube, HybridCube
from gflownet.envs.set import SetFix


class Points(SetFix):
    """
    Points environment, which represents a set of points in an Euclidean space.

    In practice, it is implemented as a SetFix of M D-dimensional ContinuousCube or
    HybridCube environments, where M is the number of points and D the dimensionality
    of the Euclidean space.

    Attributes
    ----------
    n_points : int
        Number of points.
    n_dim : int
        Dimensionality of the Euclidean space.
    cube_mode : str
        Whether the Cube defining each point should be implemented by a ContinuousCube
        [cont(inuous)] or by a HybridCube [hybrid].
    cube_kwargs : dict
        Dictionary of attributes to be passed to the Cube environments that make each
        point.
    """

    def __init__(
        self,
        n_points: int = 3,
        n_dim: int = 2,
        cube_mode: str = "continuous",
        cube_kwargs: Optional[dict] = {
            "min_incr": 0.1,
            "n_comp": 1,
            "beta_params_min": 0.1,
            "beta_params_max": 100.0,
            "epsilon": 1e-6,
            "kappa": 1e-3,
            "ignored_dims": None,
            "fixed_distr_params": {
                "beta_weights": 1.0,
                "beta_alpha": 10.0,
                "beta_beta": 10.0,
                "bernoulli_bts_prob": 0.1,
                "bernoulli_eos_prob": 0.1,
            },
            "random_distr_params": {
                "beta_weights": 1.0,
                "beta_alpha": 10.0,
                "beta_beta": 10.0,
                "bernoulli_bts_prob": 0.1,
                "bernoulli_eos_prob": 0.1,
            },
        },
        **kwargs,
    ):
        self.n_points = n_points
        self.n_dim = n_dim
        self.cube_mode = cube_mode
        self.cube_kwargs = cube_kwargs
        # If n_dim not in cube_kwargs, add it
        if "n_dim" not in self.cube_kwargs:
            self.cube_kwargs["n_dim"] = self.n_dim
        # Define Cube sub-environments
        if self.cube_mode.startswith("cont"):
            subenvs = [ContinuousCube(**self.cube_kwargs) for _ in range(self.n_points)]
        elif self.cube_mode.startswith("hybrid"):
            subenvs = [HybridCube(**self.cube_kwargs) for _ in range(self.n_points)]
        else:
            raise ValueError(f"Uknown cube mode {self.cube_mode}")
        super().__init__(subenvs=subenvs, **kwargs)
