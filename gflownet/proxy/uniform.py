import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


class Uniform(Proxy):
    """
    A proxy that returns a uniform (constant) energy across all terminating states. In
    practice, the output is constructed with a normal distribution, with the mean and
    std given as arguments, so as to have noise and prevent numerical issues when
    computing metrics such as the correlation of rewards with log probabilities.

    mean : float
        Mean of the normal distribution that makes the proxy output. The mean of the
        proxy will thus be -mean. 1.0 by default.

    std : float
        Standard deviation of the Gaussian noise added to the proxy output. If std is
        equal to zero, then the proxy will be constant and equal to -mean.
    """

    def __init__(self, mean: float = 1.0, std: float = 1e-3, **kwargs):
        self.mean = mean
        self.std = std
        super().__init__(**kwargs)

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        n = states.shape[0]
        return -1.0 * torch.normal(
            torch.full((n,), self.mean), torch.full((n,), self.std)
        )

    @property
    def min(self):
        return -1.0
