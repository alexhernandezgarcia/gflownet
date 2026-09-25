import numpy as np
import torch
from pyro.distributions import ProjectedNormal
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily


def get_mixture_of_projected_normals(weights, concentrations):
    """
    :param weights: torch.tensor of shape [*batch_shape, n_components]
    :param concentrations: torch.tensor of shape [*batch_shape, n_components, n_dim]
    """
    mix = Categorical(weights)
    comp = ProjectedNormal(concentrations)
    return MixtureSameFamily(mix, comp)


class WrappedNormal:
    """
    Wrapped normal distribution on the circle.

    The wrapped normal distribution is obtained by taking a standard (unwrapped)
    Gaussian random variable defined on the real line and "wrapping" it around
    the circle of circumference 2*pi, i.e. mapping x -> x mod 2*pi. Its density
    is given by summing the ordinary Gaussian density over all integer shifts of
    2*pi, which is approximated here by truncating the (infinite) sum to `N`
    terms in each direction.

    See equation (3) in https://arxiv.org/pdf/2206.01729 for the definition.

    Attributes
    ----------
    means : torch.Tensor
        Mean of the underlying unwrapped Gaussian, of shape ``(batch_size, n_dims)``.
    stds : torch.Tensor
        Standard deviation of the underlying unwrapped Gaussian, of shape
        ``(batch_size,)``. Broadcast across ``n_dims`` when used in ``prob`` /
        ``sample``.
    N : int
        Number of wraps considered on each side of the circle when approximating the
        infinite wrapping sum. The sum runs over ``2 * N + 1`` shifted Gaussians (from
        -N to +N).
    """

    def __init__(self, means, stds, N=10):
        """
        Parameters
        ----------
        means : torch.Tensor
            Mean of the unwrapped Gaussian, shape ``(batch_size, n_dims)``.
        stds : torch.Tensor
            Standard deviation of the unwrapped Gaussian, shape ``(batch_size,)``.
        N : int, optional
            Number of times to wrap around the circle in each direction when
            approximating the wrapped density. Higher values give a more accurate
            approximation at the cost of more computation. Defaults to 10.
        """
        self.means = means
        self.stds = stds
        self.N = N

    def prob(self, actions):
        """
        Compute the (unnormalized) wrapped normal density at `actions`.

        Approximates the wrapped normal density by summing the unwrapped
        Gaussian density over ``2 * self.N + 1`` copies of ``actions``, shifted
        by integer multiples of 2*pi.

        Note: The returned value omits the usual ``1 / (std * sqrt(2*pi))``
        normalization constant, so it is proportional to, but not exactly
        equal to, the true probability density.

        Parameters
        ----------
        actions : torch.Tensor
            Points at which to evaluate the density, broadcastable against
            ``self.means``.

        Returns
        -------
        torch.Tensor
            Unnormalized wrapped normal density evaluated at ``actions``, same shape as
            the broadcast of ``actions`` and ``self.means``.
        """
        # TODO: consider using logsumexp instead and computing everything in logscale
        # TODO: why it is not normalised? figure this out, this might be a bug
        p_ = 0
        for i in range(-self.N, self.N + 1):
            p_ += torch.exp(
                -((actions - self.means + 2 * torch.pi * i) ** 2)
                / 2
                / (self.stds.unsqueeze(1).repeat(1, 2)) ** 2
            )
        return p_

    def log_prob(self, actions):
        """
        Compute the log of the (unnormalized) wrapped normal density.

        Parameters
        ----------
        actions : torch.Tensor
            Points at which to evaluate the log-density, broadcastable against
            ``self.means``.

        Returns
        -------
        torch.Tensor
            Log of ``self.prob(actions)``.
        """
        return torch.log(self.prob(actions))

    def sample(self, n_samples=None):
        """
        Draw samples from the wrapped normal distribution.

        Samples are drawn from the underlying unwrapped Gaussian and then
        wrapped onto the circle via `mod 2*pi`.

        Parameters
        ----------
        n_samples : torch.Size or tuple, optional
            Sample shape to draw.  If None, a single sample matching the shape of
            ``self.means`` is returned. Defaults to None.

        Returns
        -------
        torch.Tensor
            Samples wrapped into ``[0, 2*pi)``, with shape determined by ``n_samples``
            (if given) or by ``self.means``.
        """
        n_dims = self.means.shape[-1]
        unwrapped_gaussian = Normal(
            self.means, self.stds.unsqueeze(1).repeat(1, n_dims)
        )
        if n_samples is None:
            sample = unwrapped_gaussian.sample() % (2 * torch.pi)
        else:
            sample = unwrapped_gaussian.sample(n_samples) % (2 * torch.pi)
        return sample
