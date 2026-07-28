"""
Bayesian regression tree proxy for the composite RegressionTree environment.

Provides ``NormalGammaTreeProxy``, the regression analogue of
``gflownet.proxy.tree.CategoricalTreeProxy``: the Dirichlet-Multinomial
marginal likelihood over leaf class counts is replaced by the
Normal-Inverse-Gamma (NIG) marginal likelihood over the continuous targets
routed to each leaf. The structure priors ("node_count", "exponential",
"bcart", "none") and the reward computation are inherited unchanged from
``CategoricalTreeProxy``.
"""

import math
from typing import Dict, List, Optional

import numpy as np
from scipy.special import gammaln

from gflownet.envs.tree.tree import Tree
from gflownet.proxy.tree import CategoricalTreeProxy, _route_samples_to_leaves


class NormalGammaTreeProxy(CategoricalTreeProxy):
    """
    Bayesian regression tree proxy using the Normal-Inverse-Gamma marginal
    log-likelihood (integrating out the leaf mean and variance) combined with
    a configurable structure prior.

    The leaf model is the standard conjugate Bayesian CART regression model
    (mean-variance shift; Chipman et al., 1998): at each leaf,

        y_i | mu, sigma^2 ~ N(mu, sigma^2)
        mu | sigma^2 ~ N(mu_0, sigma^2 / kappa_0)
        sigma^2 ~ InvGamma(alpha_0, beta_0)

    Under conjugacy, the marginal likelihood of the ``n`` targets ``y`` routed
    to a leaf is available in closed form:

        log p(y) = -n/2 log(2 pi) + 1/2 (log kappa_0 - log kappa_n)
                   + alpha_0 log beta_0 - alpha_n log beta_n
                   + log Gamma(alpha_n) - log Gamma(alpha_0)

    with the posterior parameters

        kappa_n = kappa_0 + n
        alpha_n = alpha_0 + n/2
        beta_n  = beta_0 + 1/2 sum_i (y_i - ybar)^2
                  + kappa_0 n (ybar - mu_0)^2 / (2 kappa_n).

    The total log-likelihood of a tree is the sum over its leaves, exactly
    mirroring the per-leaf factorization of the classification proxy.

    Structure priors, ``normalize_likelihood`` and ``__call__`` (which returns
    the log-posterior, to be used with ``reward_function: "exponential"``) are
    inherited from ``CategoricalTreeProxy``.
    """

    def __init__(
        self,
        prior_type: str = "node_count",
        mu_0: Optional[float] = None,
        kappa_0: float = 0.1,
        alpha_0: float = 2.0,
        beta_0: Optional[float] = None,
        beta: float = 1.0,
        sigma: float = 0.95,
        phi: float = 2.0,
        normalize_likelihood: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        prior_type : str
            Structure prior type. One of: ``"node_count"``, ``"exponential"``,
            ``"bcart"``, ``"none"`` (see ``CategoricalTreeProxy``).
        mu_0 : float, optional
            Prior mean of the leaf means. If None, set to the mean of the
            training targets at setup.
        kappa_0 : float
            Prior strength (pseudo-count) of ``mu_0``. Small values yield a
            weakly informative prior on the leaf means.
        alpha_0 : float
            Shape of the Inverse-Gamma prior over the leaf variance.
        beta_0 : float, optional
            Scale of the Inverse-Gamma prior over the leaf variance. If None,
            set to ``(alpha_0 - 1) * var(y_train)`` at setup (for
            ``alpha_0 > 1``), so that the prior mean of sigma^2 equals the
            variance of the training targets; for ``alpha_0 <= 1`` the
            variance itself is used.
        beta : float
            Penalty coefficient for the ``"exponential"`` structure prior.
        sigma : float
            Base splitting probability for the BCART structure prior.
        phi : float
            Depth decay exponent for the BCART structure prior.
        normalize_likelihood : bool
            If True, divide the marginal log-likelihood by the number of
            training samples (see ``CategoricalTreeProxy``).
        """
        super().__init__(
            prior_type=prior_type,
            beta=beta,
            sigma=sigma,
            phi=phi,
            normalize_likelihood=normalize_likelihood,
            **kwargs,
        )
        self.mu_0 = mu_0
        self.kappa_0 = kappa_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

    def setup(self, env: Optional[Tree] = None):
        if env is None:
            raise ValueError("NormalGammaTreeProxy.setup requires a Tree environment.")
        self.env = env
        if env.X_train is None or env.y_train is None:
            raise ValueError("NormalGammaTreeProxy requires training data in the env.")

        self.n_features = env.X_train.shape[1]
        self.n_train = len(env.y_train)

        if self.kappa_0 <= 0.0 or self.alpha_0 <= 0.0:
            raise ValueError(
                f"NormalGammaTreeProxy requires kappa_0 > 0 and alpha_0 > 0, "
                f"got kappa_0={self.kappa_0}, alpha_0={self.alpha_0}."
            )

        # Resolve data-driven defaults for the NIG hyper-parameters
        y = np.asarray(env.y_train, dtype=float)
        self._mu_0 = float(self.mu_0) if self.mu_0 is not None else float(np.mean(y))
        self._kappa_0 = float(self.kappa_0)
        self._alpha_0 = float(self.alpha_0)
        if self.beta_0 is not None:
            self._beta_0 = float(self.beta_0)
        else:
            var = float(np.var(y))
            if var <= 0.0:
                var = 1.0
            self._beta_0 = (self._alpha_0 - 1.0) * var if self._alpha_0 > 1.0 else var
        if self._beta_0 <= 0.0:
            raise ValueError(
                f"NormalGammaTreeProxy requires beta_0 > 0, got {self._beta_0}."
            )

    def _compute_log_likelihood(self, state: Dict) -> float:
        """
        Computes the Normal-Inverse-Gamma marginal log-likelihood of the
        training targets under the tree ``state`` (sum of the closed-form
        per-leaf marginals; see the class docstring). Computes log[p(y)] per
        leaf and sums them up to get the likelihood of the whole tree.
        """
        leaf_samples = _route_samples_to_leaves(self.env, state, self.env.X_train)
        y_all = np.asarray(self.env.y_train, dtype=float)

        log_likelihood = 0.0
        for indices in leaf_samples.values():
            y = y_all[indices]
            n = len(y)
            ybar = float(np.mean(y))
            kappa_n = self._kappa_0 + n
            alpha_n = self._alpha_0 + 0.5 * n
            ss = float(np.sum((y - ybar) ** 2)) # sum of square
            beta_n = (
                self._beta_0
                + 0.5 * ss
                + 0.5 * self._kappa_0 * n * (ybar - self._mu_0) ** 2 / kappa_n
            )
            log_likelihood += (
                -0.5 * n * math.log(2.0 * math.pi)
                + 0.5 * (math.log(self._kappa_0) - math.log(kappa_n))
                + self._alpha_0 * math.log(self._beta_0)
                - alpha_n * math.log(beta_n)
                + float(gammaln(alpha_n) - gammaln(self._alpha_0))
            )
        return log_likelihood
