"""
RegressionTree composite environment for binary regression tree construction.

The environment is a thin subclass of :py:class:`Tree`: the MDP over tree
structures (states, actions, masks, forward/backward steps, threshold
rescaling) is identical for classification and regression, because the
GFlowNet only models the posterior over tree *structures* T. What changes is
the Bayesian model of the targets at the leaves, which affects:

1. Data handling: targets ``y`` are continuous (float), not class labels.
2. The marginal likelihood P[Y|X, T]: instead of the Dirichlet-multinomial
   marginal of the classification tree, each leaf uses a
   Normal-Inverse-Gamma (NIG) conjugate model,

       y_i | mu, sigma^2  ~  Normal(mu, sigma^2)          for i in leaf
       mu | sigma^2       ~  Normal(mu0, sigma^2 / kappa0)
       sigma^2            ~  InvGamma(alpha0, beta0)

   which yields a closed-form log marginal likelihood per leaf (a
   Student-t form). The tree marginal likelihood is the product over
   leaves; the structure prior P[T|X] = exp(-beta * n(T)) is unchanged.
3. Evaluation: leaf parameters are sampled from the NIG posterior at
   inference time (mirroring the Dirichlet draws of the classification
   tree) and metrics are regression metrics (MSE, RMSE, MAE, R2).

The reward proxy may either call :py:meth:`log_marginal_likelihood` /
:py:meth:`log_posterior_unnorm` on this environment or re-implement the NIG
marginal itself.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gflownet.envs.tree.tree import Tree


class RegressionTree(Tree):
    """
    Composite environment that constructs a binary regression tree.

    The construction MDP is inherited unchanged from :py:class:`Tree`; this
    class only replaces the treatment of the targets (continuous instead of
    categorical) and the leaf model (Normal-Inverse-Gamma instead of
    Dirichlet-multinomial).

    Attributes
    ----------
    mu0 : float
        Prior mean of the leaf means. If None, defaults to the mean of
        ``y_train`` (empirical Bayes) or 0.0 without data.
    kappa0 : float
        Prior precision scale on the leaf mean (pseudo-count of prior
        observations). Small values give a weak prior.
    alpha0 : float
        Shape of the InvGamma prior on the leaf variance.
    beta0 : float
        Scale of the InvGamma prior on the leaf variance. If None, defaults
        to the variance of ``y_train`` (so that the prior mean of sigma^2 is
        Var(y) for alpha0 = 2) or 1.0 without data.
    """

    def __init__(
        self,
        max_depth: int = 3,
        node_type: str = "continuous",
        rescale_thresholds: bool = True,
        node_kwargs: dict = None,
        X_train: Optional[npt.NDArray] = None,
        y_train: Optional[npt.NDArray] = None,
        X_test: Optional[npt.NDArray] = None,
        y_test: Optional[npt.NDArray] = None,
        data_path: Optional[str] = None,
        scale_data: bool = True,
        mu0: Optional[float] = None,
        kappa0: float = 0.1,
        alpha0: float = 2.0,
        beta0: Optional[float] = None,
        **kwargs,
    ):
        """
        See :py:meth:`Tree.__init__` for the shared parameters. The
        regression-specific parameters are the NIG prior hyperparameters
        ``mu0``, ``kappa0``, ``alpha0`` and ``beta0``.
        """
        # Keep float copies of the targets: the parent constructor casts the
        # targets to int (class labels), which would truncate continuous
        # values. The parent only uses y for evaluation, never for the MDP,
        # so overwriting the attributes after the call is safe.
        if X_train is not None and y_train is not None:
            y_train_float = np.asarray(y_train, dtype=float)
            if X_test is not None and y_test is not None:
                y_test_float = np.asarray(y_test, dtype=float)
            else:
                y_test_float = None
        elif data_path is not None:
            _, y_train_raw, _, y_test_raw, _ = Tree._load_dataset(data_path)
            y_train_float = np.asarray(y_train_raw, dtype=float)
            y_test_float = (
                np.asarray(y_test_raw, dtype=float) if y_test_raw is not None else None
            )
        else:
            y_train_float = None
            y_test_float = None

        super().__init__(
            max_depth=max_depth,
            node_type=node_type,
            rescale_thresholds=rescale_thresholds,
            node_kwargs=node_kwargs,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            data_path=data_path,
            scale_data=scale_data,
            **kwargs,
        )

        # Restore continuous targets
        if y_train_float is not None:
            self.y_train = y_train_float
            self.y_test = y_test_float

        # NIG prior hyperparameters, with empirical-Bayes defaults
        if mu0 is None:
            mu0 = float(np.mean(self.y_train)) if self.y_train is not None else 0.0
        if beta0 is None:
            if self.y_train is not None:
                beta0 = max(float(np.var(self.y_train)), 1e-6)
            else:
                beta0 = 1.0
        if kappa0 <= 0 or alpha0 <= 0 or beta0 <= 0:
            raise ValueError(
                "NIG prior requires kappa0 > 0, alpha0 > 0 and beta0 > 0, got "
                f"kappa0={kappa0}, alpha0={alpha0}, beta0={beta0}."
            )
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)

    # =========================================================================
    # Normal-Inverse-Gamma leaf model
    # =========================================================================

    def _leaf_posterior(self, y_leaf: npt.NDArray) -> Tuple[float, float, float, float]:
        """
        Computes the NIG posterior parameters at a leaf given the targets
        ``y_leaf`` of the training samples that reach it.

        Returns
        -------
        (mu_n, kappa_n, alpha_n, beta_n) : tuple of float
            Posterior hyperparameters. For an empty leaf, the prior is
            returned unchanged.
        """
        n = y_leaf.size
        if n == 0:
            return self.mu0, self.kappa0, self.alpha0, self.beta0
        y_mean = float(np.mean(y_leaf))
        sse = float(np.sum((y_leaf - y_mean) ** 2))
        kappa_n = self.kappa0 + n
        mu_n = (self.kappa0 * self.mu0 + n * y_mean) / kappa_n
        alpha_n = self.alpha0 + n / 2.0
        beta_n = (
            self.beta0
            + 0.5 * sse
            + (self.kappa0 * n * (y_mean - self.mu0) ** 2) / (2.0 * kappa_n)
        )
        return mu_n, kappa_n, alpha_n, beta_n

    def _leaf_log_marginal(self, y_leaf: npt.NDArray) -> float:
        """
        Computes the log marginal likelihood of the targets ``y_leaf`` at a
        leaf under the NIG model, integrating out (mu, sigma^2):

            log m(y) = lgamma(alpha_n) - lgamma(alpha0)
                       + alpha0 * log(beta0) - alpha_n * log(beta_n)
                       + 0.5 * (log(kappa0) - log(kappa_n))
                       - (n / 2) * log(2 * pi)

        An empty leaf contributes 0 (m(empty) = 1).
        """
        n = y_leaf.size
        if n == 0:
            return 0.0
        _, kappa_n, alpha_n, beta_n = self._leaf_posterior(y_leaf)
        return (
            math.lgamma(alpha_n)
            - math.lgamma(self.alpha0)
            + self.alpha0 * math.log(self.beta0)
            - alpha_n * math.log(beta_n)
            + 0.5 * (math.log(self.kappa0) - math.log(kappa_n))
            - (n / 2.0) * math.log(2.0 * math.pi)
        )

    def log_marginal_likelihood(self, state: Optional[Dict] = None) -> float:
        """
        Computes log P[Y|X, T] of the tree ``state`` on the training data:
        the sum of the per-leaf NIG log marginal likelihoods, with samples
        routed through the tree by :py:meth:`Tree._route_samples`.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "log_marginal_likelihood requires training data (X_train, y_train)."
            )
        state = self._get_state(state)
        leaf_samples = self._route_samples(state, self.X_train)
        return float(
            sum(
                self._leaf_log_marginal(self.y_train[idx])
                for idx in leaf_samples.values()
            )
        )

    def log_posterior_unnorm(
        self, state: Optional[Dict] = None, beta: float = 0.0
    ) -> float:
        """
        Computes the unnormalized log posterior of the tree ``state``:
        ``log P[Y|X, T] - beta * n(T)``, where ``n(T)`` is the number of
        decision nodes, matching the structure prior
        ``P[T|X] = exp(-beta * n(T))`` of the classification tree.
        """
        state = self._get_state(state)
        n_nodes = sum(state["_dones"])
        return self.log_marginal_likelihood(state) - beta * n_nodes

    # =========================================================================
    # Evaluation: posterior-predictive sampling and regression metrics
    # =========================================================================

    def _sample_leaf_params(
        self, state: Dict, rng: np.random.Generator
    ) -> Dict[int, float]:
        """
        Draws a posterior sample of the mean at every leaf of ``state``
        reached by training samples, mirroring the Dirichlet draws of the
        classification tree:

            sigma^2 ~ InvGamma(alpha_n, beta_n)
            mu      ~ Normal(mu_n, sigma^2 / kappa_n)

        Returns a mapping ``leaf_node_index -> sampled leaf mean``.
        """
        leaf_samples = self._route_samples(state, self.X_train)
        means: Dict[int, float] = {}
        for k, idx in leaf_samples.items():
            mu_n, kappa_n, alpha_n, beta_n = self._leaf_posterior(self.y_train[idx])
            sigma2 = 1.0 / rng.gamma(alpha_n, 1.0 / beta_n)
            means[k] = float(rng.normal(mu_n, math.sqrt(sigma2 / kappa_n)))
        return means

    def _predict(
        self, state: Dict, leaf_means: Dict[int, float], X: npt.NDArray
    ) -> npt.NDArray:
        """
        Returns a ``(n_samples,)`` vector of predictions for ``X`` routed
        through ``state`` with sampled leaf means ``leaf_means``. Samples
        landing at a leaf never visited during training get the prior mean
        ``mu0``.
        """
        leaf_samples = self._route_samples(state, X)
        preds = np.full(len(X), self.mu0, dtype=float)
        for k, idx in leaf_samples.items():
            if k in leaf_means:
                preds[idx] = leaf_means[k]
        return preds

    @staticmethod
    def _compute_tree_scores(preds: npt.NDArray, y: npt.NDArray) -> Dict[str, float]:
        """
        Given a ``(n_trees, n_samples)`` matrix of predictions and the target
        vector, returns:

        - ``mean_tree_mse`` / ``mean_tree_rmse`` / ``mean_tree_mae`` /
          ``mean_tree_r2``: mean of per-tree scores.
        - ``forest_mse`` / ``forest_rmse`` / ``forest_mae`` / ``forest_r2``:
          scores of the ensemble prediction obtained by averaging the
          predictions across trees (the Bayesian model average).
        """
        n_trees = preds.shape[0]
        per_tree_mse = [mean_squared_error(y, preds[i]) for i in range(n_trees)]
        per_tree_mae = [mean_absolute_error(y, preds[i]) for i in range(n_trees)]
        per_tree_r2 = [r2_score(y, preds[i]) for i in range(n_trees)]
        forest_pred = preds.mean(axis=0)
        forest_mse = mean_squared_error(y, forest_pred)
        return {
            "mean_tree_mse": float(np.mean(per_tree_mse)),
            "mean_tree_rmse": float(np.mean(np.sqrt(per_tree_mse))),
            "mean_tree_mae": float(np.mean(per_tree_mae)),
            "mean_tree_r2": float(np.mean(per_tree_r2)),
            "forest_mse": float(forest_mse),
            "forest_rmse": float(np.sqrt(forest_mse)),
            "forest_mae": float(mean_absolute_error(y, forest_pred)),
            "forest_r2": float(r2_score(y, forest_pred)),
        }

    def test(
        self,
        samples: List[Dict],
        top_k_trees: int = 0,
        plot_top_k: bool = True,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, object]:
        """
        Evaluates a batch of sampled terminating regression trees.

        Mirrors :py:meth:`Tree.test`, replacing the Dirichlet leaf draws by
        NIG posterior draws and the classification metrics by regression
        metrics. Top-k trees are ranked by per-tree train MSE (lower is
        better).

        Parameters
        ----------
        samples : list of dict
            Terminating tree states sampled from the policy.
        top_k_trees : int
            If > 0, additionally report top-k and top-1 metrics and plots.
        plot_top_k : bool
            If True and ``top_k_trees > 0``, include figures of the top-k
            trees in the returned ``figs`` dict (uses ``self.display``).
        seed : int, optional
            RNG seed for the posterior draws (reproducibility).

        Returns
        -------
        dict
            ``{"metrics": {...}, "figs": {...}}``.
        """
        if not samples:
            return {"metrics": {}, "figs": {}}
        if self.X_train is None or self.y_train is None:
            return {"metrics": {}, "figs": {}}

        if isinstance(samples, torch.Tensor):
            samples = list(samples)
        states: List[Dict] = list(samples)
        n_states = len(states)

        rng = np.random.default_rng(seed)

        # --- Per-state NIG posterior draw of leaf means (from train data) ---
        leaf_means_list = [self._sample_leaf_params(s, rng) for s in states]

        # --- Train predictions / scores ---
        train_preds = np.stack(
            [
                self._predict(states[i], leaf_means_list[i], self.X_train)
                for i in range(n_states)
            ],
            axis=0,
        )  # (n_states, n_train)
        train_scores = RegressionTree._compute_tree_scores(train_preds, self.y_train)

        result_metrics: Dict[str, float] = {
            "mean_n_nodes": float(np.mean([sum(s["_dones"]) for s in states]))
        }
        for key, val in train_scores.items():
            result_metrics[f"train_{key}"] = val

        # --- Top-k ranking by per-tree train MSE (ascending) ---
        top_k_indices = None
        figs: Dict[str, object] = {}
        if top_k_trees > 0 and n_states > 0:
            top_k_trees_eff = min(top_k_trees, n_states)
            per_tree_mse = np.array(
                [
                    mean_squared_error(self.y_train, train_preds[i])
                    for i in range(n_states)
                ]
            )
            order = np.argsort(per_tree_mse)
            top_k_indices = order[:top_k_trees_eff]

            top_k_scores = RegressionTree._compute_tree_scores(
                train_preds[top_k_indices], self.y_train
            )
            for key, val in top_k_scores.items():
                result_metrics[f"train_top_k_{key}"] = val

            top_1_idx = int(top_k_indices[0])
            top_1_scores = RegressionTree._compute_tree_scores(
                train_preds[[top_1_idx]], self.y_train
            )
            for key, val in top_1_scores.items():
                result_metrics[f"train_top_1_{key}"] = val

            if plot_top_k:
                for rank, idx in enumerate(top_k_indices):
                    try:
                        fig = self.display(state=states[int(idx)])
                    except Exception:
                        fig = None
                    if fig is not None:
                        figs[
                            f"top_{rank + 1}_tree_mse_{per_tree_mse[int(idx)]:.4f}"
                        ] = fig

        # --- Test split, if available ---
        if self.X_test is not None and self.y_test is not None:
            test_preds = np.stack(
                [
                    self._predict(states[i], leaf_means_list[i], self.X_test)
                    for i in range(n_states)
                ],
                axis=0,
            )
            test_scores = RegressionTree._compute_tree_scores(test_preds, self.y_test)
            for key, val in test_scores.items():
                result_metrics[f"test_{key}"] = val

            if top_k_indices is not None:
                top_k_scores = RegressionTree._compute_tree_scores(
                    test_preds[top_k_indices], self.y_test
                )
                for key, val in top_k_scores.items():
                    result_metrics[f"test_top_k_{key}"] = val

                top_1_scores = RegressionTree._compute_tree_scores(
                    test_preds[[int(top_k_indices[0])]], self.y_test
                )
                for key, val in top_1_scores.items():
                    result_metrics[f"test_top_1_{key}"] = val

        return {"metrics": result_metrics, "figs": figs}
