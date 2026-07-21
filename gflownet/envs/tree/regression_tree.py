"""
Regression tree composite environment.

A RegressionTree is a :class:`~gflownet.envs.tree.tree.Tree` whose targets are
continuous instead of categorical. The MDP (states, actions, masks, steps) is
identical to the classification Tree: the environment only constructs the tree
*structure* (decision rules) and (currently) not the leaf parameters, they are
only sampled at inference.

The differences with respect to the classification Tree are:

- The targets ``y_train`` / ``y_test`` are kept as floats.
- The targets are standardized (zero mean, unit variance, computed on the
  train split) by default, as is standard practice for Bayesian CART/BART.
  Besides making the default NIG hyper-parameters sensible, this keeps the
  magnitude of the marginal log-likelihood small enough that the exponential
  reward ``exp(beta * log_posterior)`` does not underflow to zero (which
  breaks GFlowNet training, e.g. NaNs in weighted replay sampling).
- The evaluation pass (:py:meth:`test`) reports regression metrics (RMSE, R2)
  instead of accuracies. Leaf predictions are drawn from the posterior of a
  Normal-Inverse-Gamma (NIG) leaf model, mirroring how the classification Tree
  draws leaf class probabilities from a Dirichlet posterior:

      y | mu, sigma^2 ~ N(mu, sigma^2)  at each leaf, with conjugate prior
      mu | sigma^2 ~ N(mu_0, sigma^2 / kappa_0), sigma^2 ~ InvGamma(alpha_0, beta_0)

This is the standard conjugate leaf model of Bayesian CART for regression
(Chipman et al., 1997), and matches the marginal likelihood computed by
``gflownet.proxy.regression_tree.NormalGammaTreeProxy``.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import mean_squared_error, r2_score

from gflownet.envs.tree.tree import Tree


class RegressionTree(Tree):
    """
    Composite environment for binary regression tree construction.

    All construction logic is inherited from :class:`Tree`; see its docstring
    for the state and action formats. This subclass only intercepts the target
    variable (kept continuous) and provides regression-specific evaluation.
    """

    def __init__(
        self,
        X_train: Optional[npt.NDArray] = None,
        y_train: Optional[npt.NDArray] = None,
        X_test: Optional[npt.NDArray] = None,
        y_test: Optional[npt.NDArray] = None,
        data_path: Optional[str] = None,
        scale_y: bool = True,
        **kwargs,
    ):
        """
        See :py:meth:`Tree.__init__` for all parameters. The differences are
        that ``y_train`` / ``y_test`` are interpreted as continuous targets,
        and that ``scale_y`` (True by default) standardizes them with the
        train-split mean and standard deviation (stored as ``y_mean_`` /
        ``y_std_``). RMSE metrics reported by :py:meth:`test` are rescaled
        back to the original target units.
        """
        # Keep a float copy of the targets before the parent casts them to int.
        # The feature matrices and feature names are handled by the parent.
        y_train_cont = None
        y_test_cont = None
        if X_train is not None and y_train is not None:
            y_train_cont = np.asarray(y_train, dtype=float)
            if X_test is not None and y_test is not None:
                y_test_cont = np.asarray(y_test, dtype=float)
        elif data_path is not None:
            _, y_tr, _, y_te, _ = Tree._load_dataset(data_path)
            y_train_cont = np.asarray(y_tr, dtype=float)
            if y_te is not None:
                y_test_cont = np.asarray(y_te, dtype=float)

        super().__init__(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            data_path=data_path,
            **kwargs,
        )

        # Restore the continuous targets discarded by the parent's int cast
        if y_train_cont is not None:
            self.y_train = y_train_cont
        if y_test_cont is not None:
            self.y_test = y_test_cont

        # Standardize the targets with the train-split statistics
        self.scale_y = scale_y
        self.y_mean_ = 0.0
        self.y_std_ = 1.0
        if scale_y and self.y_train is not None:
            self.y_mean_ = float(np.mean(self.y_train))
            std = float(np.std(self.y_train))
            self.y_std_ = std if std > 0.0 else 1.0
            self.y_train = (self.y_train - self.y_mean_) / self.y_std_
            if self.y_test is not None:
                self.y_test = (self.y_test - self.y_mean_) / self.y_std_

    # =========================================================================
    # Normal-Inverse-Gamma leaf model
    # =========================================================================

    @staticmethod
    def _nig_posterior(
        y: npt.NDArray,
        mu_0: float,
        kappa_0: float,
        alpha_0: float,
        beta_0: float,
    ) -> Tuple[float, float, float, float]:
        """
        Returns the parameters ``(mu_n, kappa_n, alpha_n, beta_n)`` of the
        Normal-Inverse-Gamma posterior over ``(mu, sigma^2)`` given the samples
        ``y`` at a leaf.
        """
        n = len(y)
        ybar = float(np.mean(y))
        kappa_n = kappa_0 + n
        mu_n = (kappa_0 * mu_0 + n * ybar) / kappa_n
        alpha_n = alpha_0 + 0.5 * n
        ss = float(np.sum((y - ybar) ** 2))
        beta_n = beta_0 + 0.5 * ss + 0.5 * kappa_0 * n * (ybar - mu_0) ** 2 / kappa_n
        return mu_n, kappa_n, alpha_n, beta_n

    def _resolve_nig_params(
        self,
        mu_0: Optional[float],
        kappa_0: float,
        alpha_0: float,
        beta_0: Optional[float],
    ) -> Tuple[float, float, float, float]:
        """
        Resolves data-driven defaults for the NIG hyper-parameters:

        - ``mu_0``: mean of the training targets if None.
        - ``beta_0``: ``(alpha_0 - 1) * var(y_train)`` if None (so that the
          prior mean of sigma^2 equals the target variance, for alpha_0 > 1).
        """
        if mu_0 is None:
            mu_0 = float(np.mean(self.y_train))
        if beta_0 is None:
            var = float(np.var(self.y_train))
            if var <= 0.0:
                var = 1.0
            beta_0 = (alpha_0 - 1.0) * var if alpha_0 > 1.0 else var
        return float(mu_0), float(kappa_0), float(alpha_0), float(beta_0)

    def _sample_leaf_nig(
        self,
        state: Dict,
        mu_0: float,
        kappa_0: float,
        alpha_0: float,
        beta_0: float,
        rng: np.random.Generator,
    ) -> Dict[int, float]:
        """
        Draws one posterior sample of the leaf mean at every leaf of ``state``,
        using routing on ``self.X_train``.

        For each leaf reached by some training samples ``y``, draws
        ``sigma^2 ~ InvGamma(alpha_n, beta_n)`` and then
        ``mu ~ N(mu_n, sigma^2 / kappa_n)``. This is the regression analogue of
        :py:meth:`Tree._sample_leaf_dirichlet`.
        """
        leaf_samples = self._route_samples(state, self.X_train)
        means: Dict[int, float] = {}
        for k, idx in leaf_samples.items():
            y = self.y_train[idx]
            mu_n, kappa_n, alpha_n, beta_n = RegressionTree._nig_posterior(
                y, mu_0, kappa_0, alpha_0, beta_0
            )
            # If G ~ Gamma(alpha_n, 1), then beta_n / G ~ InvGamma(alpha_n, beta_n)
            sigma2 = beta_n / rng.gamma(alpha_n)
            means[k] = float(rng.normal(mu_n, math.sqrt(sigma2 / kappa_n)))
        return means

    def _predict(
        self,
        state: Dict,
        leaf_means: Dict[int, float],
        X: npt.NDArray,
        default: float,
    ) -> npt.NDArray:
        """
        Returns a vector of predictions for ``X`` routed through ``state`` with
        leaf means ``leaf_means``. Samples landing at a leaf never visited
        during training get the ``default`` prediction (the prior mean).
        """
        leaf_samples = self._route_samples(state, X)
        preds = np.full(len(X), default, dtype=float)
        for k, idx in leaf_samples.items():
            if k in leaf_means:
                preds[idx] = leaf_means[k]
        return preds

    @staticmethod
    def _compute_tree_scores_regression(
        preds: npt.NDArray, y: npt.NDArray
    ) -> Dict[str, float]:
        """
        Given a ``(n_trees, n_samples)`` matrix of predictions and the target
        vector, returns:

        - ``mean_tree_rmse`` / ``mean_tree_r2``: mean of per-tree scores.
        - ``forest_rmse`` / ``forest_r2``: scores of the ensemble prediction
          obtained by averaging predictions across trees (the Bayesian model
          average of the posterior-mean predictors).
        """
        per_tree_rmse = [
            math.sqrt(mean_squared_error(y, preds[i])) for i in range(preds.shape[0])
        ]
        per_tree_r2 = [r2_score(y, preds[i]) for i in range(preds.shape[0])]
        forest_pred = preds.mean(axis=0)
        return {
            "mean_tree_rmse": float(np.mean(per_tree_rmse)),
            "mean_tree_r2": float(np.mean(per_tree_r2)),
            "forest_rmse": float(math.sqrt(mean_squared_error(y, forest_pred))),
            "forest_r2": float(r2_score(y, forest_pred)),
        }

    # =========================================================================
    # Evaluation: train/test regression metrics of sampled trees
    # =========================================================================

    def test(
        self,
        samples: List[Dict],
        alpha: Optional[npt.NDArray] = None,
        alpha_value: float = 1.0,
        top_k_trees: int = 0,
        plot_top_k: bool = True,
        seed: Optional[int] = None,
        mu_0: Optional[float] = None,
        kappa_0: float = 0.1,
        alpha_0: float = 2.0,
        beta_0: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Evaluates a batch of sampled terminating trees with regression metrics.

        The procedure mirrors :py:meth:`Tree.test`:

        1. For each sampled tree, draw one posterior sample of the per-leaf
           mean from the Normal-Inverse-Gamma posterior given the training
           samples routed to that leaf.
        2. Compute predictions for ``X_train`` (and ``X_test`` if available).
        3. Report ``mean_tree_rmse`` / ``mean_tree_r2`` (per-tree scores
           averaged over the ensemble) and ``forest_rmse`` / ``forest_r2``
           (scores of the prediction averaged across trees).
        4. If ``top_k_trees > 0``, rank the trees by train RMSE (ascending) and
           additionally report metrics on the top-k subset and the top-1 tree,
           optionally with plots.

        Parameters
        ----------
        samples : list of dict
            Terminating tree states sampled from the policy.
        alpha : np.ndarray, optional
            Unused. Present for signature compatibility with
            :py:meth:`Tree.test` (and hence the TreeEvaluator).
        alpha_value : float
            Unused. Present for signature compatibility with the TreeEvaluator.
        top_k_trees : int
            If > 0, additionally report top-k and top-1 metrics and plots.
        plot_top_k : bool
            If True and ``top_k_trees > 0``, include figures of the top-k
            trees in the returned ``figs`` dict (uses ``self.display``).
        seed : int, optional
            RNG seed for the posterior draws (reproducibility).
        mu_0, kappa_0, alpha_0, beta_0 : float, optional
            NIG hyper-parameters. ``mu_0`` and ``beta_0`` default to
            data-driven values (see :py:meth:`_resolve_nig_params`).

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

        mu_0, kappa_0, alpha_0, beta_0 = self._resolve_nig_params(
            mu_0, kappa_0, alpha_0, beta_0
        )
        rng = np.random.default_rng(seed)

        # --- Per-state NIG draw of leaf means (from train data) ---
        leaf_means_list = [
            self._sample_leaf_nig(s, mu_0, kappa_0, alpha_0, beta_0, rng)
            for s in states
        ]

        # --- Train predictions / scores ---
        train_preds = np.stack(
            [
                self._predict(states[i], leaf_means_list[i], self.X_train, mu_0)
                for i in range(n_states)
            ],
            axis=0,
        )  # (n_states, n_train)
        train_scores = RegressionTree._compute_tree_scores_regression(
            train_preds, self.y_train
        )

        result_metrics: Dict[str, float] = {
            "mean_n_nodes": float(np.mean([sum(s["_dones"]) for s in states]))
        }
        for key, val in train_scores.items():
            result_metrics[f"train_{key}"] = val

        # --- Top-k ranking by per-tree train RMSE (lower is better) ---
        top_k_indices = None
        figs: Dict[str, object] = {}
        if top_k_trees > 0 and n_states > 0:
            top_k_trees_eff = min(top_k_trees, n_states)
            per_tree_rmse = np.array(
                [
                    math.sqrt(mean_squared_error(self.y_train, train_preds[i]))
                    for i in range(n_states)
                ]
            )
            order = np.argsort(per_tree_rmse)
            top_k_indices = order[:top_k_trees_eff]

            top_k_scores = RegressionTree._compute_tree_scores_regression(
                train_preds[top_k_indices], self.y_train
            )
            for key, val in top_k_scores.items():
                result_metrics[f"train_top_k_{key}"] = val

            top_1_idx = int(top_k_indices[0])
            top_1_scores = RegressionTree._compute_tree_scores_regression(
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
                            f"top_{rank + 1}_tree_rmse_{per_tree_rmse[int(idx)]:.4f}"
                        ] = fig

        # --- Test split, if available ---
        if self.X_test is not None and self.y_test is not None:
            test_preds = np.stack(
                [
                    self._predict(states[i], leaf_means_list[i], self.X_test, mu_0)
                    for i in range(n_states)
                ],
                axis=0,
            )
            test_scores = RegressionTree._compute_tree_scores_regression(
                test_preds, self.y_test
            )
            for key, val in test_scores.items():
                result_metrics[f"test_{key}"] = val

            if top_k_indices is not None:
                top_k_scores = RegressionTree._compute_tree_scores_regression(
                    test_preds[top_k_indices], self.y_test
                )
                for key, val in top_k_scores.items():
                    result_metrics[f"test_top_k_{key}"] = val

                top_1_scores = RegressionTree._compute_tree_scores_regression(
                    test_preds[[int(top_k_indices[0])]], self.y_test
                )
                for key, val in top_1_scores.items():
                    result_metrics[f"test_top_1_{key}"] = val

        # Report RMSE metrics in the original target units (R2 is invariant
        # under the linear target standardization)
        if self.y_std_ != 1.0:
            for key in result_metrics:
                if "rmse" in key:
                    result_metrics[key] *= self.y_std_

        return {"metrics": result_metrics, "figs": figs}
