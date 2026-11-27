"""
Regression tree composite environment.

A RegressionTree is a :class:`~gflownet.envs.tree.tree.Tree` whose targets are
continuous instead of categorical. The MDP (states, actions, masks, steps) is
identical to the classification Tree: the environment only constructs the tree
*structure* (decision rules) and (currently) not the leaf parameters, they are
only inferred at evaluation time.

The differences with respect to the classification Tree are:

- The targets ``y_train`` / ``y_test`` are kept as floats.
- The targets are standardized (zero mean, unit variance, computed on the
  train split) by default, as is standard practice for Bayesian CART/BART.
  Besides making the default NIG hyper-parameters sensible, this keeps the
  magnitude of the marginal log-likelihood small enough that the exponential
  reward ``exp(beta * log_posterior)`` does not underflow to zero (which
  breaks GFlowNet training, e.g. NaNs in weighted replay sampling).
- The evaluation pass (:py:meth:`test`) reports regression metrics (RMSE, R2,
  predictive NLL, interval coverage) instead of accuracies. The leaf model is
  a Normal-Inverse-Gamma (NIG) conjugate model, mirroring the Dirichlet leaf
  model of the classification Tree:

      y | mu, sigma^2 ~ N(mu, sigma^2)  at each leaf, with conjugate prior
      mu | sigma^2 ~ N(mu_0, sigma^2 / kappa_0), sigma^2 ~ InvGamma(alpha_0, beta_0)

  The leaf parameters ``(mu, sigma^2)`` are integrated out analytically:
  point predictions use the closed-form posterior predictive mean ``mu_n``
  of each leaf (no Monte-Carlo draws, so the evaluation is deterministic),
  and probabilistic metrics use the closed-form Student-t posterior
  predictive of each leaf. This mirrors the closed-form Dirichlet predictive
  used by ``gflownet/envs/tree/eval_tree.py`` for classification.

This is the standard conjugate leaf model of Bayesian CART for regression
(Chipman et al., 1998), and matches the marginal likelihood computed by
``gflownet.proxy.regression_tree.NormalGammaTreeProxy``.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from scipy.special import gammaln, logsumexp
from scipy.stats import t as student_t
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
        ``y_std_``). RMSE and NLL metrics reported by :py:meth:`test` are
        rescaled back to the original target units.
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
        beta_0: Optional[Union[float, str]],
    ) -> Tuple[float, float, float, float]:
        """
        Resolves data-driven defaults for the NIG hyper-parameters, with the
        same semantics as ``NormalGammaTreeProxy.setup``:

        - ``mu_0``: mean of the training targets if None.
        - ``beta_0``: ``(alpha_0 - 1) * scale`` (so that the prior mean of
          sigma^2 equals ``scale``, for alpha_0 > 1), where ``scale`` is
          ``var(y_train)`` if None, or the residual variance of an overfit
          greedy CART if ``"overfit"``.
        """
        if mu_0 is None:
            mu_0 = float(np.mean(self.y_train))
        if beta_0 is None or isinstance(beta_0, str):
            var = float(np.var(self.y_train))
            if var <= 0.0:
                var = 1.0
            if beta_0 is None:
                scale = var
            elif beta_0.lower() == "overfit":
                # Lazy import: the proxy module imports from gflownet.envs.tree
                from gflownet.proxy.regression_tree import NormalGammaTreeProxy

                scale = NormalGammaTreeProxy._overfit_residual_variance(
                    self.X_train, self.y_train, var
                )
            else:
                raise ValueError(
                    f"Unknown beta_0 option '{beta_0}'. "
                    f"Expected a float, None, or 'overfit'."
                )
            beta_0 = (alpha_0 - 1.0) * scale if alpha_0 > 1.0 else scale
        return float(mu_0), float(kappa_0), float(alpha_0), float(beta_0)

    def _fit_leaf_posteriors(
        self,
        state: Dict,
        mu_0: float,
        kappa_0: float,
        alpha_0: float,
        beta_0: float,
    ) -> Dict[int, Tuple[int, float, float, float, float]]:
        """
        Computes the NIG posterior at every leaf of ``state`` reached by some
        training samples, using routing on ``self.X_train``.

        Returns a dict mapping the leaf index ``k`` to
        ``(n, mu_n, kappa_n, alpha_n, beta_n)``, where ``n`` is the number of
        training samples routed to the leaf and the remaining entries are the
        NIG posterior parameters (see :py:meth:`_nig_posterior`).
        """
        leaf_samples = self._route_samples(state, self.X_train)
        posteriors: Dict[int, Tuple[int, float, float, float, float]] = {}
        for k, idx in leaf_samples.items():
            y = self.y_train[idx]
            mu_n, kappa_n, alpha_n, beta_n = RegressionTree._nig_posterior(
                y, mu_0, kappa_0, alpha_0, beta_0
            )
            posteriors[k] = (len(y), mu_n, kappa_n, alpha_n, beta_n)
        return posteriors

    @staticmethod
    def _student_t_params(
        mu: float, kappa: float, alpha: float, beta: float
    ) -> Tuple[float, float, float]:
        """
        Returns ``(loc, scale, df)`` of the Student-t posterior predictive
        of a NIG distribution with parameters ``(mu, kappa, alpha, beta)``:

            y* ~ t_{2 alpha}(mu, beta (kappa + 1) / (alpha kappa))
        """
        return mu, math.sqrt(beta * (kappa + 1.0) / (alpha * kappa)), 2.0 * alpha

    def _predictive_params(
        self,
        state: Dict,
        leaf_posteriors: Dict[int, Tuple[int, float, float, float, float]],
        X: npt.NDArray,
        mu_0: float,
        kappa_0: float,
        alpha_0: float,
        beta_0: float,
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Returns the per-sample Student-t posterior predictive parameters
        ``(loc, scale, df)`` (each of shape ``(len(X),)``) for ``X`` routed
        through ``state``. ``loc`` is the closed-form posterior predictive
        mean, i.e. the point prediction of the tree.

        Samples landing at a leaf never visited by training data get the
        *prior* predictive (a Student-t centered at ``mu_0``).
        """
        loc_0, scale_0, df_0 = RegressionTree._student_t_params(
            mu_0, kappa_0, alpha_0, beta_0
        )
        loc = np.full(len(X), loc_0, dtype=float)
        scale = np.full(len(X), scale_0, dtype=float)
        df = np.full(len(X), df_0, dtype=float)
        leaf_samples = self._route_samples(state, X)
        for k, idx in leaf_samples.items():
            if k in leaf_posteriors:
                _, mu_n, kappa_n, alpha_n, beta_n = leaf_posteriors[k]
                loc[idx], scale[idx], df[idx] = RegressionTree._student_t_params(
                    mu_n, kappa_n, alpha_n, beta_n
                )
        return loc, scale, df

    def _log_posterior(
        self,
        state: Dict,
        leaf_posteriors: Dict[int, Tuple[int, float, float, float, float]],
        kappa_0: float,
        alpha_0: float,
        beta_0: float,
    ) -> float:
        """
        Closed-form (unnormalized) log-posterior of the tree structure:
        the NIG marginal log-likelihood of the training targets (leaf
        parameters integrated out; same formula as
        ``NormalGammaTreeProxy._compute_log_likelihood``) plus the default
        "node_count" structure log-prior
        ``-(log 4 + log n_features) * n_internal``.

        Used to rank trees when no external ``log_posteriors`` are provided
        to :py:meth:`test`.
        """
        log_likelihood = 0.0
        for n, _, kappa_n, alpha_n, beta_n in leaf_posteriors.values():
            log_likelihood += (
                -0.5 * n * math.log(2.0 * math.pi)
                + 0.5 * (math.log(kappa_0) - math.log(kappa_n))
                + alpha_0 * math.log(beta_0)
                - alpha_n * math.log(beta_n)
                + float(gammaln(alpha_n) - gammaln(alpha_0))
            )
        n_internal = int(np.sum(state["_dones"]))
        n_features = self.X_train.shape[1]
        log_prior = -(math.log(4.0) + math.log(n_features)) * n_internal
        return log_likelihood + log_prior

    @staticmethod
    def _compute_tree_scores_regression(
        loc: npt.NDArray,
        scale: npt.NDArray,
        df: npt.NDArray,
        y: npt.NDArray,
    ) -> Dict[str, float]:
        """
        Given the ``(n_trees, n_samples)`` Student-t predictive parameters of
        an ensemble and the target vector, returns:

        - ``mean_tree_rmse`` / ``mean_tree_r2`` / ``mean_tree_nll``: mean of
          per-tree scores (how good a single sampled tree is on average).
        - ``forest_rmse`` / ``forest_r2``: scores of the ensemble point
          prediction obtained by averaging the posterior predictive means
          across trees (the uniform Monte-Carlo estimate of the Bayesian
          model average).
        - ``forest_nll``: negative log-likelihood of the BMA posterior
          predictive, i.e. the uniform mixture of the per-tree Student-t
          predictives.
        - ``forest_coverage_90``: fraction of targets falling inside the
          central 90% interval of the mixture predictive (well-calibrated
          uncertainties give ~0.90).
        """
        n_trees = loc.shape[0]
        preds = loc
        per_tree_rmse = [
            math.sqrt(mean_squared_error(y, preds[i])) for i in range(n_trees)
        ]
        per_tree_r2 = [r2_score(y, preds[i]) for i in range(n_trees)]
        forest_pred = preds.mean(axis=0)

        # Per-tree and mixture log predictive densities, (n_trees, n_samples)
        logpdf = student_t.logpdf(y[None, :], df, loc=loc, scale=scale)
        mixture_logpdf = logsumexp(logpdf, axis=0) - math.log(n_trees)
        # Mixture CDF at the true targets: y is inside the central 90%
        # interval of the mixture iff its CDF value is in [0.05, 0.95]
        mixture_cdf = student_t.cdf(y[None, :], df, loc=loc, scale=scale).mean(axis=0)
        coverage_90 = float(np.mean((mixture_cdf >= 0.05) & (mixture_cdf <= 0.95)))

        return {
            "mean_tree_rmse": float(np.mean(per_tree_rmse)),
            "mean_tree_r2": float(np.mean(per_tree_r2)),
            "mean_tree_nll": float(-np.mean(logpdf)),
            "forest_rmse": float(math.sqrt(mean_squared_error(y, forest_pred))),
            "forest_r2": float(r2_score(y, forest_pred)),
            "forest_nll": float(-np.mean(mixture_logpdf)),
            "forest_coverage_90": coverage_90,
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
        beta_0: Optional[Union[float, str]] = None,
        log_posteriors: Optional[npt.NDArray] = None,
    ) -> Dict[str, object]:
        """
        Evaluates a batch of sampled terminating trees with regression metrics.

        The procedure mirrors :py:meth:`Tree.test` and the closed-form
        classification evaluation of ``eval_tree.py``:

        1. For each sampled tree, compute the NIG posterior at every leaf
           from the training samples routed to it, and integrate the leaf
           parameters out analytically: point predictions are the posterior
           predictive means ``mu_n``, predictive distributions are the
           corresponding Student-t. No Monte-Carlo draws are involved, so
           the evaluation is deterministic.
        2. Compute predictions for ``X_train`` (and ``X_test`` if available).
        3. Report ``mean_tree_rmse`` / ``mean_tree_r2`` / ``mean_tree_nll``
           (per-tree scores averaged over the ensemble) and ``forest_rmse`` /
           ``forest_r2`` / ``forest_nll`` / ``forest_coverage_90`` (scores of
           the Bayesian model average: predictions averaged across trees,
           predictive density the uniform mixture of per-tree predictives).
        4. If ``top_k_trees > 0``, rank the trees by log-posterior
           (descending) and additionally report metrics on the top-k subset
           and the top-1 tree, optionally with plots.

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
            Deprecated and ignored (kept for signature compatibility): the
            evaluation is deterministic since the leaf parameters are
            integrated out in closed form.
        mu_0, kappa_0, alpha_0, beta_0 : float, optional
            NIG hyper-parameters. ``mu_0`` and ``beta_0`` default to
            data-driven values (see :py:meth:`_resolve_nig_params`).
        log_posteriors : np.ndarray, optional
            Per-tree log-posteriors used to rank the trees for the top-k
            metrics (e.g. computed by ``NormalGammaTreeProxy``, so that the
            ranking honors the structure prior the run was trained with). If
            None, computed internally as the NIG marginal log-likelihood
            plus the default "node_count" structure prior (see
            :py:meth:`_log_posterior`).

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

        # Per-state closed-form NIG leaf posteriors (from train data)
        leaf_posteriors_list = [
            self._fit_leaf_posteriors(s, mu_0, kappa_0, alpha_0, beta_0) for s in states
        ]

        # Per-tree log-posteriors for ranking (higher is better)
        if log_posteriors is None:
            log_posteriors = np.array(
                [
                    self._log_posterior(
                        states[i], leaf_posteriors_list[i], kappa_0, alpha_0, beta_0
                    )
                    for i in range(n_states)
                ]
            )
        else:
            log_posteriors = np.asarray(log_posteriors, dtype=float)

        # Train predictive parameters / scores (avg tree and forest)
        train_params = [
            self._predictive_params(
                states[i],
                leaf_posteriors_list[i],
                self.X_train,
                mu_0,
                kappa_0,
                alpha_0,
                beta_0,
            )
            for i in range(n_states)
        ]
        # Each of shape (n_states, n_train)
        train_loc = np.stack([p[0] for p in train_params], axis=0)
        train_scale = np.stack([p[1] for p in train_params], axis=0)
        train_df = np.stack([p[2] for p in train_params], axis=0)
        train_scores = RegressionTree._compute_tree_scores_regression(
            train_loc, train_scale, train_df, self.y_train
        )

        result_metrics: Dict[str, float] = {
            "mean_n_nodes": float(np.mean([sum(s["_dones"]) for s in states])),
            "mean_log_posterior": float(np.mean(log_posteriors)),
        }
        for key, val in train_scores.items():
            result_metrics[f"train_{key}"] = val

        # Top-k ranking by log-posterior (higher is better)
        top_k_indices = None
        figs: Dict[str, object] = {}
        if top_k_trees > 0 and n_states > 0:
            top_k_trees_eff = min(top_k_trees, n_states)
            order = np.argsort(-log_posteriors)
            top_k_indices = order[:top_k_trees_eff]

            top_k_scores = RegressionTree._compute_tree_scores_regression(
                train_loc[top_k_indices],
                train_scale[top_k_indices],
                train_df[top_k_indices],
                self.y_train,
            )
            for key, val in top_k_scores.items():
                result_metrics[f"train_top_k_{key}"] = val

            top_1_idx = int(top_k_indices[0])
            result_metrics["top_1_log_posterior"] = float(log_posteriors[top_1_idx])
            result_metrics["train_top_1_rmse"] = float(
                math.sqrt(mean_squared_error(self.y_train, train_loc[top_1_idx]))
            )
            result_metrics["train_top_1_r2"] = float(
                r2_score(self.y_train, train_loc[top_1_idx])
            )

            if plot_top_k:
                for rank, idx in enumerate(top_k_indices):
                    try:
                        fig = self.display(state=states[int(idx)])
                    except Exception:
                        fig = None
                    if fig is not None:
                        # Report the train RMSE in the original target units
                        # (predictions are computed on the standardized
                        # targets), consistent with the logged metrics.
                        rmse_orig = (
                            math.sqrt(
                                mean_squared_error(self.y_train, train_loc[int(idx)])
                            )
                            * self.y_std_
                        )
                        figs[
                            f"top_{rank + 1}_tree_logpost_"
                            f"{log_posteriors[int(idx)]:.1f}_rmse_{rmse_orig:.4f}"
                        ] = fig

        # Test split metrics
        if self.X_test is not None and self.y_test is not None:
            test_params = [
                self._predictive_params(
                    states[i],
                    leaf_posteriors_list[i],
                    self.X_test,
                    mu_0,
                    kappa_0,
                    alpha_0,
                    beta_0,
                )
                for i in range(n_states)
            ]
            test_loc = np.stack([p[0] for p in test_params], axis=0)
            test_scale = np.stack([p[1] for p in test_params], axis=0)
            test_df = np.stack([p[2] for p in test_params], axis=0)
            test_scores = RegressionTree._compute_tree_scores_regression(
                test_loc, test_scale, test_df, self.y_test
            )
            for key, val in test_scores.items():
                result_metrics[f"test_{key}"] = val

            if top_k_indices is not None:
                top_k_scores = RegressionTree._compute_tree_scores_regression(
                    test_loc[top_k_indices],
                    test_scale[top_k_indices],
                    test_df[top_k_indices],
                    self.y_test,
                )
                for key, val in top_k_scores.items():
                    result_metrics[f"test_top_k_{key}"] = val

                result_metrics["test_top_1_rmse"] = float(
                    math.sqrt(mean_squared_error(self.y_test, test_loc[top_1_idx]))
                )
                result_metrics["test_top_1_r2"] = float(
                    r2_score(self.y_test, test_loc[top_1_idx])
                )

        # Report RMSE and NLL metrics in the original target units (R2 and
        # coverage are invariant under the linear target standardization; the
        # density of the rescaled target picks up a 1/y_std_ Jacobian, i.e.
        # +log(y_std_) per sample in NLL)
        if self.y_std_ != 1.0:
            for key in result_metrics:
                if "rmse" in key:
                    result_metrics[key] *= self.y_std_
                elif "nll" in key:
                    result_metrics[key] += math.log(self.y_std_)

        return {"metrics": result_metrics, "figs": figs}
