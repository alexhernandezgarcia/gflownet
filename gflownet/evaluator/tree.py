"""
Evaluator for the composite Tree environment.

Extends :class:`~gflownet.evaluator.base.BaseEvaluator` with tree-specific
train/test accuracy metrics, replicating the legacy ``Tree.test`` reporting
(mean_tree / forest / top_k / top_1 accuracy and balanced accuracy, on both
train and test splits). Metrics and top-k tree plots are pushed to the
GFlowNetAgent's logger (wandb) at the configured evaluation period.
"""

from typing import Optional

import torch

from gflownet.evaluator.base import BaseEvaluator


class TreeEvaluator(BaseEvaluator):
    """
    Base evaluator plus a ``Tree.test(samples)`` pass.

    Extra config keys (namespaced under ``tree_test`` in the yaml):

    - ``n_samples``: number of terminating trees to sample for the test pass.
    - ``top_k_trees``: if > 0, also log top-k / top-1 metrics and plots.
    - ``alpha_value``: Dirichlet prior concentration used when drawing leaf
      class probabilities.
    - ``plot_top_k``: whether to also log figures of the top-k trees.
    - ``seed``: RNG seed for Dirichlet draws (None = fresh each call).
    """

    @torch.no_grad()
    def eval_and_log(self, it: int, metrics: Optional[list] = None):
        # First, run the standard base evaluation (log_probs, density, etc.)
        super().eval_and_log(it, metrics=metrics)

        # Tree-specific evaluation
        if not hasattr(self.gfn.env, "test"):
            return

        tree_cfg = self.config.get("tree_test", None)
        if tree_cfg is None:
            return

        n_samples = int(tree_cfg.get("n_samples", 200))
        top_k_trees = int(tree_cfg.get("top_k_trees", 0))
        alpha_value = float(tree_cfg.get("alpha_value", 1.0))
        plot_top_k = bool(tree_cfg.get("plot_top_k", True))
        seed = tree_cfg.get("seed", None)
        seed = None if seed is None else int(seed)

        if n_samples <= 0:
            return

        # Sample terminating trees on-policy
        batch, _ = self.gfn.sample_batch(n_forward=n_samples, train=False)
        states = batch.get_terminating_states()

        result = self.gfn.env.test(
            states,
            alpha_value=alpha_value,
            top_k_trees=top_k_trees,
            plot_top_k=plot_top_k,
            seed=seed,
        )

        metrics_dict = result.get("metrics", {})
        figs_dict = result.get("figs", {})

        if metrics_dict:
            self.logger.log_metrics(
                metrics_dict, step=it, use_context=self.gfn.use_context
            )
        if figs_dict:
            self.logger.log_plots(figs_dict, step=it, use_context=self.gfn.use_context)
