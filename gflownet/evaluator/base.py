"""
Base evaluator class for a :class:`~gflownet.gflownet.GFlowNetAgent`.

In charge of evaluating a generic :class:`~gflownet.gflownet.GFlowNetAgent`,
computing metrics plotting figures and optionally logging results using the
:class:`~gflownet.gflownet.GFlowNetAgent`'s :class:`~gflownet.utils.logger.Logger`.

Take this :class:`BaseEvaluator` as example to implement your own evaluator class
for your custom use-case.

.. important::

    Prefer the :meth:`~gflownet.evaluator.abstract.AbstractEvaluator.from_dir`
    and :meth:`~gflownet.evaluator.abstract.AbstractEvaluator.from_agent`
    class methods to instantiate an evaluator.

See :ref:`using an evaluator` for more details about how to use an Evaluator.
"""

import copy
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.special import logsumexp

from gflownet.evaluator.abstract import ALL_REQS  # noqa
from gflownet.evaluator.abstract import METRICS  # noqa
from gflownet.evaluator.abstract import AbstractEvaluator
from gflownet.utils.batch import Batch
from gflownet.utils.common import batch_with_rest, tfloat, torch2np


class BaseEvaluator(AbstractEvaluator):

    def __init__(self, gfn_agent=None, **config):
        """
        Base evaluator class for GFlowNetAgent.

        In particular, implements the :meth:`eval` with:

        - :meth:`compute_log_prob_metrics` to compute log-probability metrics.
        - :meth:`compute_density_metrics` to compute density metrics.

        And the :meth:`plot` method with:

        - The :class:`~gflownet.envs.base.GFlowNetEnv`'s :meth:`plot_reward_samples`
          method.
        - The :class:`~gflownet.envs.base.GFlowNetEnv`'s :meth:`plot_kde` method if
          it exists, for both the ``kde_pred`` and ``kde_true`` arguments if they are
          returned in the ``"data"`` dict of the :meth:`eval` method.

        See the :class:`~gflownet.evaluator.abstract.AbstractEvaluator` for more
        details about other methods and attributes, including the
        :meth:`~gflownet.evaluator.abstract.AbstractEvaluator.__init__`.
        """
        super().__init__(gfn_agent, **config)

    def define_new_metrics(self):
        return {
            "l1": {
                "display_name": "L1 error",
                "requirements": ["density"],
            },
            "kl": {
                "display_name": "KL Div.",
                "requirements": ["density"],
            },
            "jsd": {
                "display_name": "Jensen Shannon Div.",
                "requirements": ["density"],
            },
            "corr_prob_traj_rewards": {
                "display_name": "Corr. (test probs., rewards)",
                "requirements": ["log_probs", "reward_batch"],
            },
            "var_logrewards_logp": {
                "display_name": "Var(logR - logp) test",
                "requirements": ["log_probs", "reward_batch"],
            },
            "nll_tt": {
                "display_name": "NLL of test data",
                "requirements": ["log_probs"],
            },
            "mean_logprobs_std": {
                "display_name": "Mean BS Std(logp)",
                "requirements": ["log_probs"],
            },
            "mean_probs_std": {
                "display_name": "Mean BS Std(p)",
                "requirements": ["log_probs"],
            },
            "logprobs_std_nll_ratio": {
                "display_name": "BS Std(logp) / NLL",
                "requirements": ["log_probs"],
            },
        }

    # TODO: this method will most likely crash if used (top_k_period != -1) because
    # self.gfn.env.top_k_metrics_and_plots still makes use of env.proxy.
    # Re-implementing this wil require a non-trivial amount of work.
    @torch.no_grad()
    def eval_top_k(self, it, gfn_states=None, random_states=None):
        """
        Sample from the current GFN and compute metrics and plots for the top k states
        according to both the energy and the reward.

        Parameters
        ----------
        it : int
            current iteration
        gfn_states : list, optional
            Already sampled gfn states. Defaults to None.
        random_states : list, optional
            Already sampled random states. Defaults to None.

        Returns
        -------
        dict
            Computed dict of metrics, and figures, and optionally (only once) summary
            metrics. Schema: ``{"metrics": {str: float}, "figs": {str: plt.Figure},
            "summary": {str: float}}``.
        """
        # only do random top k plots & metrics once
        do_random = it // self.logger.test.top_k_period == 1
        duration = None
        summary = {}
        # TODO: Why deepcopy?
        prob = copy.deepcopy(self.random_action_prob)
        print()
        if not gfn_states:
            # sample states from the current gfn
            batch = Batch(
                env=self.gfn.env,
                proxy=self.gfn.proxy,
                device=self.gfn.device,
                float_type=self.gfn.float,
            )
            self.gfn.random_action_prob = 0
            t = time.time()
            print("Sampling from GFN...", end="\r")
            for b in batch_with_rest(
                0, self.gfn.logger.test.n_top_k, self.gfn.batch_size_total
            ):
                sub_batch, _ = self.gfn.sample_batch(n_forward=len(b), train=False)
                batch.merge(sub_batch)
            duration = time.time() - t
            gfn_states = batch.get_terminating_states()

        # compute metrics and get plots
        print("[eval_top_k] Making GFN plots...", end="\r")
        metrics, figs, fig_names = self.gfn.env.top_k_metrics_and_plots(
            gfn_states, self.gfn.logger.test.top_k, name="gflownet", step=it
        )
        if duration:
            metrics["gflownet top k sampling duration"] = duration

        if do_random:
            # sample random states from uniform actions
            if not random_states:
                batch = Batch(
                    env=self.gfn.env,
                    proxy=self.gfn.proxy,
                    device=self.gfn.device,
                    float_type=self.gfn.float,
                )
                self.gfn.random_action_prob = 1.0
                print("[eval_top_k] Sampling at random...", end="\r")
                for b in batch_with_rest(
                    0, self.gfn.logger.test.n_top_k, self.gfn.batch_size_total
                ):
                    sub_batch, _ = self.gfn.sample_batch(n_forward=len(b), train=False)
                    batch.merge(sub_batch)
            # compute metrics and get plots
            random_states = batch.get_terminating_states()
            print("[eval_top_k] Making Random plots...", end="\r")
            (
                random_metrics,
                random_figs,
                random_fig_names,
            ) = self.gfn.env.top_k_metrics_and_plots(
                random_states, self.gfn.logger.test.top_k, name="random", step=None
            )
            # add to current metrics and plots
            summary.update(random_metrics)
            figs += random_figs
            fig_names += random_fig_names
            # compute training data metrics and get plots
            print("[eval_top_k] Making train plots...", end="\r")
            (
                train_metrics,
                train_figs,
                train_fig_names,
            ) = self.gfn.env.top_k_metrics_and_plots(
                None, self.gfn.logger.test.top_k, name="train", step=None
            )
            # add to current metrics and plots
            summary.update(train_metrics)
            figs += train_figs
            fig_names += train_fig_names

        self.gfn.random_action_prob = prob

        print(" " * 100, end="\r")
        print("eval_top_k metrics:")
        max_k = max([len(k) for k in (list(metrics.keys()) + list(summary.keys()))]) + 1
        print(
            "  •  "
            + "\n  •  ".join(
                f"{k:{max_k}}: {v:.4f}"
                for k, v in (list(metrics.items()) + list(summary.items()))
            )
        )
        print()

        figs = {f: n for f, n in zip(figs, fig_names)}

        return {
            "metrics": metrics,
            "figs": figs,
            "summary": summary,
        }

    def compute_log_prob_metrics(self, x_tt, metrics=None):
        """
        Compute log-probability metrics for the given test data.

        Uses :meth:`~gflownet.gflownet.GFlowNetAgent.estimate_logprobs_data`.

        Known metrics:

        - ``mean_logprobs_std``: Mean of the standard deviation of the log-probabilities.
        - ``mean_probs_std``: Mean of the standard deviation of the probabilities.
        - ``corr_prob_traj_rewards``: Correlation between the probabilities and the
            rewards.
        - ``var_logrewards_logp``: Variance of the log-rewards minus the log-probabilities.
        - ``nll_tt``: Negative log-likelihood of the test data.
        - ``logprobs_std_nll_ratio``: Ratio of the mean of the standard deviation of the
            log-probabilities over the negative log-likelihood of the test data.


        Parameters
        ----------
        x_tt : torch.Tensor
            Test data.
        metrics : List[str], optional
            List of metrics to compute, by default ``None`` i.e. the evaluator's
            ``self.metrics``

        Returns
        -------
        dict
            Computed dict of metrics and data as ``{"metrics": {str: float}}``.
        """
        metrics = self.make_metrics(metrics)
        reqs = self.make_requirements(metrics=metrics)

        logprobs_x_tt, logprobs_std, probs_std = self.gfn.estimate_logprobs_data(
            x_tt,
            n_trajectories=self.config.n_trajs_logprobs,
            max_data_size=self.config.max_data_logprobs,
            batch_size=self.config.logprobs_batch_size,
            bs_num_samples=self.config.logprobs_bootstrap_size,
        )

        lp_metrics = {}

        if "mean_logprobs_std" in metrics:
            lp_metrics["mean_logprobs_std"] = logprobs_std.mean().item()

        if "mean_probs_std" in metrics:
            lp_metrics["mean_probs_std"] = probs_std.mean().item()

        if "reward_batch" in reqs:
            rewards_x_tt = self.gfn.proxy.rewards(self.gfn.env.states2proxy(x_tt))

            if "corr_prob_traj_rewards" in metrics:
                lp_metrics["corr_prob_traj_rewards"] = np.corrcoef(
                    np.exp(logprobs_x_tt.cpu().numpy()), rewards_x_tt
                )[0, 1]

            if "var_logrewards_logp" in metrics:
                lp_metrics["var_logrewards_logp"] = torch.var(
                    torch.log(
                        tfloat(
                            rewards_x_tt,
                            float_type=self.gfn.float,
                            device=self.gfn.device,
                        )
                    )
                    - logprobs_x_tt
                ).item()
        if "nll_tt" in metrics:
            lp_metrics["nll_tt"] = -logprobs_x_tt.mean().item()

        if "logprobs_std_nll_ratio" in metrics:
            lp_metrics["logprobs_std_nll_ratio"] = (
                -logprobs_std.mean() / logprobs_x_tt.mean()
            ).item()

        return {
            "metrics": lp_metrics,
        }

    def compute_density_metrics(self, x_tt, dict_tt, metrics=None):
        """
        Compute density metrics for the given test data.

        Known metrics:

        - ``l1``: L1 error between the true and predicted densities.
        - ``kl``: KL divergence between the true and predicted densities.
        - ``jsd``: Jensen-Shannon divergence between the true and predicted densities.

        Returned data in the ``"data"`` sub-dict:

        - ``x_sampled``: Sampled states from the GFN.
        - ``kde_pred``: KDE policy as per
          :meth:`~gflownet.envs.base.GFlowNetEnv.fit_kde`.
        - ``kde_true``: True KDE.

        Parameters
        ----------
        x_tt : torch.Tensor
            Test data.
        dict_tt : dict
            Dictionary of test data.
        metrics : List[str], optional
            List of metrics to compute, by default ``None`` i.e. the evaluator's
            ``self.metrics``

        Returns
        -------
        dict
            Computed dict of metrics and data as
            ``{"metrics": {str: float}, "data": {str: object}}``.
        """
        metrics = self.make_metrics(metrics)

        density_metrics = {}
        density_data = {}

        x_sampled = density_true = density_pred = None

        if self.gfn.buffer.test_type == "all":
            batch, _ = self.gfn.sample_batch(n_forward=self.config.n, train=False)
            assert batch.is_valid()
            x_sampled = batch.get_terminating_states()

            if "density_true" in dict_tt:
                density_true = torch2np(dict_tt["density_true"])
            else:
                rewards = torch2np(
                    self.gfn.proxy.rewards(self.gfn.env.states2proxy(x_tt))
                )
                z_true = rewards.sum()
                density_true = rewards / z_true
                with open(self.gfn.buffer.test_pkl, "wb") as f:
                    dict_tt["density_true"] = density_true
                    pickle.dump(dict_tt, f)
            hist = defaultdict(int)
            for x in x_sampled:
                hist[tuple(x)] += 1
            z_pred = sum([hist[tuple(x)] for x in x_tt]) + 1e-9
            density_pred = np.array([hist[tuple(x)] / z_pred for x in x_tt])
            log_density_true = np.log(density_true + 1e-8)
            log_density_pred = np.log(density_pred + 1e-8)

        elif self.gfn.continuous and hasattr(self.gfn.env, "fit_kde"):
            batch, _ = self.gfn.sample_batch(n_forward=self.config.n, train=False)
            assert batch.is_valid()
            x_sampled = batch.get_terminating_states(proxy=True)
            # TODO make it work with conditional env
            x_tt = torch2np(self.gfn.env.states2proxy(x_tt))
            kde_pred = self.gfn.env.fit_kde(
                x_sampled,
                kernel=self.config.kde.kernel,
                bandwidth=self.config.kde.bandwidth,
            )
            if "log_density_true" in dict_tt and "kde_true" in dict_tt:
                log_density_true = dict_tt["log_density_true"]
                kde_true = dict_tt["kde_true"]
            else:
                # Sample from reward via rejection sampling
                x_from_reward = self.gfn.env.states2proxy(
                    self.gfn.sample_from_reward(n_samples=self.config.n)
                )
                # Fit KDE with samples from reward
                kde_true = self.gfn.env.fit_kde(
                    x_from_reward,
                    kernel=self.config.kde.kernel,
                    bandwidth=self.config.kde.bandwidth,
                )
                # Estimate true log density using test samples
                # TODO: this may be specific-ish for the torus or not
                scores_true = kde_true.score_samples(x_tt)
                log_density_true = scores_true - logsumexp(scores_true, axis=0)
                # Add log_density_true and kde_true to pickled test dict
                with open(self.gfn.buffer.test_pkl, "wb") as f:
                    dict_tt["log_density_true"] = log_density_true
                    dict_tt["kde_true"] = kde_true
                    pickle.dump(dict_tt, f)
            # Estimate pred log density using test samples
            # TODO: this may be specific-ish for the torus or not
            scores_pred = kde_pred.score_samples(x_tt)
            log_density_pred = scores_pred - logsumexp(scores_pred, axis=0)
            density_true = np.exp(log_density_true)
            density_pred = np.exp(log_density_pred)

            density_data["kde_pred"] = kde_pred
            density_data["kde_true"] = kde_true

        else:
            density_metrics["l1"] = self.gfn.l1
            density_metrics["kl"] = self.gfn.kl
            density_metrics["jsd"] = self.gfn.jsd
            density_data["x_sampled"] = x_sampled
            return {
                "metrics": density_metrics,
                "data": density_data,
            }

        # L1 error
        density_metrics["l1"] = np.abs(density_pred - density_true).mean()
        # KL divergence
        density_metrics["kl"] = (
            density_true * (log_density_true - log_density_pred)
        ).mean()
        # Jensen-Shannon divergence
        log_mean_dens = np.logaddexp(log_density_true, log_density_pred) + np.log(0.5)
        density_metrics["jsd"] = 0.5 * np.sum(
            density_true * (log_density_true - log_mean_dens)
        )
        density_metrics["jsd"] += 0.5 * np.sum(
            density_pred * (log_density_pred - log_mean_dens)
        )

        density_data["x_sampled"] = x_sampled

        return {
            "metrics": density_metrics,
            "data": density_data,
        }

    def eval(self, metrics=None, **plot_kwargs):
        """
        Evaluate the GFlowNetAgent and compute metrics and plots.

        If `metrics` is not provided, the evaluator's `self.metrics` attribute is used
        (default).

        Extand in subclasses to add more metrics and plots:

        .. code-block:: python

            def eval(self, metrics=None, **plot_kwargs):
                result = super().eval(metrics=metrics, **plot_kwargs)
                result["metrics"]["my_custom_metric"] = my_custom_metric_function()
                result["figs"]["My custom plot"] = my_custom_plot_function()
                return result

        Parameters
        ----------
        metrics : List[str], optional
            List of metrics to compute, by default the evaluator's `self.metrics`
            attribute.
        plot_kwargs : dict, optional
            Additional keyword arguments to pass to the plotting methods.

        Returns
        -------
        dict
            Computed dict of metrics and figures as
            `{"metrics": {str: float}, "figs": {str: plt.Figure}}`.
        """
        metrics = self.make_metrics(metrics)
        reqs = self.make_requirements(metrics=metrics)

        if self.gfn.buffer.test_pkl is None:
            return {
                "metrics": {
                    k: getattr(self.gfn, k) if hasattr(self.gfn, k) else None
                    for k in metrics
                },
                "data": {},
            }

        all_data = {}
        all_metrics = {}

        with open(self.gfn.buffer.test_pkl, "rb") as f:
            dict_tt = pickle.load(f)
            x_tt = dict_tt["x"]

        # Compute correlation between the rewards of the test data and the log
        # likelihood of the data according the the GFlowNet policy; and NLL.
        # TODO: organise code for better efficiency and readability
        if "log_probs" in reqs:
            lp_results = self.compute_log_prob_metrics(x_tt, metrics=metrics)
            all_metrics.update(lp_results.get("metrics", {}))
            all_data.update(lp_results.get("data", {}))

        if "density" in reqs:
            density_results = self.compute_density_metrics(
                x_tt, dict_tt, metrics=metrics
            )
            all_metrics.update(density_results.get("metrics", {}))
            all_data.update(density_results.get("data", {}))

        return {
            "metrics": all_metrics,
            "data": all_data,
        }

    def plot(
        self, x_sampled=None, kde_pred=None, kde_true=None, plot_kwargs={}, **kwargs
    ):
        """
        Plots this evaluator should do, returned as a dict `{str: plt.Figure}` which
        will be logged.

        By default, this method will call the following methods of the GFlowNetAgent's
        environment if they exist:

        - `plot_reward_samples`
        - `plot_kde` (for both the `kde_pred` and `kde_true` arguments)
        - `plot_samples_topk`

        Extend this method to add more plots:

        .. code-block:: python

            def plot(self, x_sampled, kde_pred, kde_true, plot_kwargs, **kwargs):
                figs = super().plot(x_sampled, kde_pred, kde_true, plot_kwargs)
                figs["My custom plot"] = my_custom_plot_function(x_sampled, kde_pred)
                return figs

        Parameters
        ----------
        x_sampled : list, optional
            List of sampled states.
        kde_pred : sklearn.neighbors.KernelDensity
            KDE policy as per `Environment.fit_kde`
        kde_true : object
            True KDE.
        plot_kwargs : dict
            Additional keyword arguments to pass to the plotting methods.
        kwargs : dict
            Catch-all for additional arguments.

        Returns
        -------
        dict[str, plt.Figure]
            Dictionary of figures to be logged. The keys are the figure names and the
            values are the figures.
        """

        fig_kde_pred = fig_kde_true = fig_reward_samples = fig_samples_topk = None

        if hasattr(self.gfn.env, "plot_reward_samples") and x_sampled is not None:
            (sample_space_batch, rewards_sample_space) = (
                self.gfn.get_sample_space_and_reward()
            )
            fig_reward_samples = self.gfn.env.plot_reward_samples(
                x_sampled,
                sample_space_batch,
                rewards_sample_space,
                **plot_kwargs,
            )

        if hasattr(self.gfn.env, "plot_kde"):
            sample_space_batch, _ = self.gfn.get_sample_space_and_reward()
            if kde_pred is not None:
                fig_kde_pred = self.gfn.env.plot_kde(
                    sample_space_batch, kde_pred, **plot_kwargs
                )
            if kde_true is not None:
                fig_kde_true = self.gfn.env.plot_kde(
                    sample_space_batch, kde_true, **plot_kwargs
                )

        # TODO: consider moving this to eval_top_k once fixed
        if hasattr(self.gfn.env, "plot_samples_topk"):
            if x_sampled is None:
                batch, _ = self.gfn.sample_batch(
                    n_forward=self.config.n_top_k, train=False
                )
                x_sampled = batch.get_terminating_states()
            rewards = self.gfn.proxy.rewards(self.gfn.env.states2proxy(x_sampled))
            fig_samples_topk = self.gfn.env.plot_samples_topk(
                x_sampled,
                rewards,
                self.config.top_k,
                **plot_kwargs,
            )

        return {
            "True reward and GFlowNet samples": fig_reward_samples,
            "GFlowNet KDE Policy": fig_kde_pred,
            "Reward KDE": fig_kde_true,
            "Samples TopK": fig_samples_topk,
        }
