"""
Base evaluator class for GFlowNetAgent.

In charge of evaluating the GFlowNetAgent, computing metrics plotting figures and
optionally logging results using the GFlowNetAgent's logger.

.. important::

    Only the :py:meth:`~gflownet.evaluator.abstract.GFlowNetAbstractEvaluator.from_dir`
    and :py:meth:`~gflownet.evaluator.abstract.GFlowNetAbstractEvaluator.from_agent`
    class methods should be used to instantiate this class.

Create a new evaluator by subclassing this class and extending the :py:meth:`eval`
method to add more metrics and plots.

Typical call stack:

1. :py:meth:`gflownet.gflownet.GFlowNetAgent.train` calls the evaluator's

2. :py:meth:`~gflownet.evaluator.base.GFlowNetEvaluator.should_eval`.
   If it returns ``True`` then :py:meth:`~gflownet.gflownet.GFlowNetAgent.train` calls

3. :py:meth:`~gflownet.evaluator.base.GFlowNetEvaluator.eval_and_log` which itself calls

4. :py:meth:`~gflownet.evaluator.base.GFlowNetEvaluator.eval` as
   ``results = self.eval(metrics=None)`` and then
   ``figs = self.plot(**results["data"])``

5. finally, :py:meth:`~gflownet.evaluator.base.GFlowNetEvaluator.eval_and_log` logs the
   results using the GFlowNetAgent's logger as
   ``self.logger.log_metrics(results["metrics"])`` and ``self.logger.log_plots(figs)``.

Example
-------

.. code-block:: python

    # How to create a new evaluator:
    from gflownet.evaluator.base import GFlowNetEvaluator

    gfn_run_dir = "PUT_YOUR_RUN_DIR_HERE"  # a run dir contains a .hydra folder
    gfne = GFlowNetEvaluator.from_dir(gfn_run_dir)
    results = gfne.eval()

    for name, metric in results["metrics"].items():
        print(f"{name:20}: {metric:.4f}")

    data = results.get("data", {})

    plots = gfne.plot(**data)

    print(
        "Available figures in plots:",
        ", ".join([fname for fname, fig in plots.items() if fig is not None])
        or "None",
    )


.. code-block:: python

    # gflownet/evaluator/my_evaluator.py
    from gflownet.evaluator.base import GFlowNetEvaluator, METRICS, ALL_REQS

    class MyEvaluator(GFlowNetEvaluator):
        def update_all_metrics_and_requirements(self):
            global METRICS, ALL_REQS

            METRICS["my_custom_metric"] = {
                "display_name": "My custom metric",
                "requirements": ["density", "new_req"],
            }

            ALL_REQS = set([r for m in METRICS.values() for r in m["requirements"]])


        def my_custom_metric(self, some, arguments):
            intermediate = some + arguments

            return {
                "metrics": {
                    "my_custom_metric": intermediate ** (-0.5)
                },
                "data": {
                    "some_other": some ** 2,
                    "arguments": arguments,
                    "intermediate": intermediate,
                }
            }
            ...

        def my_custom_plot(
            self, some_other=None, arguments=None, intermediate=None, **kwargs
        ):
            # whatever gets to **kwargs will be ignored, this is used to handle
            # methods with varying signatures.
            figs = {}
            if some_other is not None:
                f = plt.figure()
                # some plotting procedure for some_other
                figs["My Title"] = f

                if arguments is not None:
                    f = plt.figure()
                    # some other plotting procedure with both
                    figs["My Other Title"] = f
            elif arguments is not None:
                f = plt.figure()
                # some other plotting procedure with arguments
                figs["My 3rd Title"] = f

            if intermediate is not None:
                f = plt.figure()
                # some other plotting procedure with intermediate
                figs["My 4th Title"] = f

            return figs

        def plot(self, **kwargs):
            figs = super().plot(**kwargs)
            figs.update(self.my_custom_plot(**kwargs))

            return figs

        def eval(self, metrics=None, **plot_kwargs):
            gfn = self.gfn_agent
            metrics = self.make_metrics(metrics)
            reqs = self.make_requirements(metrics=metrics)

            results = super().eval(metrics=metrics, **plot_kwargs)

            if "new_req" in reqs:
                my_results = self.my_custom_metric(some, arguments)
                results["metrics"].update(my_results.get("metrics", {}))
                results["data"].update(my_results.get("data", {}))

            return results

In the previous example, the `update_all_metrics_and_requirements` method is used to
update the global `METRICS` and `ALL_REQS` variables. It will be called when the
`MyEvaluator` class is instantiated, in the init of `BaseEvaluator`.

By defining a new requirement, you ensure that the new metrics and plots will only be
computed if user asks for a metric that requires such computations.

By default, the train loop will call
:py:meth:`~gflownet.evaluator.base.GFlowNetEvaluator.eval_and_log` which itself calls
:py:meth:`~gflownet.evaluator.base.GFlowNetEvaluator.eval` so if you override ``eval()``
as above, the new metrics and plots will be computed and logged.

Similarly, `eval_and_log` will compute the ``dict`` of figures as
``fig_dict = self.plot(**results["data"])`` where ``results`` is the output of ``eval``.
"""

import pickle
from collections import defaultdict

import numpy as np
import torch
from scipy.special import logsumexp

from gflownet.evaluator.abstract import ALL_REQS  # noqa
from gflownet.evaluator.abstract import METRICS  # noqa
from gflownet.evaluator.abstract import GFlowNetAbstractEvaluator
from gflownet.utils.common import tfloat, torch2np


class GFlowNetEvaluator(GFlowNetAbstractEvaluator):

    def compute_log_prob_metrics(self, x_tt, metrics=None):
        gfn = self.gfn_agent
        metrics = self.make_metrics(metrics)
        reqs = self.make_requirements(metrics=metrics)

        logprobs_x_tt, logprobs_std, probs_std = gfn.estimate_logprobs_data(
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
            rewards_x_tt = gfn.env.reward_batch(x_tt)

            if "corr_prob_traj_rewards" in metrics:
                rewards_x_tt = gfn.env.reward_batch(x_tt)
                lp_metrics["corr_prob_traj_rewards"] = np.corrcoef(
                    np.exp(logprobs_x_tt.cpu().numpy()), rewards_x_tt
                )[0, 1]

            if "var_logrewards_logp" in metrics:
                rewards_x_tt = gfn.env.reward_batch(x_tt)
                lp_metrics["var_logrewards_logp"] = torch.var(
                    torch.log(
                        tfloat(rewards_x_tt, float_type=gfn.float, device=gfn.device)
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
        gfn = self.gfn_agent
        metrics = self.make_metrics(metrics)

        density_metrics = {}
        density_data = {}

        x_sampled = density_true = density_pred = None

        if gfn.buffer.test_type is not None and gfn.buffer.test_type == "all":
            batch, _ = gfn.sample_batch(n_forward=self.config.n, train=False)
            assert batch.is_valid()
            x_sampled = batch.get_terminating_states()

            if "density_true" in dict_tt:
                density_true = dict_tt["density_true"]
            else:
                rewards = gfn.env.reward_batch(x_tt)
                z_true = rewards.sum()
                density_true = rewards / z_true
                with open(gfn.buffer.test_pkl, "wb") as f:
                    dict_tt["density_true"] = density_true
                    pickle.dump(dict_tt, f)
            hist = defaultdict(int)
            for x in x_sampled:
                hist[tuple(x)] += 1
            z_pred = sum([hist[tuple(x)] for x in x_tt]) + 1e-9
            density_pred = np.array([hist[tuple(x)] / z_pred for x in x_tt])
            log_density_true = np.log(density_true + 1e-8)
            log_density_pred = np.log(density_pred + 1e-8)

        elif gfn.continuous and hasattr(gfn.env, "fit_kde"):
            batch, _ = gfn.sample_batch(n_forward=self.config.n, train=False)
            assert batch.is_valid()
            x_sampled = batch.get_terminating_states()
            # TODO make it work with conditional env
            x_sampled = torch2np(gfn.env.states2proxy(x_sampled))
            x_tt = torch2np(gfn.env.states2proxy(x_tt))
            kde_pred = gfn.env.fit_kde(
                x_sampled,
                kernel=self.config.kde.kernel,
                bandwidth=self.config.kde.bandwidth,
            )
            if "log_density_true" in dict_tt and "kde_true" in dict_tt:
                log_density_true = dict_tt["log_density_true"]
                kde_true = dict_tt["kde_true"]
            else:
                # Sample from reward via rejection sampling
                x_from_reward = gfn.env.sample_from_reward(n_samples=self.config.n)
                x_from_reward = torch2np(gfn.env.states2proxy(x_from_reward))
                # Fit KDE with samples from reward
                kde_true = gfn.env.fit_kde(
                    x_from_reward,
                    kernel=self.config.kde.kernel,
                    bandwidth=self.config.kde.bandwidth,
                )
                # Estimate true log density using test samples
                # TODO: this may be specific-ish for the torus or not
                scores_true = kde_true.score_samples(x_tt)
                log_density_true = scores_true - logsumexp(scores_true, axis=0)
                # Add log_density_true and kde_true to pickled test dict
                with open(gfn.buffer.test_pkl, "wb") as f:
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
            density_metrics["l1"] = gfn.l1
            density_metrics["kl"] = gfn.kl
            density_metrics["jsd"] = gfn.jsd
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
        gfn = self.gfn_agent
        metrics = self.make_metrics(metrics)
        reqs = self.make_requirements(metrics=metrics)

        if gfn.buffer.test_pkl is None:
            return {
                "metrics": {
                    k: getattr(gfn, k) if hasattr(gfn, k) else None for k in metrics
                },
                "data": {},
            }

        all_data = {}
        all_metrics = {}

        with open(gfn.buffer.test_pkl, "rb") as f:
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

        By default, this method will call the `plot_reward_samples` method of the
        GFlowNetAgent's environment, and the `plot_kde` method of the GFlowNetAgent's
        environment if it exists for both the `kde_pred` and `kde_true` arguments.

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
        gfn = self.gfn_agent

        fig_kde_pred = fig_kde_true = fig_reward_samples = None

        if hasattr(gfn.env, "plot_reward_samples") and x_sampled is not None:
            fig_reward_samples = gfn.env.plot_reward_samples(x_sampled, **plot_kwargs)

        if hasattr(gfn.env, "plot_kde"):
            if kde_pred is not None:
                fig_kde_pred = gfn.env.plot_kde(kde_pred, **plot_kwargs)
            if kde_true is not None:
                fig_kde_true = gfn.env.plot_kde(kde_true, **plot_kwargs)

        return {
            "True reward and GFlowNet samples": fig_reward_samples,
            "GFlowNet KDE Policy": fig_kde_pred,
            "Reward KDE": fig_kde_true,
        }


if __name__ == "__main__":
    # Try using the GFlowNetEvaluator by running this script from the root:
    # $ ipython
    # In [1]: run gflownet/evaluator/base.py
    #
    # Note: this will not work on previous checkpoints whose config does not contain an
    # `eval` entry, you have to run one. Add `eval.checkpoint_period=10` to quickly
    # have a checkpoint to test.

    gfn_run_dir = "PUT_YOUR_RUN_DIR_HERE"  # a run dir contains a .hydra folder

    gfne = GFlowNetEvaluator.from_dir(gfn_run_dir)
    results = gfne.eval()

    for name, metric in results["metrics"].items():
        print(f"{name:20}: {metric:.4f}")

    print(
        "Available figures in results['figs']:",
        ", ".join([fname for fname, fig in results["figs"].items() if fig is not None])
        or "None",
    )
