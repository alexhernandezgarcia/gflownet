import copy
import os
import pickle
import time
from collections import defaultdict
from typing import Union

import numpy as np
import torch
from scipy.special import logsumexp

from gflownet.utils.batch import Batch
from gflownet.utils.common import (
    batch_with_rest,
    load_gflow_net_from_run_path,
    tfloat,
    torch2np,
)

_sentinel = object()

METRICS = {
    "l1": {
        "name": "L1 error",
        "requirements": ["density"],
    },
    "kl": {
        "name": "KL Div.",
        "requirements": ["density"],
    },
    "jsd": {
        "name": "Jensen Shannon Div.",
        "requirements": ["density"],
    },
    "corr_prob_traj_rewards": {
        "name": "Corr. (test probs., rewards)",
        "requirements": ["log_probs", "reward_batch"],
    },
    "var_logrewards_logp": {
        "name": "Var(logR - logp) test",
        "requirements": ["log_probs", "reward_batch"],
    },
    "nll_tt": {
        "name": "NLL of test data",
        "requirements": ["log_probs"],
    },
    "mean_logprobs_std": {
        "name": "Mean BS Std(logp)",
        "requirements": ["log_probs"],
    },
    "mean_probs_std": {
        "name": "Mean BS Std(p)",
        "requirements": ["log_probs"],
    },
    "logprobs_std_nll_ratio": {
        "name": "BS Std(logp) / NLL",
        "requirements": ["log_probs"],
    },
}


class GFlowNetEvaluator:
    def __init__(self, **kwargs):
        """
        Base evaluator class for GFlowNetAgent.

        In charge of evaluating the GFlowNetAgent, computing metrics plotting figures
        and optionally logging results using the GFlowNetAgent's logger.

        Only the `from_dir` and `from_agent` class methods should be used to instantiate
        this class.

        Raises
        ------
        NotImplementedError
            If the `sentinel` keyword argument is not `_sentinel`, which is used to
            prevent instantiation of the base class without using the `from_dir` or
            `from_agent` class methods.

        """
        if kwargs.get("sentinel") is not _sentinel:
            raise NotImplementedError(
                "Base evaluator class should not be instantiated. Use "
                + "GFlowNetEvaluator.from_dir or GFlowNetEvaluator.from_agent methods."
            )
        self.gfn_agent = kwargs.get("gfn_agent")
        self.config = self.gfn_agent.eval_config
        self.logger = self.gfn_agent.logger
        self.reqs = set()

        self.metrics = self.reqs = _sentinel
        self.metrics = self.make_metrics(self.config.metrics)
        self.reqs = self.make_requirements()

    def make_metrics(self, metrics=None):
        """
        Parse metrics from a list, a string or None.

        If `None`, all metrics are computed. If a string, it can be a comma-separated
        list of metric names, with or without spaces. All metrics must be in `METRICS`.

        Parameters
        ----------
        metrics : (Union[str, List[str]], optional)
            Metrics to compute when running the `evaluator.eval()` function. Defaults to
            None, i.e. all metrics in `METRICS` are computed.

        Returns
        -------
        dict
            Dictionary of metrics to compute, with the metric names as keys and the
            metric names and requirements as values.

        Raises
        ------
            ValueError
                If a metric name is not in `METRICS`.
        """
        if metrics == "all":
            metrics = METRICS.keys()

        if metrics is None:
            assert self.metrics is not _sentinel, (
                "Error setting self.metrics. This is likely due to the `metrics:`"
                + " entry missing from your eval config. Set it to 'all' to compute all"
                + " metrics or to a comma-separated list of metric names (eg 'l1, kl')."
            )
            return self.metrics

        if isinstance(metrics, str):
            if "," in metrics:
                metrics = [m.strip() for m in metrics.split(",")]
            else:
                metrics = [metrics]

        for m in metrics:
            if m not in METRICS:
                raise ValueError(f"Unknown metric name: {m}")

        return {m: METRICS[m] for m in metrics}

    def make_requirements(self, reqs=None, metrics=None):
        """
        Make requirements for the metrics to compute.

        1. If `metrics` is provided, they must be as a dict of metrics. The requirements
           are computed from the `requirements` attribute of the metrics.

        2. Otherwise, the requirements are computed from the `reqs` argument:
            - If `reqs` is `"all"`, all requirements of all metrics are computed.
            - If `reqs` is `None`, the evaluator's `self.reqs` attribute is
              used.
            - If `reqs` is a list, it is used as the requirements.

        Parameters
        ----------
        reqs : Union[str, List[str]], optional
            The metrics requirements. Either `"all"`, a list of requirements or `None`
            to use the evaluator's `self.reqs` attribute. By default None
        metrics : List[str], optional
            The list of metrics dicts to compute requirements for. By default None.

        Returns
        -------
        set[str]
            The set of requirements for the metrics.
        """

        if metrics is not None:
            return set([r for m in metrics.values() for r in m["requirements"]])

        if reqs == "all":
            reqs = set([r for m in METRICS.values() for r in m["requirements"]])
        if reqs is None:
            if self.reqs is _sentinel:
                self.reqs = set(
                    [r for m in self.metrics.values() for r in m["requirements"]]
                )
            reqs = self.reqs
        if isinstance(reqs, list):
            reqs = set(reqs)

        assert isinstance(reqs, set), f"reqs should be a set, but is {type(reqs)}"
        assert all([isinstance(r, str) for r in reqs]), (
            "All elements of reqs should be strings, but are "
            + f"{[type(r) for r in reqs]}"
        )

        return reqs

    def should_log_train(self, step):
        """
        Check if training logs should be done at the current step. The decision is based
        on the `self.config.train.period` attribute.

        Set `self.config.train.period` to `None` or a negative value to disable
        training.

        Parameters
        ----------
        step : int
            Current iteration step.

        Returns
        -------
        bool
            True if training should be done at the current step, False otherwise.
        """
        if self.config.train_log_period is None or self.config.train_log_period < 0:
            return False
        else:
            return not step % self.config.train_log_period

    def should_eval(self, step):
        """
        Check if testing should be done at the current step. The decision is based on
        the `self.config.test.period` attribute.

        Set `self.config.test.first_it` to `True` if testing should be done at the first
        iteration step. Otherwise, testing will be done aftter `self.config.test.period`
        steps.

        Set `self.config.test.period` to `None` or a negative value to disable testing.

        Parameters
        ----------
        step : int
            Current iteration step.

        Returns
        -------
        bool
            True if testing should be done at the current step, False otherwise.
        """
        if self.config.period is None or self.config.period < 0:
            return False
        elif step == 1 and self.config.first_it:
            return True
        else:
            return not step % self.config.period

    def should_eval_top_k(self, step):
        """
        Check if top k plots and metrics should be done at the current step. The
        decision is based on the `self.config.test.top_k` and
        `self.config.test.top_k_period` attributes.

        Set `self.config.test.top_k` to `None` or a negative value to disable top k
        plots and metrics.

        Parameters
        ----------
        step : int
            Current iteration step.

        Returns
        -------
        bool
            True if top k plots and metrics should be done at the current step, False
        """
        if self.config.top_k is None or self.config.top_k < 0:
            return False

        if self.config.top_k_period is None or self.config.top_k_period < 0:
            return False

        return step == 2 or step % self.config.top_k_period == 0

    def should_checkpoint(self, step):
        """
        Check if checkpoints should be done at the current step. The decision is based
        on the `self.checkpoints.period` attribute.

        Set `self.checkpoints.period` to `None` or a negative value to disable
        checkpoints.

        Parameters
        ----------
        step : int
            Current iteration step.

        Returns
        -------
        bool
            True if checkpoints should be done at the current step, False otherwise.
        """
        if self.config.checkpoints_period is None or self.config.checkpoints_period < 0:
            return False
        else:
            return not step % self.config.checkpoints_period

    @classmethod
    def from_dir(
        cls: "GFlowNetEvaluator",
        path: Union[str, os.PathLike],
        no_wandb: bool = True,
        print_config: bool = False,
        device: str = "cuda",
        load_final_ckpt: bool = True,
    ):
        """
        Instantiate a GFlowNetEvaluator from a run directory.

        Parameters
        ----------
        cls : GFlowNetEvaluator
            Class to instantiate.
        path : Union[str, os.PathLike]
            Path to the run directory from which to load the GFlowNetAgent.
        no_wandb : bool, optional
            Prevent wandb initialization, by default True
        print_config : bool, optional
            Whether or not to print the resulting (loaded) config, by default False
        device : str, optional
            Device to use for the instantiated GFlowNetAgent, by default "cuda"
        load_final_ckpt : bool, optional
            Use the latest possible checkpoint available in the path, by default True

        Returns
        -------
        GFlowNetEvaluator
            Instance of GFlowNetEvaluator with the GFlowNetAgent loaded from the run.
        """
        gfn_agent, _ = load_gflow_net_from_run_path(
            path,
            no_wandb=no_wandb,
            print_config=print_config,
            device=device,
            load_final_ckpt=load_final_ckpt,
        )
        return GFlowNetEvaluator.from_agent(gfn_agent)

    @classmethod
    def from_agent(cls, gfn_agent):
        """
        Instantiate a GFlowNetEvaluator from a GFlowNetAgent.

        Parameters
        ----------
        cls : GFlowNetEvaluator
            Evaluator class to instantiate.
        gfn_agent : GFlowNetAgent
            Instance of GFlowNetAgent to use for the GFlowNetEvaluator.

        Returns
        -------
        GFlowNetEvaluator
            Instance of GFlowNetEvaluator with the provided GFlowNetAgent.
        """
        from gflownet.gflownet import GFlowNetAgent

        assert isinstance(gfn_agent, GFlowNetAgent), (
            "gfn_agent should be an instance of GFlowNetAgent, but is an instance of "
            + f"{type(gfn_agent)}."
        )

        return GFlowNetEvaluator(gfn_agent=gfn_agent, sentinel=_sentinel)

    def plot(self, x_sampled=None, kde_pred=None, kde_true=None, **plot_kwargs):
        """
        Plots this evaluator should do, returned as a dict `{str: plt.Figure}` which
        will be logged.

        By default, this method will call the `plot_reward_samples` method of the
        GFlowNetAgent's environment, and the `plot_kde` method of the GFlowNetAgent's
        environment if it exists for both the `kde_pred` and `kde_true` arguments.

        Extend this method to add more plots:

        ```python
        def plot(self, x_sampled, kde_pred, kde_true, **plot_kwargs):
            figs = super().plot(x_sampled, kde_pred, kde_true, **plot_kwargs) figs["My
            custom plot"] = my_custom_plot_function(x_sampled, kde_pred) return figs
        ```

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
            )

        return lp_metrics

    def compute_density_metrics(self, x_tt, dict_tt, metrics=None):
        gfn = self.gfn_agent
        metrics = self.make_metrics(metrics)
        reqs = self.make_requirements(metrics=metrics)  # TODO-V: unused for now, TBD

        density_metrics = {}
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

        else:
            density_metrics["l1"] = gfn.l1
            density_metrics["kl"] = gfn.kl
            density_metrics["jsd"] = gfn.jsd
            density_metrics["x_sampled"] = x_sampled
            density_metrics["kde_pred"] = kde_pred
            density_metrics["kde_true"] = kde_true
            return density_metrics

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

        density_metrics["x_sampled"] = x_sampled
        density_metrics["kde_pred"] = kde_pred
        density_metrics["kde_true"] = kde_true

        return density_metrics

    def eval(self, metrics=None, **plot_kwargs):
        """
        Evaluate the GFlowNetAgent and compute metrics and plots.

        If `metrics` is not provided, the evaluator's `self.metrics` attribute is used
        (default).

        Extand in subclasses to add more metrics and plots:

        ```python
        def eval(self, metrics=None, **plot_kwargs):
            result = super().eval(metrics=metrics, **plot_kwargs)
            result["metrics"]["my_custom_metric"] = my_custom_metric_function()
            result["figs"]["My custom plot"] = my_custom_plot_function()
            return result
        ```

        Parameters
        ----------
        metrics : List[str], optional
            List of metrics to compute, by default the evaluator's `self.metrics`
            attribute.
        plot_kwargs : dict, optional
            Additional keyword arguments to pass to the plotting methods.

        Returns
        -------
        list
            List of computed metrics and figures: [l1, kl, jsd, corr_prob_traj_rewards,
            var_logrewards_logp, nll_tt, mean_logprobs_std, mean_probs_std,
            logprobs_std_nll_ratio, figs] (should be refactored to dict) #TODO fix docstring
        """
        gfn = self.gfn_agent
        metrics = self.make_metrics(metrics)
        reqs = self.make_requirements(metrics=metrics)

        all_metrics = {}
        x_sampled = kde_pred = kde_true = None
        figs = {}

        if gfn.buffer.test_pkl is None:
            result = {
                "metrics": {
                    k: getattr(gfn, k) if hasattr(gfn, k) else None for k in metrics
                },
                "figs": figs,
            }
            return result

        with open(gfn.buffer.test_pkl, "rb") as f:
            dict_tt = pickle.load(f)
            x_tt = dict_tt["x"]

        # Compute correlation between the rewards of the test data and the log
        # likelihood of the data according the the GFlowNet policy; and NLL.
        # TODO: organise code for better efficiency and readability
        if "log_probs" in reqs:
            lp_metrics = self.compute_log_prob_metrics(x_tt, metrics=metrics)
            all_metrics.update(lp_metrics)

        if "density" in reqs:
            density_metrics = self.compute_density_metrics(
                x_tt, dict_tt, metrics=metrics
            )
            x_sampled = density_metrics.pop("x_sampled", x_sampled)
            kde_pred = density_metrics.pop("kde_pred", kde_pred)
            kde_true = density_metrics.pop("kde_true", kde_true)
            all_metrics.update(density_metrics)

        figs = self.plot(x_sampled=x_sampled, kde_pred=kde_pred, kde_true=kde_true)

        return {
            "metrics": all_metrics,
            "figs": figs,
        }

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
        tuple[dict, dict[str, plt.Figure], dict]
            Computed dict of metrics, and figures (as {str: plt.Figure}), and optionally
            (only once) summary metrics.
        """
        # only do random top k plots & metrics once
        do_random = it // self.logger.test.top_k_period == 1
        duration = None
        summary = {}
        prob = copy.deepcopy(self.random_action_prob)
        gfn = self.gfn_agent
        print()
        if not gfn_states:
            # sample states from the current gfn
            batch = Batch(env=gfn.env, device=gfn.device, float_type=gfn.float)
            gfn.random_action_prob = 0
            t = time.time()
            print("Sampling from GFN...", end="\r")
            for b in batch_with_rest(0, gfn.logger.test.n_top_k, gfn.batch_size_total):
                sub_batch, _ = gfn.sample_batch(n_forward=len(b), train=False)
                batch.merge(sub_batch)
            duration = time.time() - t
            gfn_states = batch.get_terminating_states()

        # compute metrics and get plots
        print("[eval_top_k] Making GFN plots...", end="\r")
        metrics, figs, fig_names = gfn.env.top_k_metrics_and_plots(
            gfn_states, gfn.logger.test.top_k, name="gflownet", step=it
        )
        if duration:
            metrics["gflownet top k sampling duration"] = duration

        if do_random:
            # sample random states from uniform actions
            if not random_states:
                batch = Batch(env=gfn.env, device=gfn.device, float_type=gfn.float)
                gfn.random_action_prob = 1.0
                print("[eval_top_k] Sampling at random...", end="\r")
                for b in batch_with_rest(
                    0, gfn.logger.test.n_top_k, gfn.batch_size_total
                ):
                    sub_batch, _ = gfn.sample_batch(n_forward=len(b), train=False)
                    batch.merge(sub_batch)
            # compute metrics and get plots
            random_states = batch.get_terminating_states()
            print("[eval_top_k] Making Random plots...", end="\r")
            (
                random_metrics,
                random_figs,
                random_fig_names,
            ) = gfn.env.top_k_metrics_and_plots(
                random_states, gfn.logger.test.top_k, name="random", step=None
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
            ) = gfn.env.top_k_metrics_and_plots(
                None, gfn.logger.test.top_k, name="train", step=None
            )
            # add to current metrics and plots
            summary.update(train_metrics)
            figs += train_figs
            fig_names += train_fig_names

        gfn.random_action_prob = prob

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

        return metrics, figs, summary

    def eval_and_log(self, it, metrics=None):
        """
        Evaluate the GFlowNetAgent and log the results with its logger.

        Will call `self.eval()` and log the results using the GFlowNetAgent's logger
        `log_metrics()` and `log_plots()` methods.

        Parameters
        ----------
        it : int
            Current iteration step.
        metrics : Union[str, List[str]], optional
            List of metrics to compute, by default the evaluator's `metrics` attribute.
        """
        gfn = self.gfn_agent
        result = self.eval(metrics=metrics)
        for m, v in result["metrics"].items():
            setattr(gfn, m, v)

        mertics_to_log = {METRICS[k]["name"]: v for k, v in result["metrics"].items()}

        self.logger.log_metrics(mertics_to_log, it, gfn.use_context)
        self.logger.log_plots(result["figs"], it, use_context=gfn.use_context)

    def eval_and_log_top_k(self, it):
        """
        Evaluate the GFlowNetAgent's top k samples performance and log the results with
        its logger.

        Parameters
        ----------
        it : int
            Current iteration step, by default None.
        """

        metrics, figs, summary = self.eval_top_k(it)
        self.logger.log_plots(figs, it, use_context=self.use_context)
        self.logger.log_metrics(metrics, use_context=self.use_context, step=it)
        self.logger.log_summary(summary)


if __name__ == "__main__":
    # dev test case, will move to tests
    from pathlib import Path

    scratch = Path(os.environ["SCRATCH"])
    run_dirs = scratch / "crystals/logs/icml24/crystalgfn"
    gfn_run_dir = run_dirs / "4074836/2024-01-27_20-54-55/5908fe41"

    gfne = GFlowNetEvaluator.from_dir(gfn_run_dir)
    gfne.plot()
    gfne.compute_metrics()
