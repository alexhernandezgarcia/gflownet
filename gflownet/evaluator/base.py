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

METRICS_NAMES = {
    "l1": "L1 error",
    "kl": "KL Div.",
    "jsd": "Jensen Shannon Div.",
    "corr_prob_traj_rewards": "Corr. (test probs., rewards)",
    "var_logrewards_logp": "Var(logR - logp) test",
    "nll_tt": "NLL of test data",
    "mean_logprobs_std": "Mean BS Std(logp)",
    "mean_probs_std": "Mean BS Std(p)",
    "logprobs_std_nll_ratio": "BS Std(logp) / NLL",
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

        self.set_metrics(self.config.metrics)

    def set_metrics(self, metrics=None):
        """
        Set the metrics to be computed by the evaluator to the `self.metrics` attribute.

        If `None`, all metrics are computed. If a string, it can be a comma-separated
        list of metric names, with or without spaces. All metrics must be in
        `METRICS_NAMES`.

        Parameters
        ----------
        metrics : (Union[str, List[str]], optional)
            Metrics to compute when running the `evaluator.eval()` function. Defaults to
            None, i.e. all metrics in `METRICS_NAMES` are computed.

        Raises
        ------
            ValueError
                If a metric name is not in `METRICS_NAMES`.
        """
        if metrics is None:
            metrics = METRICS_NAMES.keys()
        if isinstance(metrics, str):
            if "," in metrics:
                metrics = [m.strip() for m in metrics.split(",")]
            else:
                metrics = [metrics]
        for m in metrics:
            if m not in METRICS_NAMES:
                raise ValueError(f"Unknown metric name: {m}")
        self.metrics = metrics

    def do_train(self, step):
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
        if self.config.train.period is None or self.config.train.period < 0:
            return False
        else:
            return not step % self.config.train.period

    def do_test(self, step):
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
        if self.config.test.period is None or self.config.test.period < 0:
            return False
        elif step == 1 and self.config.test.first_it:
            return True
        else:
            return not step % self.config.test.period

    def do_top_k(self, step):
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
        if self.config.test.top_k is None or self.config.test.top_k < 0:
            return False

        if self.config.test.top_k_period is None or self.config.test.top_k_period < 0:
            return False

        return step == 2 or step % self.config.test.top_k_period == 0

    def do_oracle(self, step):
        """
        Check if oracle should be done at the current step. The decision is based on the
        `self.config.oracle.period` attribute.

        Set `self.config.oracle.period` to `None` or a negative value to disable oracle.

        Parameters
        ----------
        step : int
            Current iteration step.

        Returns
        -------
        bool
            True if oracle should be done at the current step, False otherwise.
        """
        if self.config.oracle.period is None or self.config.oracle.period < 0:
            return False
        else:
            return not step % self.oracle.period

    def do_checkpoints(self, step):
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
        if self.checkpoints.period is None or self.checkpoints.period < 0:
            return False
        else:
            return not step % self.checkpoints.period

    @staticmethod
    def from_dir(
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
        gfn_agent = load_gflow_net_from_run_path(
            path,
            no_wandb=no_wandb,
            print_config=print_config,
            device=device,
            load_final_ckpt=load_final_ckpt,
        )
        return GFlowNetEvaluator.from_agent(gfn_agent)

    @staticmethod
    def from_agent(gfn_agent):
        """
        Instantiate a GFlowNetEvaluator from a GFlowNetAgent.

        Parameters
        ----------
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

    def plot(self):
        """
        Plots this evaluator should do. This is a base method that does nothing and
        should be overridden by subclasses.
        """
        print("Base evaluator `plot()` method does not do anything.")

    def eval(self, metrics=_sentinel, **plot_kwargs):
        """
        Evaluate the GFlowNetAgent and compute metrics and plots.

        If `metrics` is not provided, the evaluator's `metrics` attribute is
        used (default).

        Parameters
        ----------
        metrics : _type_, optional
            List of metrics to compute, by default the evaluator's `metrics` attribute.
        plot_kwargs : dict, optional
            Additional keyword arguments to pass to the plotting methods.

        Returns
        -------
        list
            List of computed metrics and figures: [l1, kl, jsd, corr_prob_traj_rewards,
            var_logrewards_logp, nll_tt, mean_logprobs_std, mean_probs_std,
            logprobs_std_nll_ratio, figs, env_metrics] (should be refactored to dict)
        """
        gfn = self.gfn_agent

        if metrics is None:
            # TODO-V use this in the rest of the code to selectively compute metrics
            metrics = set(METRICS_NAMES.keys())

        if gfn.buffer.test_pkl is None:
            result = {
                "metrics": {
                    k: getattr(gfn, k) if hasattr(gfn, k) else None for k in metrics
                }
            }
            result["figs"] = (None,)
            result["env_metrics"] = {}
            return result.values()

        with open(gfn.buffer.test_pkl, "rb") as f:
            dict_tt = pickle.load(f)
            x_tt = dict_tt["x"]

        # Compute correlation between the rewards of the test data and the log
        # likelihood of the data according the the GFlowNet policy; and NLL.
        # TODO: organise code for better efficiency and readability
        logprobs_x_tt, logprobs_std, probs_std = gfn.estimate_logprobs_data(
            x_tt,
            n_trajectories=self.logger.test.n_trajs_logprobs,
            max_data_size=self.logger.test.max_data_logprobs,
            batch_size=self.logger.test.logprobs_batch_size,
            bs_num_samples=self.logger.test.logprobs_bootstrap_size,
        )
        mean_logprobs_std = logprobs_std.mean().item()
        mean_probs_std = probs_std.mean().item()
        rewards_x_tt = gfn.env.reward_batch(x_tt)
        corr_prob_traj_rewards = np.corrcoef(
            np.exp(logprobs_x_tt.cpu().numpy()), rewards_x_tt
        )[0, 1]
        var_logrewards_logp = torch.var(
            torch.log(tfloat(rewards_x_tt, float_type=gfn.float, device=gfn.device))
            - logprobs_x_tt
        ).item()
        nll_tt = -logprobs_x_tt.mean().item()
        logprobs_std_nll_ratio = torch.mean(-logprobs_std / logprobs_x_tt).item()

        x_sampled = []
        if gfn.buffer.test_type is not None and gfn.buffer.test_type == "all":
            batch, _ = gfn.sample_batch(n_forward=self.logger.test.n, train=False)
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
            batch, _ = gfn.sample_batch(n_forward=self.logger.test.n, train=False)
            assert batch.is_valid()
            x_sampled = batch.get_terminating_states()
            # TODO make it work with conditional env
            x_sampled = torch2np(gfn.env.states2proxy(x_sampled))
            x_tt = torch2np(gfn.env.states2proxy(x_tt))
            kde_pred = gfn.env.fit_kde(
                x_sampled,
                kernel=self.logger.test.kde.kernel,
                bandwidth=self.logger.test.kde.bandwidth,
            )
            if "log_density_true" in dict_tt and "kde_true" in dict_tt:
                log_density_true = dict_tt["log_density_true"]
                kde_true = dict_tt["kde_true"]
            else:
                # Sample from reward via rejection sampling
                x_from_reward = gfn.env.sample_from_reward(n_samples=self.logger.test.n)
                x_from_reward = torch2np(gfn.env.states2proxy(x_from_reward))
                # Fit KDE with samples from reward
                kde_true = gfn.env.fit_kde(
                    x_from_reward,
                    kernel=self.logger.test.kde.kernel,
                    bandwidth=self.logger.test.kde.bandwidth,
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
            # TODO: refactor
            env_metrics = gfn.env.test(x_sampled)
            return {
                "metrics": {
                    "l1": gfn.l1,
                    "kl": gfn.kl,
                    "jsd": gfn.jsd,
                    "corr_prob_traj_rewards": corr_prob_traj_rewards,
                    "var_logrewards_logp": var_logrewards_logp,
                    "nll_tt": nll_tt,
                    "mean_logprobs_std": mean_logprobs_std,
                    "mean_probs_std": mean_probs_std,
                    "logprobs_std_nll_ratio": logprobs_std_nll_ratio,
                },
                "figs": (None,),
                "env_metrics": env_metrics,
            }
        # L1 error
        l1 = np.abs(density_pred - density_true).mean()
        # KL divergence
        kl = (density_true * (log_density_true - log_density_pred)).mean()
        # Jensen-Shannon divergence
        log_mean_dens = np.logaddexp(log_density_true, log_density_pred) + np.log(0.5)
        jsd = 0.5 * np.sum(density_true * (log_density_true - log_mean_dens))
        jsd += 0.5 * np.sum(density_pred * (log_density_pred - log_mean_dens))

        # Plots
        # TODO-V: move to evaluator.plot()?
        if hasattr(gfn.env, "plot_reward_samples"):
            fig_reward_samples = gfn.env.plot_reward_samples(x_sampled, **plot_kwargs)
        else:
            fig_reward_samples = None
        if hasattr(gfn.env, "plot_kde"):
            fig_kde_pred = gfn.env.plot_kde(kde_pred, **plot_kwargs)
            fig_kde_true = gfn.env.plot_kde(kde_true, **plot_kwargs)
        else:
            fig_kde_pred = None
            fig_kde_true = None

        return {
            "metrics": {
                "l1": l1,
                "kl": kl,
                "jsd": jsd,
                "corr_prob_traj_rewards": corr_prob_traj_rewards,
                "var_logrewards_logp": var_logrewards_logp,
                "nll_tt": nll_tt,
                "mean_logprobs_std": mean_logprobs_std,
                "mean_probs_std": mean_probs_std,
                "logprobs_std_nll_ratio": logprobs_std_nll_ratio,
            },
            "figs": {
                "True reward and GFlowNet samples": fig_reward_samples,
                "GFlowNet KDE Policy": fig_kde_pred,
                "Reward KDE": fig_kde_true,
            },
            "env_metrics": {},
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
        print()
        if not gfn_states:
            # sample states from the current gfn
            batch = Batch(env=self.env, device=self.device, float_type=self.float)
            self.random_action_prob = 0
            t = time.time()
            print("Sampling from GFN...", end="\r")
            for b in batch_with_rest(
                0, self.logger.test.n_top_k, self.batch_size_total
            ):
                sub_batch, _ = self.sample_batch(n_forward=len(b), train=False)
                batch.merge(sub_batch)
            duration = time.time() - t
            gfn_states = batch.get_terminating_states()

        # compute metrics and get plots
        print("[eval_top_k] Making GFN plots...", end="\r")
        metrics, figs, fig_names = self.env.top_k_metrics_and_plots(
            gfn_states, self.logger.test.top_k, name="gflownet", step=it
        )
        if duration:
            metrics["gflownet top k sampling duration"] = duration

        if do_random:
            # sample random states from uniform actions
            if not random_states:
                batch = Batch(env=self.env, device=self.device, float_type=self.float)
                self.random_action_prob = 1.0
                print("[eval_top_k] Sampling at random...", end="\r")
                for b in batch_with_rest(
                    0, self.logger.test.n_top_k, self.batch_size_total
                ):
                    sub_batch, _ = self.sample_batch(n_forward=len(b), train=False)
                    batch.merge(sub_batch)
            # compute metrics and get plots
            random_states = batch.get_terminating_states()
            print("[eval_top_k] Making Random plots...", end="\r")
            (
                random_metrics,
                random_figs,
                random_fig_names,
            ) = self.env.top_k_metrics_and_plots(
                random_states, self.logger.test.top_k, name="random", step=None
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
            ) = self.env.top_k_metrics_and_plots(
                None, self.logger.test.top_k, name="train", step=None
            )
            # add to current metrics and plots
            summary.update(train_metrics)
            figs += train_figs
            fig_names += train_fig_names

        self.random_action_prob = prob

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

    def eval_and_log(self, it, metrics=_sentinel):
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
        # TODO-V: do we need to set attributes?
        result = self.eval(metrics=metrics)
        for m, v in result["metrics"].items():
            setattr(gfn, m, v)

        self.logger.log_test_metrics(*result["metrics"].values(), it, gfn.use_context)
        self.logger.log_metrics(result["env_metrics"], it, use_context=gfn.use_context)
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
