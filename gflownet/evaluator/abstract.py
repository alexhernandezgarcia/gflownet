"""
Abstract evaluator class for GFlowNetAgent.

Should not be used directly, but subclassed to implement specific evaluators for
different tasks and environments.

See :py:class:`~gflownet.evaluator.base.GFlowNetEvaluator` for a default,
concrete implementation of this abstract class.

This class handles some logic that will be the same for all evaluators.
The only requirements for a subclass are to implement the `plot` and `eval` methods
which will be called by the
:py:meth:`~gflownet.evaluator.abstract.GFlowNetAbstractEvaluator.eval_and_log` method.

.. code-block:: python

        def eval_and_log(self, it, metrics=None):
            gfn = self.gfn_agent
            results = self.eval(metrics=metrics)
            for m, v in results["metrics"].items():
                setattr(gfn, m, v)

            mertics_to_log = {
                METRICS[k]["display_name"]: v for k, v in results["metrics"].items()
            }

            figs = self.plot(**results["data"])

            self.logger.log_metrics(mertics_to_log, it, gfn.use_context)
            self.logger.log_plots(figs, it, use_context=gfn.use_context)
"""

import copy
import os
import time
from abc import ABCMeta, abstractmethod
from typing import Union

import torch

from gflownet.utils.batch import Batch
from gflownet.utils.common import batch_with_rest, load_gflow_net_from_run_path

_sentinel = object()

METRICS = {
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
"""
All metrics that can be computed by a GFlowNetEvaluator. Structured as a dict with the
metric names as keys and the metric display names and requirements as values.

Requirements are used to decide which kind of data / samples is required to compute the
metric.

Display names are used to log the metrics and to display them in the console.
"""

ALL_REQS = set([r for m in METRICS.values() for r in m["requirements"]])
"""
Union of all requirements of all metrics in `METRICS`. Computed from
:py:const:`METRICS`.
"""


class GFlowNetAbstractEvaluator(metaclass=ABCMeta):
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

        self.update_all_metrics_and_requirements()

        self.metrics = self.make_metrics(self.config.metrics)
        self.reqs = self.make_requirements()

    def update_all_metrics_and_requirements(self):
        """
        Method to be implemented by subclasses to update the global dict of metrics and
        requirements.
        """
        pass

    @classmethod
    def from_dir(
        cls: "GFlowNetAbstractEvaluator",
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
        return GFlowNetAbstractEvaluator.from_agent(gfn_agent)

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

        return GFlowNetAbstractEvaluator(gfn_agent=gfn_agent, sentinel=_sentinel)

    def make_metrics(self, metrics=None):
        """
        Parse metrics from a dict, list, a string or None.

        - If `None`, all metrics are selected.
        - If a string, it can be a comma-separated list of metric names, with or without
          spaces.
        - If a list, it should be a list of metric names (keys of `METRICS`).
        - If a dict, its keys should be metric names and its values will be ignored:
          they will be assigned from `METRICS`.

        All metrics must be in `METRICS`.

        Parameters
        ----------
        metrics : Union[str, List[str]], optional
            Metrics to compute when running the `evaluator.eval()` function. Defaults to
            None, i.e. all metrics in `METRICS` are computed.

        Returns
        -------
        dict
            Dictionary of metrics to compute, with the metric names as keys and the
            metric display names and requirements as values.

        Raises
        ------
            ValueError
                If a metric name is not in `METRICS`.
        """
        if metrics is None:
            assert self.metrics is not _sentinel, (
                "Error setting self.metrics. This is likely due to the `metrics:`"
                + " entry missing from your eval config. Set it to 'all' to compute all"
                + " metrics or to a comma-separated list of metric names (eg 'l1, kl')."
            )
            return self.metrics

        if not isinstance(metrics, (str, list, dict)):
            raise ValueError(
                "metrics should be None, a string, a list or a dict,"
                + f" but is {type(metrics)}."
            )

        if metrics == "all":
            metrics = METRICS.keys()

        if isinstance(metrics, str):
            if metrics == "":
                raise ValueError(
                    "`metrics` should not be an empty string. "
                    + "Set to 'all' or a list of metric names or None (null in YAML)."
                )
            if "," in metrics:
                metrics = metrics.split(",")
            else:
                metrics = [metrics]

        if isinstance(metrics, dict):
            metrics = metrics.keys()

        metrics = [m.strip() for m in metrics]

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
            - If `reqs` is `None`, the evaluator's `self.reqs` attribute is used.
            - If `reqs` is a list, it is used as the requirements.

        Parameters
        ----------
        reqs : Union[str, List[str]], optional
            The metrics requirements. Either `"all"`, a list of requirements or `None`
            to use the evaluator's `self.reqs` attribute. By default None
        metrics : Union[str, List[str], dict], optional
            The metrics to compute requirements for. If not a dict, will be passed to
            `make_metrics`. By default None.

        Returns
        -------
        set[str]
            The set of requirements for the metrics.
        """

        if metrics is not None:
            if not isinstance(metrics, dict):
                metrics = self.make_metrics(metrics)
            for m in metrics:
                if m not in METRICS:
                    raise ValueError(f"Unknown metric name: {m}")
            return set([r for m in metrics.values() for r in m["requirements"]])

        if isinstance(reqs, str):
            if reqs == "all":
                reqs = ALL_REQS.copy()
            else:
                raise ValueError(
                    "reqs should be 'all', a list of requirements or None, but is "
                    + f"{reqs}."
                )
        if reqs is None:
            if self.reqs is _sentinel:
                if not isinstance(self.metrics, dict):
                    raise ValueError(
                        "Cannot compute requirements from `None` without the `metrics`"
                        + " argument or the `self.metrics` attribute set to a dict"
                        + " of metrics."
                    )
                self.reqs = set(
                    [r for m in self.metrics.values() for r in m["requirements"]]
                )
            reqs = self.reqs
        if isinstance(reqs, list):
            reqs = set(reqs)

        assert isinstance(
            reqs, set
        ), f"reqs should be None, 'all', a set or a list, but is {type(reqs)}"

        assert all([isinstance(r, str) for r in reqs]), (
            "All elements of reqs should be strings, but are "
            + f"{[type(r) for r in reqs]}"
        )

        for r in reqs:
            if r not in ALL_REQS:
                raise ValueError(f"Unknown requirement: {r}")

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
            True if train logging should be done at the current step, False otherwise.
        """
        if self.config.train_log_period is None or self.config.train_log_period <= 0:
            return False
        else:
            return step % self.config.train_log_period == 0

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
        if self.config.period is None or self.config.period <= 0:
            return False
        elif step == 1 and self.config.first_it:
            return True
        else:
            return step % self.config.period == 0

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
        if self.config.top_k is None or self.config.top_k <= 0:
            return False

        if self.config.top_k_period is None or self.config.top_k_period <= 0:
            return False

        if step == 1 and self.config.first_it:
            return True

        return step % self.config.top_k_period == 0

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
        if (
            self.config.checkpoints_period is None
            or self.config.checkpoints_period <= 0
        ):
            return False
        else:
            return not step % self.config.checkpoints_period

    @abstractmethod
    def plot(self, **kwargs):
        pass

    @abstractmethod
    def eval(self, metrics=None, **plot_kwargs):
        pass

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
        results = self.eval(metrics=metrics)
        for m, v in results["metrics"].items():
            setattr(gfn, m, v)

        mertics_to_log = {
            METRICS[k]["display_name"]: v for k, v in results["metrics"].items()
        }

        figs = self.plot(**results["data"])

        self.logger.log_metrics(mertics_to_log, it, gfn.use_context)
        self.logger.log_plots(figs, it, use_context=gfn.use_context)

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

        return {
            "metrics": metrics,
            "figs": figs,
            "summary": summary,
        }

    def eval_and_log_top_k(self, it):
        """
        Evaluate the GFlowNetAgent's top k samples performance and log the results with
        its logger.

        Parameters
        ----------
        it : int
            Current iteration step, by default None.
        """

        results = self.eval_top_k(it)
        self.logger.log_plots(results["figs"], it, use_context=self.use_context)
        self.logger.log_metrics(
            results["metrics"], use_context=self.use_context, step=it
        )
        self.logger.log_summary(results["summary"])
