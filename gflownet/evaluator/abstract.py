"""
Abstract evaluator class for GFlowNetAgent.

.. warning::

    Should not be used directly, but subclassed to implement specific evaluators for
    different tasks and environments.

See :class:`~gflownet.evaluator.base.BaseEvaluator` for a default,
concrete implementation of this abstract class.

This class handles some logic that will be the same for all evaluators.
The only requirements for a subclass are to implement the
:meth:`~gflownet.evaluator.abstract.AbstractEvaluator.eval` and
:meth:`~gflownet.evaluator.abstract.AbstractEvaluator.plot` methods
which will be called by the
:meth:`~gflownet.evaluator.abstract.AbstractEvaluator.eval_and_log` method:

.. code-include :: :meth:`gflownet.evaluator.abstract.AbstractEvaluator.eval_and_log`

.. code-include :: :func:`gflownet.evaluator.abstract.AbstractEvaluator.eval_and_log`

.. code-include :: :class:`gflownet.gflownet.abstract.AbstractEvaluator`

.. code-include :: :func:`gflownet.utils.common.gflownet_from_config`

.. code-block:: python

        def eval_and_log(self, it, metrics=None):
            results = self.eval(metrics=metrics)
            for m, v in results["metrics"].items():
                setattr(self.gfn, m, v)

            metrics_to_log = {
                METRICS[k]["display_name"]: v for k, v in results["metrics"].items()
            }

            figs = self.plot(**results["data"])

            self.logger.log_metrics(metrics_to_log, it, self.gfn.use_context)
            self.logger.log_plots(figs, it, use_context=self.gfn.use_context)

See :mod:`gflownet.evaluator` for a full-fledged example and
:mod:`gflownet.evaluator.base` for a concrete implementation of this abstract class.
"""

import os
from abc import ABCMeta, abstractmethod
from typing import Union

from omegaconf import OmegaConf

from gflownet.utils.common import load_gflow_net_from_run_path

# purposefully non-documented object, hidden from Sphinx docs
_sentinel = object()
"""
A sentinel object to be used as a default value for arguments that could be None.
"""

METRICS = {}
"""
All metrics that can be computed by a ``BaseEvaluator``.

Structured as a dict with the metric names as keys and the metric display
names and requirements as values.

Requirements are used to decide which kind of data / samples is required to compute the
metric.

Display names are used to log the metrics and to display them in the console.

Implementations of :class:`AbstractEvaluator` can add new metrics to
this dict by implementing the method
:meth:`AbstractEvaluator.define_new_metrics`.
"""

ALL_REQS = set([r for m in METRICS.values() for r in m["requirements"]])
"""
Union of all requirements of all metrics in :const:`METRICS`.
"""


class AbstractEvaluator(metaclass=ABCMeta):
    def __init__(self, gfn_agent=None, **config):
        """
        Abstract evaluator class for :class:`GFlowNetAgent`.

        In charge of evaluating the :class:`GFlowNetAgent`, computing metrics
        plotting figures and optionally logging results using the
        :class:`GFlowNetAgent`'s :class:`Logger`.

        You can use the :meth:`from_dir` or :meth:`from_agent` class methods
        to easily instantiate this class from a run directory or an existing
        in-memory :class:`GFlowNetAgent`.

        Use
        :meth:`~gflownet.evaluator.abstract.AbstractEvaluator.set_agent`
        to set the evaluator's :class:`GFlowNetAgent` after initialization if it was
        not provided at instantiation as ``GflowNetEvaluator(gfn_agent=...)``.

        This ``__init__`` function will call, in order:

        1. :meth:`update_all_metrics_and_requirements` which uses new metrics defined in
           the :meth:`define_new_metrics` method to update the global :const:`METRICS`
           and :const:`ALL_REQS` variables in classes inheriting from
           :class:`AbstractEvaluator`.

        2. ``self.metrics = self.make_metrics(self.config.metrics)`` using
           :meth:`make_metrics`

        3. ``self.reqs = self.make_requirements()`` using :meth:`make_requirements`

        Arguments
        ---------
        gfn_agent : GFlowNetAgent, optional
            The GFlowNetAgent to evaluate. By default None. Should be set using the
            :meth:`from_dir` or :meth:`from_agent` class methods.

        config : dict
            The configuration of the evaluator. Will be converted to an OmegaConf
            instance and stored in the ``self.config`` attribute.

        Attributes
        ----------
        config : :class:`omegaconf.OmegaConf`
            The configuration of the evaluator.
        metrics : dict
            Dictionary of metrics to compute, with the metric names as keys and the
            metric display names and requirements as values.
        reqs : set[str]
            The set of requirements for the metrics. Used to decide which kind of data /
            samples is required to compute the metric.
        logger : Logger
            The logger to use to log the results of the evaluation. Will be set to the
            GFlowNetAgent's logger.
        """

        self._gfn_agent = gfn_agent
        self.config = OmegaConf.create(config)

        if self._gfn_agent is not None:
            self.logger = self._gfn_agent.logger

        self.metrics = self.reqs = _sentinel

        self.update_all_metrics_and_requirements()
        self.metrics = self.make_metrics(self.config.metrics)
        self.reqs = self.make_requirements()

    @property
    def gfn(self):
        """
        Get the ``GFlowNetAgent`` to evaluate.

        This is a read-only property. Use the :meth:`set_agent` method to set
        the ``GFlowNetAgent``.

        Returns
        -------
        :class:`GFlowNetAgent`
            The ``GFlowNetAgent`` to evaluate.

        Raises
        ------
        ValueError
            If the ``GFlowNetAgent`` has not been set.
        """
        if type(self._gfn_agent).__name__ != "GFlowNetAgent":
            raise ValueError(
                "The GFlowNetAgent has not been set. Use the `from_dir` or `from_agent`"
                + " class methods to instantiate this class or the `set_agent` method"
            )
        return self._gfn_agent

    def set_agent(self, gfn_agent):
        """
        Set the ``GFlowNetAgent`` to evaluate after initialization.

        It is then accessible through the ``self.gfn`` property.

        Parameters
        ----------
        gfn_agent : :class:`GFlowNetAgent`
            The ``GFlowNetAgent`` to evaluate.
        """
        assert type(gfn_agent).__name__ == "GFlowNetAgent", (
            "gfn_agent should be an instance of GFlowNetAgent, but is an instance of "
            + f"{type(gfn_agent)}."
        )
        self._gfn_agent = gfn_agent
        self.logger = gfn_agent.logger

    @gfn.setter
    def gfn(self, _):
        raise AttributeError(
            "The `gfn` attribute is read-only. Use the `set_agent` method to set the"
            + " GFlowNetAgent."
        )

    def define_new_metrics(self):
        """
        Method to be implemented by subclasses to define new metrics.

        Example
        -------
        .. code-block:: python

            def define_new_metrics(self):
                return {
                    "my_custom_metric": {
                        "display_name": "My custom metric",
                        "requirements": ["density", "new_req"],
                    }
                }

        Returns
        -------
        dict
            Dictionary of new metrics to add to the global :const:`METRICS` dict.
        """
        pass

    def update_all_metrics_and_requirements(self):
        """
        Method to be implemented by subclasses to update the global dict of metrics and
        requirements.
        """
        new_metrics = self.define_new_metrics()
        if new_metrics:
            global METRICS
            global ALL_REQS
            METRICS.update(new_metrics)
            ALL_REQS = set([r for m in METRICS.values() for r in m["requirements"]])

    @classmethod
    def from_dir(
        cls: "AbstractEvaluator",
        path: Union[str, os.PathLike],
        no_wandb: bool = True,
        print_config: bool = False,
        device: str = "cuda",
        load_final_ckpt: bool = True,
    ):
        """
        Instantiate a BaseEvaluator from a run directory.

        Parameters
        ----------
        cls : BaseEvaluator
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
        BaseEvaluator
            Instance of BaseEvaluator with the GFlowNetAgent loaded from the run.
        """
        gfn_agent, _ = load_gflow_net_from_run_path(
            path,
            no_wandb=no_wandb,
            print_config=print_config,
            device=device,
            load_final_ckpt=load_final_ckpt,
        )
        return cls.from_agent(gfn_agent)

    @classmethod
    def from_agent(cls, gfn_agent):
        """
        Instantiate a BaseEvaluator from a GFlowNetAgent.

        Parameters
        ----------
        cls : BaseEvaluator
            Evaluator class to instantiate.
        gfn_agent : GFlowNetAgent
            Instance of GFlowNetAgent to use for the BaseEvaluator.

        Returns
        -------
        BaseEvaluator
            Instance of BaseEvaluator with the provided GFlowNetAgent.
        """
        from gflownet.gflownet import GFlowNetAgent

        assert isinstance(gfn_agent, GFlowNetAgent), (
            "gfn_agent should be an instance of GFlowNetAgent, but is an instance of "
            + f"{type(gfn_agent)}."
        )

        return cls(gfn_agent=gfn_agent, **gfn_agent.evaluator.config)

    def make_metrics(self, metrics=None):
        """
        Parse metrics from a dict, list, a string or ``None``.

        - If ``None``, all metrics are selected.
        - If a string, it can be a comma-separated list of metric names, with or without
          spaces.
        - If a list, it should be a list of metric names (keys of :const:`METRICS`).
        - If a dict, its keys should be metric names and its values will be ignored:
          they will be assigned from :const:`METRICS`.

        All metrics must be in :const:`METRICS`.

        Parameters
        ----------
        metrics : Union[str, List[str]], optional
            Metrics to compute when running the
            :meth:`.eval` method. Defaults to ``None``, i.e. all metrics in
            :const:`METRICS` are computed.

        Returns
        -------
        dict
            Dictionary of metrics to compute, with the metric names as keys and the
            metric display names and requirements as values.

        Raises
        ------
            ValueError
                If a metric name is not in :const:`METRICS`.
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

        1. If ``metrics`` is provided, they must be as a dict of metrics.
           The requirements are computed from the ``requirements`` attribute of
           the metrics.

        2. Otherwise, the requirements are computed from the ``reqs`` argument:
            - If ``reqs`` is ``"all"``, all requirements of all metrics are computed.
            - If ``reqs`` is ``None``, the evaluator's ``self.reqs`` attribute is used.
            - If ``reqs`` is a list, it is used as the requirements.

        Parameters
        ----------
        reqs : Union[str, List[str]], optional
            The metrics requirements. Either ``"all"``, a list of requirements or
            ``None`` to use the evaluator's ``self.reqs`` attribute.
            By default ``None``.
        metrics : Union[str, List[str], dict], optional
            The metrics to compute requirements for. If not a dict, will be passed to
            :meth:`make_metrics`. By default None.

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
        on the ``self.config.train.period`` attribute.

        Set ``self.config.train.period`` to ``None`` or a negative value to disable
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
        the ``self.config.test.period`` attribute.

        Set ``self.config.test.first_it`` to ``True`` if testing should be done at the
        first iteration step. Otherwise, testing will be done aftter
        ``self.config.test.period`` steps.

        Set ``self.config.test.period`` to ``None`` or a negative value to disable
        testing.

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
        decision is based on the ``self.config.test.top_k`` and
        ``self.config.test.top_k_period`` attributes.

        Set ``self.config.test.top_k`` to ``None`` or a negative value to disable top k
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
        on the ``self.checkpoints.period`` attribute.

        Set ``self.checkpoints.period`` to ``None`` or a negative value to disable
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
        """
        The main method to plot results.

        Will be called by the :meth:`eval_and_log` method to plot the results
        of the evaluation.
        Will be passed the results of the :meth:`eval` method:

        .. code-block:: python

            # in eval_and_log
            results = self.eval(metrics=metrics)
            figs = self.plot(**results["data"])

        Returns
        -------
        dict
            Dictionary of figures to log, with the figure names as keys and the figures
            as values.
        """
        pass

    @abstractmethod
    def eval(self, metrics=None, **plot_kwargs):
        """
        The main method to compute metrics and intermediate results.

        This method should return a dict with two keys: ``"metrics"`` and ``"data"``.

        The "metrics" key should contain the new metric(s) and the "data" key should
        contain the intermediate results that can be used to plot the new metric(s).

        Example
        -------
        >>> metrics = None # use the default metrics from the config file
        >>> results = gfne.eval(metrics=metrics)
        >>> plots = gfne.plot(**results["data"])

        >>> metrics = "all" # compute all metrics, regardless of the config
        >>> results = gfne.eval(metrics=metrics)

        >>> metrics = ["l1", "kl"] # compute only the L1 and KL metrics
        >>> results = gfne.eval(metrics=metrics)

        >>> metrics = "l1,kl" # alternative syntax
        >>> results = gfne.eval(metrics=metrics)

        See :ref:`evaluator basic concepts` for more details about ``metrics``.

        Parameters
        ----------
        metrics : Union[str, dict, list], optional
            Which metrics to compute, by default ``None``.
        """
        pass

    @abstractmethod
    def eval_top_k(self, it):
        """
        Evaluate the ``GFlowNetAgent``'s top k samples performance.

        Classes extending this abstract class should implement this method.

        Parameters
        ----------
        it : int
            Current iteration step.

        Returns
        -------
        dict
            Dictionary with the following keys schema:
            .. code-block:: python

                {
                    "metrics": {str: float},
                    "figs": {str: plt.Figure},
                    "summary": {str: float},
                }
        """
        pass

    def eval_and_log(self, it, metrics=None):
        """
        Evaluate the GFlowNetAgent and log the results with its logger.

        Will call ``self.eval()`` and log the results using the GFlowNetAgent's logger
        ``log_metrics()`` and ``log_plots()`` methods.

        Parameters
        ----------
        it : int
            Current iteration step.
        metrics : Union[str, List[str]], optional
            List of metrics to compute, by default the evaluator's ``metrics``
            attribute.
        """
        results = self.eval(metrics=metrics)
        for m, v in results["metrics"].items():
            setattr(self.gfn, m, v)

        metrics_to_log = {
            METRICS[k]["display_name"]: v for k, v in results["metrics"].items()
        }

        figs = self.plot(**results["data"])

        self.logger.log_metrics(metrics_to_log, it, self.gfn.use_context)
        self.logger.log_plots(figs, it, use_context=self.gfn.use_context)

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
