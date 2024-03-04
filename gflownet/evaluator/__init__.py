"""
Create a new evaluator by subclassing this class and extending the :py:meth:`eval`
method to add more metrics and plots.

.. important::

    Only the :py:meth:`~gflownet.evaluator.abstract.GFlowNetAbstractEvaluator.from_dir`
    and :py:meth:`~gflownet.evaluator.abstract.GFlowNetAbstractEvaluator.from_agent`
    class methods should be used to instantiate this class.

Typical call stack:

1. :py:meth:`gflownet.gflownet.GFlowNetAgent.train` calls the evaluator's

2. :py:meth:`~gflownet.evaluator.abstract.GFlowNetAbstractEvaluator.should_eval`.
   If it returns ``True`` then :py:meth:`~gflownet.gflownet.GFlowNetAgent.train` calls

3. :py:meth:`~gflownet.evaluator.abstract.GFlowNetAbstractEvaluator.eval_and_log`
   which itself calls

4. :py:meth:`~gflownet.evaluator.base.GFlowNetEvaluator.eval` as
   ``results = self.eval(metrics=None)`` and then
   ``figs = self.plot(**results["data"])``

5. finally, :py:meth:`~gflownet.evaluator.base.GFlowNetEvaluator.eval_and_log` logs the
   results using the GFlowNetAgent's logger as
   ``self.logger.log_metrics(results["metrics"])`` and ``self.logger.log_plots(figs)``.

Basic concepts
--------------

The evaluator is used to compute metrics and plots. It is used to evaluate the
performance of the agent during training and to log the results. It is also
intended to be used to evaluate the performance of a trained agent.

The ``metrics`` keyword argument usually reflect to a description of which quantities
are to be computed. They can take the following forms:

- ``None``: all metrics defined in the config file / in the evaluator's
  ``.config.metrics`` attribute will be computed.

- ``"all"``: all known metrics as defined in :py:const:`METRICS` will be computed.

  - Note that classes that inherit from :py:class:`GFlowNetEvaluator` can define new
    metrics with the :py:meth:`define_new_metrics` method.

- ``list``: a list of metric names to be computed. The names must be keys of
  :py:const:`METRICS`.

- ``dict``: a dictionary that is a subset of :py:const:`METRICS`.

The concept of ``requirements`` is used to avoid unnecessary computations. If a metric
requires a certain quantity to be computed, then the evaluator will only compute that
quantity if the metric is requested. This is done by the :py:meth:`make_requirements`
method and can be used in methods that compute metrics and plots like
``if "some_req" in reqs`` (see below for an example).


Using an Evaluator
------------------

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

Implementing your own evaluator
-------------------------------

.. code-block:: python

    # gflownet/evaluator/my_evaluator.py
    from gflownet.evaluator.base import GFlowNetEvaluator, METRICS, ALL_REQS

    class MyEvaluator(GFlowNetEvaluator):
        def define_new_metrics(self):
            '''
            This method is called when the class is instantiated and is used to update
            the global METRICS and ALL_REQS variables.
            '''
            return {
                "your_metric": {
                    "display_name": "My custom metric",
                    "requirements": ["density", "new_req"],
                },
            }


        def my_custom_metric(self, some, arguments):
            '''
            Your metric-computing method. It should return a dict with two keys:
            "metrics" and "data".

            The "metrics" key should contain the new metric(s) and the "data" key
            should contain the intermediate results that can be used to plot the
            new metric(s).

            Its arguments will come from the `eval()` method below.

            Parameters
            ----------
            some : type
                description
            arguments : type
                description

            Returns
            -------
            dict
                A dict with two keys: "metrics" and "data".
            '''
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
            '''
            Your plotting method.

            It should return a dict with figure titles as keys and the figures as
            values.

            Its arguments will come from the `plot()` method below, and basically come
            from the "data" key of the output of other metrics-computing functions.

            Parameters
            ----------
            some_other : type, optional
                description, by default None
            arguments : type, optional
                description, by default None
            intermediate : type, optional
                description, by default None

            Returns
            -------
            dict
                A dict with figure titles as keys and the figures as values.
            '''
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
            '''
            Your custom plot method.

            It should return a dict with figure titles as keys and the figures as
            values.

            It will be called by the `eval_and_log` method to log the figures,
            and given the "data" key of the output of other metrics-computing functions.

            Returns
            -------
            dict
                A dict with figure titles as keys and the figures as values.
            '''
            figs = super().plot(**kwargs)
            figs.update(self.my_custom_plot(**kwargs))

            return figs

        def eval(self, metrics=None, **plot_kwargs):
            '''
            Your custom eval method.

            It should return a dict with two keys: "metrics" and "data".

            It will be called by the `eval_and_log` method to log the metrics,

            Parameters
            ----------
            metrics : Union[list, dict], optional
                The metrics you want to compute in this evaluation procedure,
                by default None, meaning the ones defined in the config file.

            Returns
            -------
            dict
                A dict with two keys: "metrics" and "data".
            '''
            metrics = self.make_metrics(metrics)
            reqs = self.make_requirements(metrics=metrics)

            results = super().eval(metrics=metrics, **plot_kwargs)

            if "new_req" in reqs:
                some = self.gfn.sample_something()
                arguments = utils.some_other_function()
                my_results = self.my_custom_metric(some, arguments)
                results["metrics"].update(my_results.get("metrics", {}))
                results["data"].update(my_results.get("data", {}))

            return results

Then define your own ``evaluator`` in the config file:

.. code-block:: yaml

    # config/evaluator/my_evaluator.yaml
    defaults:
      - base

    _target_: gflownet.evaluator.my_evaluator.MyEvaluator

    # any other params hereafter will extend or override the base class params:

    period: 1000


In the previous example, the ``define_new_metrics`` method is used to define new
metrics and associated requirements. It will be called when the
``MyEvaluator`` class is instantiated, in the init of ``GFlowNetAbstractEvaluator``.

By defining a new requirement, you ensure that the new metrics and plots will only be
computed if user asks for a metric that requires such computations.

By default, the train loop will call
:py:meth:`~gflownet.evaluator.base.GFlowNetAbstractEvaluator.eval_and_log` which itself
calls :py:meth:`~gflownet.evaluator.base.GFlowNetEvaluator.eval` so if you override
``eval()`` as above, the new metrics and plots will be computed and logged.

Similarly, :py:meth:`~gflownet.evaluator.base.GFlowNetEvaluator.eval_and_log`
will compute the ``dict`` of figures as ``fig_dict = self.plot(**results["data"])``
where ``results`` is the output of ``eval``.
"""
