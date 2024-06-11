from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from omegaconf import OmegaConf

from gflownet.evaluator.abstract import METRICS, _sentinel
from gflownet.evaluator.base import BaseEvaluator

PERIOD_STEP_TARGET = [
    (0, 0, False),
    (0, 1, False),
    (0, 2, False),
    (-1, 0, False),
    (-1, 1, False),
    (-1, 2, False),
    (None, 0, False),
    (None, 1, False),
    (None, 2, False),
    (1, 0, True),
    (1, 1, True),
    (1, 2, True),
    (2, 0, True),
    (2, 1, False),
    (2, 2, True),
    (3, 0, True),
    (3, 1, False),
    (3, 2, False),
    (3, 3, True),
]

CONSTANT_EVALUATOR = BaseEvaluator(metrics="all")


@pytest.fixture
def dummy_evaluator(config_for_tests):
    return BaseEvaluator(**config_for_tests.evaluator)


@pytest.fixture
def constant_evaluator():  # faster fixture for state-less tests
    CONSTANT_EVALUATOR.config = OmegaConf.create({"metrics": "all"})
    return CONSTANT_EVALUATOR


@pytest.fixture
def all_reqs():
    return set([r for m in METRICS.values() for r in m["requirements"]])


def test__make_metrics__all(dummy_evaluator):
    assert dummy_evaluator.make_metrics("all") == METRICS


def test__make_metrics__str(dummy_evaluator):
    assert dummy_evaluator.make_metrics("l1,kl") == {
        k: METRICS[k] for k in ["l1", "kl"]
    }
    assert dummy_evaluator.make_metrics(" l1, kl") == {
        k: METRICS[k] for k in ["l1", "kl"]
    }
    with pytest.raises(ValueError, match="Unknown metric name.*"):
        dummy_evaluator.make_metrics("invalid")
    with pytest.raises(ValueError, match="Unknown metric name.*"):
        dummy_evaluator.make_metrics("l1,kj")

    with pytest.raises(ValueError, match=".*should not be an empty string.*"):
        dummy_evaluator.make_metrics("")


def test__make_metrics__list(dummy_evaluator):
    assert dummy_evaluator.make_metrics(["l1", "kl"]) == {
        k: METRICS[k] for k in ["l1", "kl"]
    }
    assert dummy_evaluator.make_metrics([" l1", " kl"]) == {
        k: METRICS[k] for k in ["l1", "kl"]
    }
    with pytest.raises(ValueError, match="Unknown metric name.*"):
        dummy_evaluator.make_metrics(["invalid"])


def test__make_metrics__None(dummy_evaluator):
    assert dummy_evaluator.make_metrics() is dummy_evaluator.metrics

    with pytest.raises(AssertionError):
        dummy_evaluator.metrics = _sentinel
        dummy_evaluator.make_metrics()


def test__make_metrics__dict(dummy_evaluator):
    assert dummy_evaluator.make_metrics({"l1": METRICS["l1"], "kl": METRICS["kl"]}) == {
        k: METRICS[k] for k in ["l1", "kl"]
    }
    with pytest.raises(ValueError, match="Unknown metric name.*"):
        dummy_evaluator.make_metrics(
            {
                "l1": METRICS["l1"],
                "invalid": {"name": "invalid", "requirements": ["anything"]},
            }
        )


def test__make_metrics__other(dummy_evaluator):
    with pytest.raises(ValueError, match="metrics should be None, a string.*"):
        dummy_evaluator.make_metrics(1)

    with pytest.raises(ValueError, match="metrics should be None, a string.*"):
        dummy_evaluator.make_metrics(1.0)

    with pytest.raises(ValueError, match="metrics should be None, a string.*"):
        dummy_evaluator.make_metrics({1, 2, 3})

    with pytest.raises(ValueError, match="metrics should be None, a string.*"):
        dummy_evaluator.make_metrics(_sentinel)


def test__make_requirements__all(dummy_evaluator, all_reqs):
    assert dummy_evaluator.make_requirements("all") == all_reqs


def test__make_requirements__str(dummy_evaluator, all_reqs):
    with pytest.raises(ValueError, match="reqs should be 'all'.*"):
        dummy_evaluator.make_requirements("")

    with pytest.raises(ValueError, match="reqs should be 'all'.*"):
        dummy_evaluator.make_requirements("any_str_but_all")


def test__make_requirements__list(dummy_evaluator, all_reqs):
    with pytest.raises(ValueError, match="Unknown requirement.*"):
        dummy_evaluator.make_requirements(["l1", "kl"])

    assert dummy_evaluator.make_requirements(list(all_reqs)) == all_reqs

    sub = list(all_reqs)[:2]
    assert dummy_evaluator.make_requirements(sub) == set(sub)

    dummy_evaluator.make_requirements(metrics=["l1", "corr_prob_traj_rewards"]) == set(
        r for r in all_reqs if r in ["l1", "corr_prob_traj_rewards"]
    )


def test__make_requirements__dict(dummy_evaluator, all_reqs):
    assert all(
        r in all_reqs
        for r in dummy_evaluator.make_requirements(
            metrics={k: METRICS[k] for k in ["l1", "corr_prob_traj_rewards"]}
        )
    )
    with pytest.raises(ValueError, match="Unknown metric name.*"):
        dummy_evaluator.make_requirements(
            metrics={
                "l1": METRICS["l1"],
                "invalid": {"name": "invalid", "requirements": ["anything"]},
            }
        )
    with pytest.raises(AssertionError, match="reqs should be None, 'all'.*"):
        dummy_evaluator.make_requirements(METRICS)


def test__make_requirements__None(dummy_evaluator, all_reqs):
    assert dummy_evaluator.make_requirements() == all_reqs

    with pytest.raises(ValueError, match="Cannot compute requirements from.*"):
        dummy_evaluator.reqs = _sentinel
        dummy_evaluator.metrics = _sentinel
        dummy_evaluator.make_requirements()


@pytest.mark.parametrize("period,step,target", PERIOD_STEP_TARGET)
def test__should_log_train(constant_evaluator, period, step, target):
    constant_evaluator.config.train_log_period = period
    assert constant_evaluator.should_log_train(step) is target


@pytest.mark.parametrize("period,step,target", PERIOD_STEP_TARGET)
def test__should_checkpoint(constant_evaluator, period, step, target):
    constant_evaluator.config.checkpoints_period = period
    assert constant_evaluator.should_checkpoint(step) is target


@pytest.mark.parametrize("first_it", [True, False])
@pytest.mark.parametrize("period,step,target", PERIOD_STEP_TARGET)
def test__should_eval(constant_evaluator, period, step, target, first_it):
    constant_evaluator.config.period = period
    constant_evaluator.config.first_it = first_it

    if step == 1 and first_it and period and period > 0:
        target = True

    assert constant_evaluator.should_eval(step) is target


@pytest.mark.parametrize("top_k", [None, -1, 0, 1, 2])
@pytest.mark.parametrize("first_it", [True, False])
@pytest.mark.parametrize("period,step,target", PERIOD_STEP_TARGET)
def test__should_eval_top_k(constant_evaluator, period, step, target, first_it, top_k):
    constant_evaluator.config.top_k_period = period
    constant_evaluator.config.top_k = top_k
    constant_evaluator.config.first_it = first_it

    if first_it and period and period > 0 and step == 1:
        target = True

    if not top_k or top_k <= 0:
        target = False

    assert constant_evaluator.should_eval_top_k(step) is target


@pytest.mark.parametrize(
    "config_for_tests,parameterization",
    [
        (None, "default"),
        (["env.length=4"], "grid_length_4"),
        (["env=ctorus"], "ctorus"),
    ],
    indirect=[
        # overrides arg for conftest.py::config_for_tests fixture
        "config_for_tests"
    ],
)
def test__eval(gflownet_for_tests, parameterization):
    assert gflownet_for_tests.buffer.replay_pkl.exists()
    # results: {"metrics": dict[str, float], "figs": list[plt.Figure]}
    results = gflownet_for_tests.evaluator.eval()
    figs = gflownet_for_tests.evaluator.plot(**results["data"])

    for k, v in results["metrics"].items():
        assert isinstance(k, str)
        assert isinstance(v, float)

    if parameterization == "default":
        pass
    elif parameterization == "grid_length_4":
        pass
    elif parameterization == "ctorus":
        for figname, fig in figs.items():
            assert isinstance(figname, str)
            # plot_samples_topk not implemented in ctorus
            if figname == "Samples TopK":
                continue
            assert isinstance(fig, plt.Figure)
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")
