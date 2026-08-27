"""
Unit tests for aggregate_treeclass_results.py (pure helpers + the collectors).

Run from the repo root (venv active):

    pytest gflownet/envs/tree/helpers_for_experiments/test_aggregate_treeclass_results.py -v

No wandb access or real runs needed: everything runs on synthetic records and
temporary directories.
"""

import json

import pytest
from omegaconf import OmegaConf

from gflownet.envs.tree.helpers_for_experiments import (
    aggregate_treeclass_results as agg,
)

TREE_TARGET = "gflownet.envs.tree.tree.Tree"


# ---------------------------------------------------------------------------
# Helpers to build synthetic configs / records
# ---------------------------------------------------------------------------


def make_config(seed=0, lr=0.001, data_path="/data/iris/iris_1.csv", **extra):
    """A minimal but realistic resolved tree config."""
    cfg = {
        "env": {"_target_": TREE_TARGET, "data_path": data_path, "max_depth": 5},
        "gflownet": {
            "optimizer": {"n_train_steps": 1000, "lr": lr},
            "random_action_prob": 0.1,
        },
        "policy": {"_target_": "gflownet.policy.mlp.MLPPolicy", "backward": None},
        "seed": seed,
        "logger": {"run_name": "run", "logdir": {"path": "/logs/CAMP/run"}},
        "user": {"name": "someone"},
    }
    cfg.update(extra)
    return cfg


def make_record(
    ghash="aaaaaaaa",
    split="1",
    launched=100.0,
    recency=0.0,
    settings=None,
    metrics=None,
    campaign="CAMP",
    task="classification",
    dataset="iris",
    debug=False,
    run_name="run",
    **extra,
):
    rec = {
        "task": task,
        "dataset": dataset,
        "split": split,
        "hash": ghash,
        # Real records always carry the full settings dict produced by
        # settings_from_config (with "?" defaults for missing keys).
        "settings": (
            settings
            if settings is not None
            else agg.settings_from_config(make_config())
        ),
        "config": {},
        "run_name": run_name,
        "campaign": campaign,
        "debug": debug,
        "metrics": metrics if metrics is not None else {},
        "launched": launched,
        "recency": recency,
    }
    rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# normalize_numbers
# ---------------------------------------------------------------------------


def test_normalize_numbers_integer_floats():
    assert agg.normalize_numbers(1.0) == 1
    assert isinstance(agg.normalize_numbers(1.0), int)
    assert agg.normalize_numbers(1.5) == 1.5


def test_normalize_numbers_recurses_into_dicts_and_lists():
    obj = {"a": 2.0, "b": [3.0, 3.5, {"c": 4.0}], "d": "1.0"}
    out = agg.normalize_numbers(obj)
    assert out == {"a": 2, "b": [3, 3.5, {"c": 4}], "d": "1.0"}
    assert isinstance(out["b"][2]["c"], int)


def test_normalize_numbers_leaves_bools_and_none_alone():
    assert agg.normalize_numbers(True) is True
    assert agg.normalize_numbers(None) is None


# ---------------------------------------------------------------------------
# fmt_mean_std
# ---------------------------------------------------------------------------


def test_fmt_mean_std_empty_is_dash():
    assert agg.fmt_mean_std([], 3) == "-"
    assert agg.fmt_mean_std([float("nan")], 1) == "-"


def test_fmt_mean_std_single_value_zero_std():
    assert agg.fmt_mean_std([0.5], 1) == "0.5000 ± 0.0000"


def test_fmt_mean_std_marks_missing_values():
    # 2 finite values but 3 expected -> [n=2] suffix.
    assert agg.fmt_mean_std([1.0, 2.0], 3).endswith("[n=2]")
    assert "[n=" not in agg.fmt_mean_std([1.0, 2.0], 2)


def test_fmt_mean_std_precision_switches_at_100():
    assert agg.fmt_mean_std([250.0], 1).startswith("250.0 ")
    assert agg.fmt_mean_std([2.5], 1).startswith("2.5000 ")


def test_fmt_mean_std_uses_sample_std():
    # ddof=1: std of [1, 3] is sqrt(2), not 1.
    assert agg.fmt_mean_std([1.0, 3.0], 2) == "2.0000 ± 1.4142"


# ---------------------------------------------------------------------------
# dedupe
# ---------------------------------------------------------------------------


def test_dedupe_eval_keeps_newest_mtime(capsys):
    old = make_record(recency=100.0, run_name="old")
    new = make_record(recency=200.0, run_name="new")
    kept = agg.dedupe([old, new], "eval")
    assert [r["run_name"] for r in kept] == ["new"]
    assert "duplicate eval runs" in capsys.readouterr().out


def test_dedupe_wandb_keeps_furthest_step_then_latest_launch():
    slow = make_record(recency=(500, 100.0), run_name="far")
    fast = make_record(recency=(100, 999.0), run_name="near-but-newer")
    kept = agg.dedupe([slow, fast], "wandb")
    assert [r["run_name"] for r in kept] == ["far"]
    # Equal step: latest launch wins.
    a = make_record(recency=(500, 100.0), run_name="a")
    b = make_record(recency=(500, 200.0), run_name="b")
    assert [r["run_name"] for r in agg.dedupe([a, b], "wandb")] == ["b"]


def test_dedupe_keeps_different_splits_and_hashes():
    recs = [
        make_record(split="1"),
        make_record(split="2"),
        make_record(split="1", ghash="bbbbbbbb"),
    ]
    assert len(agg.dedupe(recs, "eval")) == 3


def test_dedupe_rejects_mixed_recency_types():
    recs = [make_record(recency=100.0), make_record(recency=(1, 2.0), split="2")]
    with pytest.raises(AssertionError):
        agg.dedupe(recs, "eval")


# ---------------------------------------------------------------------------
# group_identity: disk vs wandb configs must hash identically
# ---------------------------------------------------------------------------


def test_group_identity_ignores_run_identity_keys():
    a = make_config(data_path="/data/iris/iris_1.csv")
    b = make_config(data_path="/data/iris/iris_2.csv")
    b["logger"] = {"run_name": "other", "logdir": {"path": "/elsewhere"}}
    b["user"] = {"name": "someone-else"}
    b["_wandb"] = {"cli_version": "0.17"}
    ia, ib = agg.group_identity(a), agg.group_identity(b)
    assert ia["hash"] == ib["hash"]
    assert (ia["split"], ib["split"]) == ("1", "2")
    assert ia["dataset"] == ib["dataset"] == "iris"


def test_group_identity_float_int_roundtrip_same_hash():
    # Disk yaml yields 1000.0 / 0.0, wandb round-trips them as 1000 / 0.
    disk = make_config()
    disk["gflownet"]["optimizer"]["n_train_steps"] = 1000.0
    wandb_style = make_config()
    wandb_style["gflownet"]["optimizer"]["n_train_steps"] = 1000
    assert agg.group_identity(disk)["hash"] == agg.group_identity(wandb_style)["hash"]


def test_group_identity_different_hyperparams_different_hash():
    assert (
        agg.group_identity(make_config(lr=0.001))["hash"]
        != agg.group_identity(make_config(lr=0.01))["hash"]
    )


def test_group_identity_non_tree_env_is_none():
    cfg = make_config()
    cfg["env"]["_target_"] = "gflownet.envs.grid.Grid"
    assert agg.group_identity(cfg) is None


def test_group_identity_extra_excluded_keys():
    a, b = make_config(), make_config()
    b["torch_profile"] = True
    assert agg.group_identity(a)["hash"] != agg.group_identity(b)["hash"]
    assert (
        agg.group_identity(a, ("torch_profile",))["hash"]
        == agg.group_identity(b, ("torch_profile",))["hash"]
    )


# ---------------------------------------------------------------------------
# campaign_from_config / parse_wandb_created_at / fmt_launch / is_debug
# ---------------------------------------------------------------------------


def test_campaign_from_config_parent_of_logdir_path():
    cfg = {"logger": {"logdir": {"path": "/scratch/gflownet-logs/MAGIC_STAB/run_ab12"}}}
    assert agg.campaign_from_config(cfg, "fallback") == "MAGIC_STAB"


def test_campaign_from_config_fallback():
    assert agg.campaign_from_config({}, "proj") == "proj"
    assert agg.campaign_from_config({"logger": {"logdir": {"path": None}}}, "p") == "p"


def test_parse_wandb_created_at():
    assert agg.parse_wandb_created_at("2026-08-27T17:49:00Z") > 0
    assert agg.parse_wandb_created_at(None) == 0.0
    assert agg.parse_wandb_created_at("not-a-date") == 0.0
    # Z suffix and explicit offset are the same instant.
    assert agg.parse_wandb_created_at("2026-08-27T17:49:00Z") == (
        agg.parse_wandb_created_at("2026-08-27T17:49:00+00:00")
    )


def test_fmt_launch_unknown():
    assert agg.fmt_launch(0.0) == "?"
    assert agg.fmt_launch(1735689600.0).startswith("20")


def test_is_debug():
    assert agg.is_debug("DEBUG_iris_run")
    assert agg.is_debug("iris", "SMOKE_TEST/dir")
    assert not agg.is_debug("MAGIC_STAB_magic1")


# ---------------------------------------------------------------------------
# build_table
# ---------------------------------------------------------------------------


def _three_split_group(ghash="aaaaaaaa", launched=100.0, acc=0.9):
    return [
        make_record(
            ghash=ghash,
            split=str(i),
            launched=launched + i,
            metrics={"test_acc_top1": acc},
        )
        for i in (1, 2, 3)
    ]


def test_build_table_hides_groups_below_min_splits():
    recs = _three_split_group("aaaaaaaa") + [make_record(ghash="bbbbbbbb", split="1")]
    df, n_hidden = agg.build_table(recs, ["test_acc_top1"], "eval", min_splits=3)
    assert list(df["config"]) == ["aaaaaaaa"]
    assert n_hidden == 1
    df_all, n_hidden_all = agg.build_table(
        recs, ["test_acc_top1"], "eval", min_splits=1
    )
    assert len(df_all) == 2 and n_hidden_all == 0


def test_build_table_column_order_and_campaign():
    df, _ = agg.build_table(_three_split_group(), ["test_acc_top1"], "eval", 3)
    assert list(df.columns[:3]) == ["config", "campaign", "launched"]
    assert df.iloc[0]["campaign"] == "CAMP"
    # wandb tables get the same leading columns plus last_step/state.
    wrecs = [
        make_record(split=str(i), recency=(10, 1.0), step=10, state="finished")
        for i in (1, 2, 3)
    ]
    dfw, _ = agg.build_table(wrecs, [], "wandb", 3)
    assert list(dfw.columns[:3]) == ["config", "campaign", "launched"]
    assert {"last_step", "state"} <= set(dfw.columns)


def test_build_table_sorted_by_launch_newest_first():
    recs = _three_split_group("aaaaaaaa", launched=100.0) + _three_split_group(
        "bbbbbbbb", launched=500.0
    )
    df, _ = agg.build_table(recs, ["test_acc_top1"], "eval", 3)
    assert list(df["config"]) == ["bbbbbbbb", "aaaaaaaa"]


def test_build_table_renames_mean_n_nodes():
    recs = [make_record(split=str(i), metrics={"mean_n_nodes": 7.0}) for i in (1, 2, 3)]
    df, _ = agg.build_table(recs, ["mean_n_nodes"], "eval", 3)
    assert "mean_n_decisionnodes" in df.columns
    assert "mean_n_nodes" not in df.columns


def test_build_table_asserts_on_settings_mismatch_within_group():
    recs = _three_split_group()
    recs[1]["settings"] = {"lr": 999}
    with pytest.raises(AssertionError, match="different settings"):
        agg.build_table(recs, ["test_acc_top1"], "eval", 3)


def test_build_table_aggregates_metrics_over_splits():
    recs = [
        make_record(split="1", metrics={"test_acc_top1": 0.8}),
        make_record(split="2", metrics={"test_acc_top1": 1.0}),
        make_record(split="3", metrics={}),  # missing metric -> [n=2]
    ]
    df, _ = agg.build_table(recs, ["test_acc_top1"], "eval", 3)
    assert df.iloc[0]["test_acc_top1"].startswith("0.9000 ±")
    assert df.iloc[0]["test_acc_top1"].endswith("[n=2]")
    assert df.iloc[0]["n"] == 3
    assert df.iloc[0]["splits"] == "1,2,3"


# ---------------------------------------------------------------------------
# collect_eval_runs on a synthetic runs tree
# ---------------------------------------------------------------------------


def _write_run(root, campaign, name, cfg, metrics):
    run_dir = root / campaign / name
    (run_dir / ".hydra").mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(cfg), run_dir / ".hydra" / "config.yaml")
    (run_dir / "eval_results.json").write_text(json.dumps(metrics))
    return run_dir


def test_collect_eval_runs_end_to_end(tmp_path):
    for split in (1, 2):
        _write_run(
            tmp_path,
            "CAMPAIGN_A",
            f"run_split{split}",
            make_config(data_path=f"/data/iris/iris_{split}.csv"),
            {"test_acc_top1": 0.9},
        )
    records = agg.collect_eval_runs(tmp_path)
    assert len(records) == 2
    rec = records[0]
    assert rec["task"] == "classification"
    assert rec["dataset"] == "iris"
    assert rec["campaign"] == "CAMPAIGN_A"
    assert rec["metrics"] == {"test_acc_top1": 0.9}
    assert rec["launched"] > 0
    # Both splits share the training configuration -> one group hash.
    assert len({r["hash"] for r in records}) == 1


def test_collect_eval_runs_skips_resume_and_missing_config(tmp_path, capsys):
    # A resume dir and a run without .hydra/config.yaml are both skipped.
    _write_run(tmp_path, "CAMP", "resume", make_config(), {"a": 1.0})
    bare = tmp_path / "CAMP" / "bare_run"
    bare.mkdir(parents=True)
    (bare / "eval_results.json").write_text("{}")
    records = agg.collect_eval_runs(tmp_path)
    assert records == []
    assert "no .hydra/config.yaml" in capsys.readouterr().out


def test_collect_eval_runs_warns_on_unresolvable_config(tmp_path, capsys):
    cfg = make_config()
    cfg["proxy"] = {"beta": "${oc.env:DOES_NOT_EXIST_ANYWHERE_12345}"}
    _write_run(tmp_path, "CAMP", "run1", cfg, {"test_acc_top1": 0.5})
    records = agg.collect_eval_runs(tmp_path)
    out = capsys.readouterr().out
    assert "could not resolve config" in out and "UNRESOLVED" in out
    assert len(records) == 1  # still reported, just with a loud warning


def test_collect_eval_runs_skips_corrupt_json(tmp_path, capsys):
    run_dir = _write_run(tmp_path, "CAMP", "run1", make_config(), {"a": 1.0})
    (run_dir / "eval_results.json").write_text("{not json")
    assert agg.collect_eval_runs(tmp_path) == []
    assert "unreadable eval_results.json" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Metric configuration sanity
# ---------------------------------------------------------------------------


def test_wandb_metrics_no_training_diagnostics():
    for task, metrics in agg.WANDB_METRICS.items():
        assert "logZ" not in metrics, task
        assert "Loss" not in metrics, task
        assert "Train batch - logrewards mean" not in metrics, task


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
