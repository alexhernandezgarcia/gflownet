"""
Derive a run's identity (naming fields + config hash) from a Hydra config.

Composes exactly the same configuration that ``train.py`` would build from the
given overrides -- without instantiating or training anything -- and prints a
shell-eval-able block of ``CFG_*`` assignments::

    eval "$(python config_hash.py +experiments=tree/classification_tree \\
                env.data_path=/abs/path/wine_1.csv seed=0)"

    $CFG_HASH        8-hex-char hash of the fully resolved config
    $CFG_DATASET     wine        (parsed from env.data_path)
    $CFG_SPLIT       1           (parsed from env.data_path)
    $CFG_MAX_DEPTH   5
    $CFG_N_STEPS     1000
    $CFG_LR          0.01
    $CFG_POLICY      mlp         (short label derived from policy._target_)
    $CFG_SEED        0
    $CFG_ALPHA       0.1         (proxy.alpha_value, needed by eval_tree.py)
    $CFG_N_SAMPLES   1000

The point of $CFG_HASH is that a run directory named ``<human_name>_<hash>``
changes whenever *any* configuration value changes -- including hyperparameters
that the human-readable part of the name does not mention. Two launches with
the same name and the same config resolve to the same directory (so the second
can be skipped or resumed); two launches that differ in any config value land
in different directories and can never overwrite each other.

The remaining CFG_* fields exist so that the launcher never has to hard-code a
value that the config might contradict: the name is built from what the config
actually resolves to.

What the hash does NOT cover
----------------------------
Only the resolved config is hashed. The code version, the contents of the
dataset CSV and the installed package versions are not part of it, so an
identical hash across two commits does not imply identical results. The
launcher records the git commit in the run's LAUNCH file and warns when it
differs from the one that started the run.

``logger.run_name`` is excluded from the hash by construction: the run name
contains the hash, so including it would be circular.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]  # <repo>/gflownet/envs/tree/helpers_for_experiments
sys.path.insert(0, str(REPO_ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# Config keys removed before hashing. `logger.run_name` would be circular (the
# name embeds the hash). The rest are per-job noise that must not change the
# identity of a run: they differ between the first launch and every resume.
EXCLUDED_KEYS = [
    "logger.run_name",
    "logger.run_id",
    "logger.logdir.path",
    "logger.logdir.root",
    "logger.is_resumed",
]

# Short, readable labels for the policy classes used by the tree experiments.
POLICY_LABELS = {
    "MLPPolicy": "mlp",
    "TreeTransformerPolicy": "trfm",
    "MultiheadTreePolicy": "multihead",
    "SimpleTreePolicy": "simpletree",
}


def drop_key(container: dict, dotted_key: str) -> None:
    """Delete a dotted key from a nested dict, ignoring missing intermediates."""
    parts = dotted_key.split(".")
    node = container
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]
    if isinstance(node, dict):
        node.pop(parts[-1], None)


def config_hash(container: dict, length: int) -> str:
    """SHA-256 over a canonical JSON serialization of the resolved config."""
    canonical = json.dumps(container, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def policy_label(container: dict) -> str:
    """
    Short label for the forward policy.

    The generic ``Policy`` class covers several architectures and names the one
    in use in ``policy.forward.type`` (e.g. "mlp"); the tree-specific policies
    are their own classes and carry no ``type`` key, so they are labelled from
    the ``_target_`` class name instead.
    """
    policy = container.get("policy", {})
    forward_type = (policy.get("forward") or {}).get("type", None)
    if forward_type:
        return str(forward_type)
    class_name = str(policy.get("_target_", "")).rsplit(".", 1)[-1]
    return POLICY_LABELS.get(class_name, class_name.lower() or "policy")


def dataset_and_split(container: dict):
    """
    Parse ``env.data_path`` as ``.../<dataset>/<dataset>_<split>.csv``.

    Mirrors the convention of aggregate_treeclass_results.py. Falls back to the
    bare file stem with an empty split when the name has no ``_<split>`` suffix.
    """
    data_path = container.get("env", {}).get("data_path", None)
    if not data_path:
        return "nodata", ""
    stem = Path(str(data_path)).stem
    dataset, _, split = stem.rpartition("_")
    if dataset and split.isdigit():
        return dataset, split
    return stem, ""


def shell_quote(value) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-name",
        default="train",
        help="Hydra config name, without extension (default: train).",
    )
    parser.add_argument(
        "--config-dir",
        default=str(REPO_ROOT / "config"),
        help="Directory holding the Hydra configs (default: <repo>/config).",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=8,
        help="Number of hex characters of the hash to keep (default: 8).",
    )
    parser.add_argument(
        "--hash-only",
        action="store_true",
        help="Print just the hash instead of the CFG_* shell block.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved, hashed config to stderr (for debugging).",
    )
    args, overrides = parser.parse_known_args()

    with initialize_config_dir(
        version_base="1.1", config_dir=args.config_dir, job_name="config_hash"
    ):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    container = OmegaConf.to_container(cfg, resolve=True)
    for key in EXCLUDED_KEYS:
        drop_key(container, key)

    if args.print_config:
        print(OmegaConf.to_yaml(OmegaConf.create(container)), file=sys.stderr)

    digest = config_hash(container, args.length)
    if args.hash_only:
        print(digest)
        return

    dataset, split = dataset_and_split(container)
    optimizer = container.get("gflownet", {}).get("optimizer", {})
    fields = {
        "CFG_HASH": digest,
        "CFG_DATASET": dataset,
        "CFG_SPLIT": split,
        "CFG_MAX_DEPTH": container.get("env", {}).get("max_depth", ""),
        "CFG_N_STEPS": optimizer.get("n_train_steps", ""),
        "CFG_LR": optimizer.get("lr", ""),
        "CFG_POLICY": policy_label(container),
        "CFG_SEED": container.get("seed", ""),
        "CFG_ALPHA": container.get("proxy", {}).get("alpha_value", ""),
        "CFG_N_SAMPLES": container.get("n_samples", ""),
    }
    for key, value in fields.items():
        print(f"{key}={shell_quote(value)}")


if __name__ == "__main__":
    main()
