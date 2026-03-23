"""
Write or update a Hydra experiment config for a given region/year/target_variable.

If the target config already exists, only the tuned fields are updated (comments
are preserved when ruamel.yaml is available, otherwise PyYAML is used with a
warning).

If the target config does not exist, it is generated from the consumption
template at TEMPLATE_PATH, substituting all variable-specific fields.

Fields managed by this module
------------------------------
buffer.test.path
env.amount_values_mapping
proxy.key_region
proxy.key_year
proxy.target_variable          (added if absent)
proxy.reward_function_kwargs.gamma / beta / alpha
logger.project_name
logger.run_name
logger.tags                    (target variable tag replaces "consumption")
hydra.run.dir
"""

import os
import re
import shutil

# ---------------------------------------------------------------------------
# YAML backend: prefer ruamel (comment-preserving), fall back to PyYAML
# ---------------------------------------------------------------------------
try:
    from ruamel.yaml import YAML
    _RUAMEL = True
except ImportError:
    import yaml as _pyyaml
    _RUAMEL = False
    print(
        "  [config_writer] ruamel.yaml not found — falling back to PyYAML. "
        "Comments in existing configs will be lost on update."
    )


TEMPLATE_PATH = "config/experiments/iam/plan_fairy_consumption.yaml"
CONFIG_DIR = "config/experiments/iam"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _var_to_slug(target_variable: str) -> str:
    """EMI_total_CO2 → emi_total_co2  (safe for filenames and tags)."""
    return target_variable.lower().replace("/", "_").replace(" ", "_")


def _config_path(target_variable: str) -> str:
    slug = _var_to_slug(target_variable)
    return os.path.join(CONFIG_DIR, f"plan_fairy_{slug}.yaml")


def _test_pkl_path(region: str, year: int, target_variable: str) -> str:
    slug = _var_to_slug(target_variable)
    return f"gflownet/proxy/iam/test_{region}_{year}_{slug}.pkl"


def _project_name(region: str, year: int, target_variable: str) -> str:
    slug = _var_to_slug(target_variable)
    return f"iameval-{region[:2]}{str(year)[2:]}-{slug}"


def _run_name() -> str:
    return "Plan Fairy TB"


def _hydra_dir(region: str, year: int, target_variable: str) -> str:
    slug = _var_to_slug(target_variable)
    return (
        "${user.logdir.root}/plan/"
        + slug
        + "/sigmoid/"
        + region
        + "/"
        + str(year)
        + "/${logger.run_name}"
    )


def _tags(target_variable: str) -> list:
    slug = _var_to_slug(target_variable)
    return ["gflownet", "plan", "fairy", "sigmoid", slug]


def _amount_list(amounts: dict) -> list:
    """Convert amounts dict → [0.0, HIGH, MEDIUM, LOW, NONE] list."""
    return [
        0.0,
        amounts["HIGH"],
        amounts["MEDIUM"],
        amounts["LOW"],
        amounts["NONE"],
    ]


# ---------------------------------------------------------------------------
# ruamel.yaml read/write (comment-preserving)
# ---------------------------------------------------------------------------

def _load_ruamel(path: str):
    ryaml = YAML()
    ryaml.preserve_quotes = True
    with open(path, "r") as f:
        return ryaml.load(f), ryaml


def _save_ruamel(doc, ryaml, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        ryaml.dump(doc, f)


# ---------------------------------------------------------------------------
# PyYAML read/write (fallback — loses comments)
# ---------------------------------------------------------------------------

def _load_pyyaml(path: str):
    with open(path, "r") as f:
        return _pyyaml.safe_load(f)


def _save_pyyaml(doc: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        # Emit the @package header manually — PyYAML strips top-level comments
        f.write("# @package _global_\n")
        _pyyaml.dump(doc, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Core: apply tuned values to a loaded config document
# ---------------------------------------------------------------------------

def _apply_fields(doc, region, year, target_variable, amounts, sigmoid_params):
    """Mutate `doc` in-place with all tuned fields."""
    slug = _var_to_slug(target_variable)

    # buffer.test.path
    doc["buffer"]["test"]["path"] = _test_pkl_path(region, year, target_variable)

    # env.amount_values_mapping
    doc["env"]["amount_values_mapping"] = _amount_list(amounts)

    # proxy fields
    doc["proxy"]["key_region"] = region
    doc["proxy"]["key_year"] = year
    doc["proxy"]["target_variable"] = target_variable
    doc["proxy"]["reward_function_kwargs"]["gamma"] = sigmoid_params["gamma"]
    doc["proxy"]["reward_function_kwargs"]["beta"] = sigmoid_params["beta"]
    doc["proxy"]["reward_function_kwargs"]["alpha"] = sigmoid_params["alpha"]

    # logger
    doc["logger"]["project_name"] = _project_name(region, year, target_variable)
    doc["logger"]["run_name"] = _run_name()
    doc["logger"]["tags"] = _tags(target_variable)

    # hydra.run.dir
    doc["hydra"]["run"]["dir"] = _hydra_dir(region, year, target_variable)

    return doc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_or_update_config(
    region: str,
    year: int,
    target_variable: str,
    amounts: dict,
    sigmoid_params: dict,
    template_path: str = TEMPLATE_PATH,
    dry_run: bool = False,
) -> str:
    """
    Create or update the Hydra experiment config for the given parameters.

    Parameters
    ----------
    region : str
    year : int
    target_variable : str
    amounts : dict   — {"HIGH": float, "MEDIUM": float, "LOW": float, "NONE": float}
    sigmoid_params : dict — {"gamma": float, "beta": float, "alpha": float}
    template_path : str  — path to the consumption template (used when target
                           config does not exist yet)
    dry_run : bool       — if True, print what would be written but don't save

    Returns
    -------
    str — path to the written config file
    """
    target_path = _config_path(target_variable)
    is_consumption = _var_to_slug(target_variable) == "consumption"

    # Decide source file
    if os.path.exists(target_path) and not is_consumption:
        source_path = target_path
        action = "update"
    elif os.path.exists(template_path):
        source_path = template_path
        action = "create" if not is_consumption else "update"
    else:
        raise FileNotFoundError(
            f"Template not found at '{template_path}'. "
            "Please ensure the consumption config exists at that path."
        )

    print(f"\n  Config {action}: {target_path}")
    print(f"  Source: {source_path}")

    # Load
    if _RUAMEL:
        doc, ryaml = _load_ruamel(source_path)
    else:
        doc = _load_pyyaml(source_path)

    # Apply tuned fields
    _apply_fields(doc, region, year, target_variable, amounts, sigmoid_params)

    if dry_run:
        print("  [dry_run] Would write:")
        if _RUAMEL:
            import sys
            ryaml.dump(doc, sys.stdout)
        else:
            print(_pyyaml.dump(doc, default_flow_style=False, sort_keys=False))
        return target_path

    # Save
    if _RUAMEL:
        _save_ruamel(doc, ryaml, target_path)
    else:
        _save_pyyaml(doc, target_path)

    print(f"  ✓ Config written to {target_path}")
    return target_path


# ---------------------------------------------------------------------------
# CLI (standalone use)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Write or update a Hydra experiment config"
    )
    parser.add_argument("--region", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--target_variable", default="CONSUMPTION")
    parser.add_argument("--amounts", required=True,
                        help="HIGH,MEDIUM,LOW,NONE as comma-separated floats")
    parser.add_argument("--sigmoid", required=True,
                        help="gamma,beta,alpha as comma-separated floats")
    parser.add_argument("--template", default=TEMPLATE_PATH)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    a_vals = [float(v) for v in args.amounts.split(",")]
    amounts = dict(zip(["HIGH", "MEDIUM", "LOW", "NONE"], a_vals))

    s_vals = [float(v) for v in args.sigmoid.split(",")]
    sigmoid_params = dict(zip(["gamma", "beta", "alpha"], s_vals))

    write_or_update_config(
        region=args.region,
        year=args.year,
        target_variable=args.target_variable,
        amounts=amounts,
        sigmoid_params=sigmoid_params,
        template_path=args.template,
        dry_run=args.dry_run,
    )
