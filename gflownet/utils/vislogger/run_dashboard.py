"""
Run this script to start the dashboard manually.
python gflownet/utils/vislogger/run_dashboard.py

Arguments:
    --path
        path to the directory with the logged data (eg. logs/local/YY-MM-DD_hh-.../)
        expects the subfolder .hydra with config.yaml
        and the subfolder visdata with data.db
        By default the latest run in the logs/local folder is used.
    --default_s0
        set flag if your env does not have env.source specified, the the default
        '#' of the dashboard is used
    --debug-mode
        set flag to use debug mode
"""

import argparse
import os
from pathlib import Path

from dashboard import run_dashboard
from hydra.utils import instantiate
from omegaconf import OmegaConf


def get_latest_local_run(base_dir: Path) -> Path:
    """
    Returns the latest directory in base_dir based on lexicographic order.
    Expected format: YYYY-MM-DD_hh-mm-ss_microseconds
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"{base_dir} does not exist")
    candidates = [d for d in base_dir.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {base_dir}")
    # Lexicographic sort works for this timestamp format
    return max(candidates, key=lambda d: d.name)


def find_project_root(start: Path, marker: str = "gflownet") -> Path:
    """
    Walk upwards from start until a directory named `marker` is found.
    """
    matches = [p for p in start.parents if p.name == marker]
    if not matches:
        raise RuntimeError("Could not find project root, specify path")
    return matches[-1]


def main():
    parser = argparse.ArgumentParser(description="Run dashboard")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to logging directory (defaults to latest logs/local run)",
    )

    parser.add_argument(
        "--default-s0",
        action="store_true",
        help="Use if env does not have a source state ",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = find_project_root(script_path)
    if args.path is None:
        logs_local = project_root / "logs" / "local"
        run_dir = get_latest_local_run(logs_local)
        path = run_dir
        print(f"[INFO] Using latest run: {path}")
    else:
        path = Path(os.path.expanduser(args.path))
    path = path.resolve()

    cfg = OmegaConf.load(path / ".hydra" / "config.yaml")
    env = instantiate(cfg.env)
    s0 = "#" if args.default_s0 else env.state2readable(env.source)

    run_dashboard(
        data=str(path / "visdata"),
        text_to_img_fn=env.text_to_img_fn,
        state_aggregation_fn=env.state_aggregation_fn,
        s0=s0,
        debug_mode=args.debug,
    )


if __name__ == "__main__":
    main()
