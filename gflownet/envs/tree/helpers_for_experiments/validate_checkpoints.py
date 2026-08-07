"""
Quarantine unloadable checkpoints so that a resume can make progress.

``resume.py`` picks its checkpoint with
:func:`gflownet.utils.common.find_latest_checkpoint`, which returns ``final.ckpt``
if present and otherwise the highest ``iter_<step>.ckpt``. A job preempted in
the middle of ``torch.save`` leaves a truncated file that this selection still
prefers, and ``torch.load`` then raises -- on that resume and on *every*
subsequent one, because the selection is deterministic. The run is stuck until
the bad file is removed by hand.

This script applies the same selection rule, tries to load the file it picks,
and moves it to ``ckpts/corrupt/`` if the load fails, repeating until a
loadable checkpoint is found or none are left. Moving rather than deleting
keeps the evidence around; moving into a subdirectory rather than renaming in
place matters because ``find_latest_checkpoint`` globs ``iter_*`` and
``*final*`` in the checkpoint directory and would still pick up (and then choke
on the name of) a file merely suffixed with ``.corrupt``.

Usage:
    python validate_checkpoints.py <run_dir>/ckpts

Exit code is always 0: "no loadable checkpoint" is a normal outcome (it simply
means the caller should train from scratch), not an error.
"""

import sys
import warnings
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]  # <repo>/gflownet/envs/tree/helpers_for_experiments
sys.path.insert(0, str(REPO_ROOT))

import torch

from gflownet.utils.common import find_latest_checkpoint

# torch.load is called here exactly as resume.py calls it, so that a file this
# script accepts is one the resume can also read. That includes leaving
# `weights_only` at its default, whose deprecation warning is only noise here.
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <ckpts_dir>", file=sys.stderr)
        sys.exit(2)

    ckpts_dir = Path(sys.argv[1])
    if not ckpts_dir.is_dir():
        print(f"[ckpts] {ckpts_dir} does not exist; nothing to validate.")
        return

    quarantine = ckpts_dir / "corrupt"
    while True:
        try:
            path = find_latest_checkpoint(ckpts_dir)
        except ValueError:
            print(f"[ckpts] No loadable checkpoint left in {ckpts_dir}.")
            return
        try:
            torch.load(path, map_location="cpu")
        except Exception as e:
            quarantine.mkdir(exist_ok=True)
            path.rename(quarantine / path.name)
            print(f"[ckpts] {path.name} is unreadable ({e}); moved to corrupt/.")
            continue
        print(f"[ckpts] Latest loadable checkpoint: {path.name}")
        return


if __name__ == "__main__":
    main()
