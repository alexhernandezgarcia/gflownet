"""
Runnable script with hydra capabilities
"""

import cProfile  # Imported to monitor number and time of function calls
import os
import pickle
import pstats  # Imported to monitor number and time of function calls
import random
import signal
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import open_dict

from gflownet.utils.common import gflownet_from_config


@hydra.main(config_path="./config", config_name="train", version_base="1.1")
def main(config):

    # Set and print working and logging directory
    with open_dict(config):
        config.logger.logdir.path = (
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )
    print(f"\nWorking directory of this run: {os.getcwd()}")
    print(f"Logging directory of this run: {config.logger.logdir.path}\n")

    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)

    # Initialize a GFlowNet agent from the configuration file
    gflownet = gflownet_from_config(config)

    # Slurm sends SIGTERM at the time limit (SIGKILL follows ~30 s later, after
    # KillWait); Python's default SIGTERM action skips all finally blocks, so a
    # timed-out profiled run would lose its stats. Converting the signal to
    # SystemExit lets the dump below run within the KillWait grace period.
    if config.get("profile", False) or config.get("torch_profile", False):
        signal.signal(signal.SIGTERM, _sigterm_to_exit)

    # Train GFlowNet with monitoring number and time of function calls
    if config.get("torch_profile", False):
        train_with_torch_profiler(gflownet, config)
    elif config.get("profile", False):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            gflownet.train()
        finally:
            # Report stats even if training crashes or is interrupted
            profiler.disable()

            sort_key = config.get("profile_sort", "cumtime")
            n_rows = config.get("profile_n_rows", 20)
            stats = pstats.Stats(profiler).sort_stats(sort_key)
            print(f"\n=== cProfile: top {n_rows} by {sort_key} ===")
            profile_filter = config.get("profile_filter", None)
            if profile_filter:
                stats.print_stats(profile_filter, n_rows)
            else:
                stats.print_stats(n_rows)

            profile_path = Path(config.logger.logdir.path) / "train.prof"
            stats.dump_stats(profile_path)
            print(f"[profile] Raw stats saved to: {profile_path}")
            print(f"[profile] Inspect with:  snakeviz {profile_path}")
    else:
        gflownet.train()

    # Sample from trained GFlowNet
    # TODO: move to method in GFlowNet agent, like sample_and_log()
    if config.n_samples > 0 and config.n_samples <= 1e5:
        batch, times = gflownet.sample_batch(n_forward=config.n_samples, train=False)
        x_sampled = batch.get_terminating_states(proxy=True)
        energies = gflownet.proxy(x_sampled)
        x_sampled = batch.get_terminating_states()
        df = pd.DataFrame(
            {
                "readable": [gflownet.env.state2readable(x) for x in x_sampled],
                "energies": energies.tolist(),
            }
        )
        samples_dir = Path("./samples/")
        samples_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(samples_dir / "gfn_samples.csv")
        dct = {"x": x_sampled, "energy": energies}
        pickle.dump(dct, open(samples_dir / "gfn_samples.pkl", "wb"))

    # Print replay buffer
    if len(gflownet.buffer.replay) > 0:
        print("\nReplay buffer:")
        print(gflownet.buffer.replay)

    # Close logger
    # TODO: make it gflownet.end() - perhaps there are other things to end
    gflownet.logger.end()


def _sigterm_to_exit(signum, frame):
    raise SystemExit(143)


def train_with_torch_profiler(gflownet, config):
    """
    Trains while profiling a window of iterations with torch.profiler, which
    sees inside the C++ ops (notably autograd's run_backward) that cProfile
    reports as a single opaque call.

    The profiled window is iterations [torch_profile_wait + 2,
    torch_profile_wait + 2 + torch_profile_active): the schedule waits
    torch_profile_wait iterations (so the batch can grow toward its plateau),
    warms up for 2, then records torch_profile_active. One training iteration
    performs exactly one optimizer step, so the global optimizer post-step hook
    advances the schedule without any change to the training loop.

    Prints the aggregated op table and writes torch_trace.json (open at
    https://ui.perfetto.dev or chrome://tracing) to the run directory once the
    window completes; a run cut short before that produces no output.
    """
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    from torch.profiler import ProfilerActivity, profile, schedule

    wait = config.get("torch_profile_wait", 10)
    warmup = 2
    active = config.get("torch_profile_active", 5)
    n_rows = config.get("profile_n_rows", 20)
    # Per-op input shapes multiply the profiler's own memory footprint; with
    # the millions of small ops per iteration in this codebase, recording them
    # can OOM the job. Enable only with generous --mem and a 1-2 step window.
    record_shapes = config.get("torch_profile_shapes", False)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    sort_by = (
        "self_cuda_time_total"
        if ProfilerActivity.CUDA in activities
        else "self_cpu_time_total"
    )
    logdir = Path(config.logger.logdir.path)

    def on_trace_ready(prof):
        first, last = wait + warmup, wait + warmup + active - 1
        print(
            f"\n=== torch.profiler: iterations {first}-{last}, "
            f"top {n_rows} by {sort_by} ==="
        )
        print(prof.key_averages().table(sort_by=sort_by, row_limit=n_rows))
        trace_path = logdir / "torch_trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"[torch_profile] Chrome trace saved to: {trace_path}")

    prof = profile(
        activities=activities,
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=on_trace_ready,
        record_shapes=record_shapes,
        with_stack=False,
    )
    hook = register_optimizer_step_post_hook(lambda *args, **kwargs: prof.step())
    try:
        with prof:
            gflownet.train()
    finally:
        hook.remove()


def set_seeds(seed):
    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
    sys.exit()
