"""
Evaluation script for 2D Ising GFN experiments.

This script supports two modes:

1. evaluation:
   - Sample a trained GFlowNet
   - Compute empirical observables
   - Compute theoretical observables

2. theory:
   - Compute theoretical observables only

Only nearest-neighbor interactions --J_nn 1, no magnetic field --h 0,
and periodic boundary conditions are currently supported for theoretical values.

Examples
--------
Evaluate trained model:
    python evaluate_ising.py evaluation --rundir logs/run1 --n_samples 5000

Theoretical only:
    python evaluate_ising.py theory --length 6 --beta 0.44
"""

import argparse
import itertools
import json
import os
import pickle
import subprocess
import warnings

import numpy as np
import yaml

from gflownet.utils.common import load_gflownet_from_rundir

# -----------------------------------------------------------
# Argument parser
# -----------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    eval_parser = subparsers.add_parser(
        "run_gfn_evaluation", help="Evaluate a trained GFlowNet run."
    )
    eval_parser.add_argument(
        "--rundir", type=str, required=True, help="Path to run (where .hydra/ is)."
    )
    eval_parser.add_argument(
        "--wandbdir",
        type=str,
        required=True,
        help="Path to WandB run (where wandb-summary.json is).",
    )
    eval_parser.add_argument(
        "--n_samples", type=int, default=100, help="Number of samples."
    )
    eval_parser.add_argument(
        "--sampling_batch_size", type=int, default=1, help="Batch size."
    )
    eval_parser.add_argument("--eval_script", type=str, default="../../eval.py")
    eval_parser.add_argument(
        "--estimate_logz",
        action="store_true",
        help="Estimate logZ from model samples using evaluator.logz_est.",
    )
    eval_parser.add_argument(
        "--n_logz_samples",
        type=int,
        default=100,
        help="Number of forward model samples used in logZ estimation.",
    )
    eval_parser.add_argument(
        "--logz_batch_size",
        type=int,
        default=100,
        help="Batch size for drawing model samples in logZ estimation.",
    )
    eval_parser.add_argument(
        "--n_trajs_logz",
        type=int,
        default=10,
        help="Number of backward trajectories per sampled terminal state for logZ estimation.",
    )
    eval_parser.add_argument(
        "--resample",
        action="store_true",
        help="Force re-sampling even if samples already exist",
    )
    theory_parser = subparsers.add_parser(
        "run_theory", help="Compute theoretical Ising quantities."
    )
    theory_parser.add_argument("--periodic", type=bool, default=True)
    theory_parser.add_argument(
        "--length", type=int, required=True, help="Lattice size."
    )
    theory_parser.add_argument(
        "--beta",
        type=float,
        required=True,
        help="Inverse temperature in the physics convention (no minus sign).",
    )
    theory_parser.add_argument(
        "--J_nn", type=float, default=1.0, help="Nearest neighbor interaction."
    )
    theory_parser.add_argument("--h", type=float, default=0.0, help="Magnetic field.")

    return parser.parse_args()


# -----------------------------------------------------------
# Read Ising parameters from Hydra config
# -----------------------------------------------------------


def load_ising_params(rundir):
    config_path = os.path.join(rundir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg.get("env", {})
    proxy_cfg = cfg.get("proxy", {})

    n_dim = env_cfg.get("n_dim")
    length = env_cfg.get("length")

    J_nn = proxy_cfg.get("J_nn")
    # beta_raw is negative
    beta_raw = proxy_cfg.get("reward_function_kwargs", {}).get("beta")

    return n_dim, length, J_nn, beta_raw


# -----------------------------------------------------------
# Read logZ from WandB
# -----------------------------------------------------------


def load_wandb_logZ(wandbdir):
    def _extract_run_id(path):
        run_dir = os.path.basename(os.path.dirname(path.rstrip("/")))
        # Expected format: run-YYYYMMDD_HHMMSS-<run_id>
        if run_dir.startswith("run-") and "-" in run_dir:
            return run_dir.split("-")[-1]
        return None

    def _load_entity_project(path):
        metadata_path = os.path.join(path, "wandb-metadata.json")
        if not os.path.exists(metadata_path):
            return None, None
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata.get("entity"), metadata.get("project")
        except Exception:
            return None, None

    def _load_from_wandb_api(path):
        run_id = _extract_run_id(path)
        if run_id is None:
            return None

        entity, project = _load_entity_project(path)
        if not entity:
            entity = os.environ.get("WANDB_ENTITY")
        if not project:
            project = os.environ.get("WANDB_PROJECT")
        if not entity or not project:
            return None

        try:
            import wandb

            api = wandb.Api(timeout=30)
            run = api.run(f"{entity}/{project}/{run_id}")

            logz = run.summary.get("logZ")
            if logz is not None:
                return float(logz)

            # Fallback to history in case summary is not finalized on timed-out runs.
            last = None
            for row in run.scan_history(keys=["logZ"]):
                val = row.get("logZ")
                if val is not None:
                    last = val
            if last is not None:
                return float(last)
        except Exception:
            return None
        return None

    wandb_summary_path = os.path.join(wandbdir, "wandb-summary.json")
    if os.path.exists(wandb_summary_path):
        try:
            with open(wandb_summary_path, "r") as f:
                data = json.load(f)
            if "logZ" in data:
                return float(data["logZ"])
        except Exception:
            pass

    logz_api = _load_from_wandb_api(wandbdir)
    if logz_api is not None:
        return logz_api

    warnings.warn(
        f"Could not retrieve logZ from local summary or WandB API for {wandbdir}. "
        "Using NaN for logZ from GFN."
    )
    return float("nan")


# -----------------------------------------------------------
# Call eval script if needed
# -----------------------------------------------------------


def maybe_sample(args):
    sample_path = os.path.join(args.rundir, "eval", "samples", "gfn_samples.pkl")

    if os.path.exists(sample_path) and not args.resample:
        print(f"Found existing sample file:\n  {sample_path}\nSkipping sampling.\n")
        return

    eval_script = os.path.expanduser(args.eval_script)
    rundir_abolute_path = os.path.abspath(args.rundir)
    print("path", rundir_abolute_path)
    cmd = (
        f"python {eval_script} "
        f"rundir={os.path.abspath(args.rundir)} "
        f"n_samples={args.n_samples} "
        f"sampling_batch_size={args.sampling_batch_size}"
    )

    print("Running:", cmd)
    subprocess.run(cmd, shell=True, check=True)
    print("\nSampling finished.\n")


def estimate_logz_from_eval(args, sampled_states=None):
    """Estimate logZ from evaluator-driven model samples.

    Loads a trained GFlowNet from a run directory, enables evaluator-side logZ
    estimation, and returns the scalar estimate produced by
    :py:meth:`gflownet.evaluator.base.BaseEvaluator.eval`.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. The following attributes are used:
        ``rundir``, ``n_logz_samples``, ``logz_batch_size``, and
        ``n_trajs_logz``.
    sampled_states : array-like, optional
        Optional terminating states reused for logZ estimation. If ``None``,
        the evaluator samples terminal states from the model policy.

    Returns
    -------
    float
        Estimated log-partition value produced by evaluator metric ``logz_est``.

    Raises
    ------
    RuntimeError
        If evaluator output does not contain metric ``logz_est``.
    """
    print("Running logZ estimation through evaluator API")
    gflownet, _ = load_gflownet_from_rundir(
        rundir=os.path.abspath(args.rundir),
        device="cpu",
        no_wandb=True,
        print_config=False,
        load_last_checkpoint=True,
    )

    evaluator = gflownet.evaluator
    evaluator.config.compute_logz_est = True
    evaluator.config.n_logz_samples = args.n_logz_samples
    evaluator.config.logz_batch_size = args.logz_batch_size
    evaluator.config.n_trajs_logz = args.n_trajs_logz
    evaluator.metrics = evaluator.make_metrics("logz_est")
    evaluator.reqs = evaluator.make_requirements(metrics=evaluator.metrics)

    result = evaluator.eval(metrics="logz_est", logz_states=sampled_states)
    logz_est = result.get("metrics", {}).get("logz_est")
    if logz_est is None:
        raise RuntimeError(
            "Could not compute `logz_est` from evaluator. "
            "Check that the run has a valid checkpoint and environment config."
        )
    return float(logz_est)


# -----------------------------------------------------------
# Load GFN-generated samples from pickle
# -----------------------------------------------------------


def load_samples_from_pkl(rundir):
    """
    Loads GFN-generated samples and energies from a pickle file.

    Returns:
    -------
        configs
        energies
    """

    pkl_path = os.path.join(rundir, "eval", "samples", "gfn_samples.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found at {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    samples = np.array(data["x"])
    energies = np.array(data["energy"])

    return samples, energies


# -----------------------------------------------------------
# Empirical Physical quantities
# -----------------------------------------------------------


def mean_abs_magnetization(samples):
    return np.mean(abs_magnetization(samples))


def mean_susceptibility(samples, beta):
    r"""
    Mean over samples and over sites of susceptibility computed using the absolute magnetization.

    $$
    \chi = \beta N \left( \langle |m|^2 \rangle - \langle |m| \rangle^2 \right)
    $$

    where:
    - $|m|$ is the mean over sites absolute magentization.
    - $\beta$ is the inverse temperature,
    - $N$ is the number of spins per configuration,
    - $\langle \cdot \rangle$ denotes the empirical average over samples.

    Notes
    -----
        This definition uses the absolute magnetization $|m|$.
    """
    N = samples[0].size
    m_abs = abs_magnetization(samples)
    return beta * N * (np.mean(m_abs**2) - np.mean(m_abs) ** 2)


def abs_magnetization(samples):
    r"""
    Mean over sites of absolute magentization for each sample.
    $\abs{m} = |\frac{1}{N} \sum_{i=1}^{N} s_i|$
    """
    # total number of spins per configuration
    N = samples[0].size
    flat = samples.reshape(len(samples), N)
    mags = np.abs(flat.mean(axis=1))
    return mags


def mean_energy(energies):
    return np.mean(energies)


def heat_capacity(samples, energies, beta):
    r"""
    Mean over samples and over sites of hea capacity.

    $$
    C = \frac{\beta^2}{N} \left( \langle E^2 \rangle - \langleE \rangle^2 \right)
    $$
    """
    N = samples[0].size

    return beta**2 * (np.mean(energies**2) - np.mean(energies) ** 2) / N


# -----------------------------------------------------------
# Theoretical logZ. The formulas used here are taken from the code base of
# the followwing paper: https://arxiv.org/abs/2503.08918
# -----------------------------------------------------------


def theoretical_logZ(beta, L):
    """
    Compute finite-size theoretical log Z for periodic 2D Ising with no magnetic field.

    Parameters
    ----------
    beta : float
    L : int

    Returns
    -------
    float
        log Z
    """

    def Z1(beta, L):
        z = 1
        for r in range(0, L):
            z *= 2.0 * np.cosh(0.5 * L * gamma_r(beta, L, 2.0 * r + 1.0))
        return z

    def Z2(beta, L):
        z = 1
        for r in range(0, L):
            z *= 2.0 * np.sinh(0.5 * L * gamma_r(beta, L, 2.0 * r + 1.0))
        return z

    def Z3(beta, L):
        z = 1
        for r in range(0, L):
            z *= 2.0 * np.cosh(0.5 * L * gamma_r(beta, L, 2.0 * r))
        return z

    def Z4(beta, L):
        z = 1
        for r in range(0, L):
            z *= 2.0 * np.sinh(0.5 * L * gamma_r(beta, L, 2.0 * r))
        return z

    def gamma_r(beta, L, r):
        if r == 0:
            return 2.0 * beta + np.log(np.tanh(beta))
        else:
            return np.log(c_R(beta, r, L) + np.sqrt(c_R(beta, r, L) ** 2 - 1.0))

    def c_R(beta, r, L):
        return np.cosh(2.0 * beta) * coth(2.0 * beta) - np.cos(r * np.pi / L)

    def coth(x):
        return np.cosh(x) / np.sinh(x)

    logz0 = np.log((2 * np.sinh(2 * beta))) * (L**2 / 2) - np.log(2)
    z_tmp = Z1(beta, L) + Z2(beta, L) + Z3(beta, L) + Z4(beta, L)
    logz = logz0 + np.log(z_tmp)

    return logz

def compute_theoretical_quantities(beta, length):
    """
    Compute all available theoretical quantities. For now, only includes logZ.
    Analytical expressions in the thermodynamic limit of <|m|> and C can be added in the future.
    """
    return {
        "logZ": theoretical_logZ(beta, length),
    }
# -----------------------------------------------------------
# Run modes
# -----------------------------------------------------------


def run_theory(args):
    """
    Run theoretical computations only. For now, only includes logZ.
    Analytical expressions in the thermodynamic limit of <|m|> and C can be added in the future.
    """
    if args.J_nn != 1.0:
        raise ValueError("Analytical logZ currently only supports J_nn = 1.")

    if args.h != 0.0:
        raise ValueError(
            "Analytical logZ currently only supports zero external field (h = 0)."
        )
    if not args.periodic:
        raise ValueError(
            "Analytical logZ currently only supports periodic boundary conditions."
        )
    theory = compute_theoretical_quantities(args.beta, args.length)
    print("\nTheoretical quantities:")
    print(f"log_Z (theoretical) = {theory['logZ']:.6f}")


def run_gfn_evaluation(args):
    """Run trained GFN evaluation and theoretical comparaison"""
    n_dim, length, J_nn, beta_raw = load_ising_params(args.rundir)
    logZ_gfn = load_wandb_logZ(args.wandbdir)
    logZ_estimated = None
    # convert negative GFN beta to physical beta
    beta = abs(beta_raw)

    print("\nDetected Ising parameters:")
    print(f"  n_dim      = {n_dim}")
    print(f"  length     = {length}")
    print(f"  J_nn       = {J_nn}")
    print(f"  beta_raw   = {beta_raw}")
    print(f"  beta_phys  = {beta}\n")

    # Sample if needed
    maybe_sample(args)

    # Load samples
    samples, energies = load_samples_from_pkl(args.rundir)

    if args.estimate_logz:
        logZ_estimated = estimate_logz_from_eval(args, sampled_states=samples)

    # Theoretical quantities
    theory = compute_theoretical_quantities(beta, length)
    # Estimated quantities
    m_abs_magnetization = mean_abs_magnetization(samples)
    m_susceptibility = mean_susceptibility(samples, beta)
    m_energy = mean_energy(energies)
    C = heat_capacity(samples, energies, beta)

    # --- Collect all results in a dictionary ---
    results = {
        "run_type": "evaluation",
        "rundir": args.rundir,
        "n_samples": args.n_samples,
        "sampling_batch_size": args.sampling_batch_size,
        "n_dim": n_dim,
        "length": length,
        "J_nn": J_nn,
        "beta_raw": beta_raw,
        "beta_phys": beta,
        "logZ_gfn": logZ_gfn,
        "logZ_estimated": logZ_estimated,
        "mean_abs_magnetization": m_abs_magnetization,
        "mean_susceptibility": m_susceptibility,
        "mean_energy": m_energy,
        "heat_capacity": C,
        "logZ_theoretical": theory["logZ"],
    }

    print("\nEstimated quantities from GFN run:")
    print(f"  logZ from GFN                  = {logZ_gfn:.6f}")
    if logZ_estimated is not None:
        print(f"  logZ estimated from trajectories = {logZ_estimated:.6f}")
    print(f"  Mean absolute magnetization    = {m_abs_magnetization:.6f}")
    print(f"  Mean susceptibility            = {m_susceptibility:.6f}")
    print(f"  Mean energy                    = {m_energy:.6f}")
    print(f"  Heat capacity                  = {C:.6f}")

    print("\nTheoretical quantities:")
    print(f"logZ (theoretical) = {theory['logZ']:.6f}")

    # --- Save JSON ---
    out_dir = os.path.join(os.path.abspath(args.rundir), "eval", "samples")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {out_path}\n")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
def main():
    args = parse_args()

    if args.command == "run_theory":
        run_theory(args)

    elif args.command == "run_gfn_evaluation":
        run_gfn_evaluation(args)


if __name__ == "__main__":
    main()
