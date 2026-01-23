"""
Computes evaluation metrics and plots from a pre-trained GFlowNet model.
"""

import pickle
import shutil
import sys
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from hydra.utils import instantiate

from gflownet.utils.common import load_gflownet_from_rundir, read_hydra_config


@hydra.main(config_path="./config", config_name="eval", version_base="1.1")
def main(config):

    # Print configuration
    print(OmegaConf.to_yaml(config))

    if config.randominit:
        prefix = "randominit"
        load_last_checkpoint = False
    else:
        prefix = "gfn"
        load_last_checkpoint = True

    print(f"Loading GFlowNet from the configuration in {config.rundir}...")
    gflownet, run_config = load_gflownet_from_rundir(
        rundir=config.rundir,
        device=config.device,
        no_wandb=True,
        print_config=config.print_config,
        load_last_checkpoint=load_last_checkpoint,
    )
    env = gflownet.env

    base_dir = Path(config.output_dir or config.rundir)

    # ---------------------------------
    # -----  Test GFlowNet model  -----
    # ---------------------------------

    if not config.samples_only:
        gflownet.evaluator.n = config.n_samples
        eval_results = gflownet.evaluator.eval()

        # TODO-V: legacy -> ok to remove?
        # keys = ["True reward and GFlowNet samples", "GFlowNet KDE Policy", "Reward KDE"]
        # fignames = ["samples", "kde_gfn", "kde_reward"]

        output_dir = base_dir / "figures"
        print("output_dir: ", str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        if "figs" in eval_results:
            for figname, fig in eval_results["figs"].items():
                output_fig = output_dir / (path_compatible(figname) + ".pdf")
                if fig is not None:
                    fig.savefig(output_fig, bbox_inches="tight")
            print(f"Saved figures to {output_dir}")

        # Print metrics
        if "metrics" in eval_results:
            print("Metrics:")
            for k, v in eval_results["metrics"].items():
                print(f"\t{k}: {v:.4f}")

    # ------------------------------------------
    # -----  Sample GFlowNet  -----
    # ------------------------------------------

    # Read conditional environment config, if provided
    if config.conditional_env_config_path is not None:
        conditional_env_config_path = Path(config.conditional_env_config_path)
        if conditional_env_config_path.parent == Path("."):
            conditional_env_config_path = (
                Path(config.rundir) / ".hydra" / conditional_env_config_path.name
            )
        config_cond_env = read_hydra_config(config_name=conditional_env_config_path)
        if "env" in config_cond_env:
            config_cond_env = config_cond_env.env
        env_cond = instantiate(
            config_cond_env,
            device=run_config.device,
            float_precision=run_config.float_precision,
        )
    else:
        env_cond = None

    # Handle output directory
    output_dir = base_dir / "eval" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if config.n_samples > 0 and config.n_samples <= 1e5:
        print(
            f"Sampling {config.n_samples} forward trajectories",
            f"from GFlowNet in batches of {config.sampling_batch_size}",
        )
        with torch.no_grad():
            for i, bs in enumerate(
                tqdm(get_batch_sizes(config.n_samples, config.sampling_batch_size))
            ):
                batch, times = gflownet.sample_batch(
                    n_forward=bs, env_cond=env_cond, train=False
                )
                x_proxy = batch.get_terminating_states(proxy=True)
                energies = gflownet.proxy(x_proxy)
                x_sampled = batch.get_terminating_states()
                df = pd.DataFrame(
                    {
                        "readable": [env.state2readable(x) for x in x_sampled],
                        "energies": energies.tolist(),
                    }
                )
                df.to_csv(tmp_dir / f"gfn_samples_{i}.csv")
                dct = {"x": x_sampled, "energy": energies.tolist(), "proxy": x_proxy.tolist()}
                pickle.dump(dct, open(tmp_dir / f"gfn_samples_{i}.pkl", "wb"))

        # Concatenate all samples
        print("Concatenating sample CSVs")
        df = pd.concat(
            [pd.read_csv(f, index_col=0) for f in tqdm(list(tmp_dir.glob("*.csv")))]
        )
        df.to_csv(output_dir / f"{prefix}_samples.csv")
        dct = {k: [] for k in dct.keys()}
        for f in tqdm(list(tmp_dir.glob("*.pkl"))):
            tmp_dict = pickle.load(open(f, "rb"))
            dct = {k: v + tmp_dict[k] for k, v in dct.items()}
        pickle.dump(dct, open(output_dir / f"{prefix}_samples.pkl", "wb"))

        if "y" in input("Delete temporary files? (y/n)"):
            shutil.rmtree(tmp_dir)

    else:
        print(
            "Skipping sampling from GFlowNet. To enable sampling, set n_samples to an ",
            "integer between 0 and 100,000.",
        )


def get_batch_sizes(total, b=1):
    """
    Batches an iterable into chunks of size n and returns their expected lengths.

    Example
    -------

    .. code-block:: python

        >>> get_batch_sizes(10, 3)
        [3, 3, 3, 1]

    Parameters
    ----------
    total : int
        total samples to produce
    b : int
        the batch size

    Returns
    -------
    list
        list of batch sizes
    """
    n = total // b
    chunks = [b] * n
    if total % b != 0:
        chunks += [total % b]
    return chunks


def set_device(device: str):
    if device.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def path_compatible(str):
    """
    Replace all non-alphanumeric characters with underscores

    Parameters
    ----------
    str : str
        The string to be made compatible
    """
    return "".join([c if c.isalnum() else "_" for c in str])


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    main()
    sys.exit()
