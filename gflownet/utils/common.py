from pathlib import Path
from os.path import expandvars
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import get_original_cwd, instantiate
from omegaconf import OmegaConf


def set_device(device: str):
    if device.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_float_precision(precision: int):
    if precision == 16:
        return torch.float16
    elif precision == 32:
        return torch.float32
    elif precision == 64:
        return torch.float64
    else:
        raise ValueError("Precision must be one of [16, 32, 64]")


def torch2np(x):
    if hasattr(x, "is_cuda") and x.is_cuda:
        x = x.detach().cpu()
    return np.array(x)


def handle_logdir():
    # TODO - just copy-pasted
    if "logdir" in config and config.logdir is not None:
        if not Path(config.logdir).exists() or config.overwrite_logdir:
            Path(config.logdir).mkdir(parents=True, exist_ok=True)
            with open(config.logdir + "/config.yml", "w") as f:
                yaml.dump(
                    numpy2python(namespace2dict(config)), f, default_flow_style=False
                )
            torch.set_num_threads(1)
            main(config)
        else:
            print(f"logdir {config.logdir} already exists! - Ending run...")
    else:
        print(f"working directory not defined - Ending run...")


def download_file_if_not_exists(path: str, url: str):
    """
    Download a file from google drive if path doestn't exist.
    url should be in the format: https://drive.google.com/uc?id=FILE_ID
    """
    import gdown

    path = Path(path)
    if not path.is_absolute():
        # to avoid storing downloaded files with the logs, prefix is set to the original working dir
        prefix = get_original_cwd()
        path = Path(prefix) / path
    if not path.exists():
        path.absolute().parent.mkdir(parents=True, exist_ok=True)
        gdown.download(url, str(path.absolute()), quiet=False)
    return path


def resolve_path(path: str) -> Path:
    return Path(expandvars(str(path))).expanduser().resolve()


def find_latest_checkpoint(ckpt_dir, pattern):
    final = list(ckpt_dir.glob(f"{pattern}*final*"))
    if len(final) > 0:
        return final[0]
    ckpts = list(ckpt_dir.glob(f"{pattern}*"))
    if not ckpts:
        raise ValueError(f"No checkpoints found in {ckpt_dir} with pattern {pattern}")
    return sorted(ckpts, key=lambda f: float(f.stem.split("iter")[1]))[-1]


def load_gflow_net_from_run_path(run_path, device="cuda"):
    device = str(device)
    run_path = resolve_path(run_path)
    hydra_dir = run_path / ".hydra"
    with initialize_config_dir(
        version_base=None, config_dir=str(hydra_dir), job_name="xxx"
    ):
        config = compose(config_name="config")
        print(OmegaConf.to_yaml(config))
    # Disable wandb
    config.logger.do.online = False
    # Logger
    logger = instantiate(config.logger, config, _recursive_=False)
    # The proxy is required in the env for scoring: might be an oracle or a model
    proxy = instantiate(
        config.proxy,
        device=device,
        float_precision=config.float_precision,
    )
    # The proxy is passed to env and used for computing rewards
    env = instantiate(
        config.env,
        proxy=proxy,
        device=device,
        float_precision=config.float_precision,
    )
    gflownet = instantiate(
        config.gflownet,
        device=device,
        float_precision=config.float_precision,
        env=env,
        buffer=config.env.buffer,
        logger=logger,
    )
    # Load final models
    ckpt_dir = Path(run_path) / config.logger.logdir.ckpts
    forward_latest = find_latest_checkpoint(
        ckpt_dir, config.gflownet.policy.forward.checkpoint
    )
    gflownet.forward_policy.model.load_state_dict(
        torch.load(forward_latest, map_location=device)
    )
    try:
        backward_latest = find_latest_checkpoint(
            ckpt_dir, config.gflownet.policy.backward.checkpoint
        )
        gflownet.backward_policy.model.load_state_dict(
            torch.load(backward_latest, map_location=device)
        )
    except AttributeError:
        print("No backward policy found")

    return gflownet
