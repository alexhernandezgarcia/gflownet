import os
import random
from os.path import expandvars
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import get_original_cwd, instantiate
from omegaconf import OmegaConf
from torchtyping import TensorType

from gflownet.utils.policy import parse_policy_config


def set_device(device: Union[str, torch.device]):
    if isinstance(device, torch.device):
        return device
    if device.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_float_precision(precision: Union[int, torch.dtype]):
    if isinstance(precision, torch.dtype):
        return precision
    if precision == 16:
        return torch.float16
    elif precision == 32:
        return torch.float32
    elif precision == 64:
        return torch.float64
    else:
        raise ValueError("Precision must be one of [16, 32, 64]")


def set_int_precision(precision: Union[int, torch.dtype]):
    if isinstance(precision, torch.dtype):
        return precision
    if precision == 16:
        return torch.int16
    elif precision == 32:
        return torch.int32
    elif precision == 64:
        return torch.int64
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


def find_latest_checkpoint(ckpt_dir, ckpt_name):
    ckpt_name = Path(ckpt_name).stem
    final = list(ckpt_dir.glob(f"{ckpt_name}*final*"))
    if len(final) > 0:
        return final[0]
    ckpts = list(ckpt_dir.glob(f"{ckpt_name}*"))
    if not ckpts:
        raise ValueError(
            f"No final checkpoints found in {ckpt_dir} with pattern {ckpt_name}*final*"
        )
    return sorted(ckpts, key=lambda f: float(f.stem.split("iter")[1]))[-1]


def load_gflow_net_from_run_path(
    run_path,
    no_wandb=True,
    print_config=False,
    device="cuda",
    load_final_ckpt=True,
):
    run_path = resolve_path(run_path)
    hydra_dir = run_path / ".hydra"

    with initialize_config_dir(
        version_base=None, config_dir=str(hydra_dir), job_name="xxx"
    ):
        config = compose(config_name="config")

    if print_config:
        print(OmegaConf.to_yaml(config))

    if no_wandb:
        # Disable wandb
        config.logger.do.online = False

    # Logger
    logger = instantiate(config.logger, config, _recursive_=False)
    # The proxy is required in the env for scoring: might be an oracle or a model
    proxy = instantiate(
        config.proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    # The proxy is passed to env and used for computing rewards
    env = instantiate(
        config.env,
        proxy=proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    forward_config = parse_policy_config(config, kind="forward")
    backward_config = parse_policy_config(config, kind="backward")
    forward_policy = instantiate(
        forward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
    )
    backward_policy = instantiate(
        backward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
        base=forward_policy,
    )
    gflownet = instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env=env,
        buffer=config.env.buffer,
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        logger=logger,
    )

    if not load_final_ckpt:
        return gflownet, config

    # -------------------------------
    # -----  Load final models  -----
    # -------------------------------

    ckpt = [f for f in run_path.rglob(config.logger.logdir.ckpts) if f.is_dir()][0]
    forward_final = find_latest_checkpoint(ckpt, config.policy.forward.checkpoint)
    gflownet.forward_policy.model.load_state_dict(
        torch.load(forward_final, map_location=set_device(device))
    )
    try:
        backward_final = find_latest_checkpoint(ckpt, config.policy.backward.checkpoint)
        gflownet.backward_policy.model.load_state_dict(
            torch.load(backward_final, map_location=set_device(device))
        )
    except ValueError:
        print("No backward policy found")
    return gflownet, config


def batch_with_rest(start, stop, step, tensor=False):
    for i in range(start, stop, step):
        if tensor:
            yield torch.arange(i, min(i + step, stop))
        else:
            yield np.arange(i, min(i + step, stop))


def tfloat(x, device, float_type):
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).to(device=device, dtype=float_type)
    if torch.is_tensor(x):
        return x.to(device=device, dtype=float_type)
    else:
        return torch.tensor(x, dtype=float_type, device=device)


def tlong(x, device):
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).to(device=device, dtype=torch.long)
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.long)
    else:
        return torch.tensor(x, dtype=torch.long, device=device)


def tint(x, device, int_type):
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).to(device=device, dtype=int_type)
    if torch.is_tensor(x):
        return x.to(device=device, dtype=int_type)
    else:
        return torch.tensor(x, dtype=int_type, device=device)


def tbool(x, device):
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).to(device=device, dtype=torch.bool)
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.bool)
    else:
        return torch.tensor(x, dtype=torch.bool, device=device)


def concat_items(list_of_items, index=None):
    if isinstance(list_of_items[0], np.ndarray):
        result = np.concatenate(list_of_items)
        if index is not None:
            index = index.cpu().numpy()
            result = result[index]
    elif torch.is_tensor(list_of_items[0]):
        result = torch.cat(list_of_items)
        if index is not None:
            result = result[index]
    else:
        raise NotImplementedError(
            "cannot concatenate {}".format(type(list_of_items[0]))
        )

    return result


def extend(
    orig: Union[List, TensorType["..."]], new: Union[List, TensorType["..."]]
) -> Union[List, TensorType["..."]]:
    assert type(orig) == type(new)
    if isinstance(orig, list):
        orig.extend(new)
    elif torch.tensor(orig):
        orig = torch.cat([orig, new])
    else:
        raise NotImplementedError(
            "Extension only supported for lists and torch tensors"
        )
    return orig


def copy(x: Union[List, TensorType["..."]]):
    if torch.is_tensor(x):
        return x.clone().detach()
    else:
        return x.copy()


def chdir_random_subdir():
    """
    Creates a directory with random name and changes current working directory to it.

    Aimed as a hotfix for race conditions: currently, by default, the directory in
    which the experiment will be logged is named based on the current timestamp. If
    multiple jobs start at exactly the same time, they can be trying to log to
    the same directory. In particular, this causes issues when using dataset
    evaluation (e.g., JSD computation).
    """
    cwd = os.getcwd()
    cwd += "/%08x" % random.getrandbits(32)
    os.mkdir(cwd)
    os.chdir(cwd)


def bootstrap_samples(tensor, num_samples):
    """
    Bootstraps tensor along the last dimention
    returns tensor of the shape [initial_shape, num_samples]
    """
    dim_size = tensor.size(-1)
    bs_indices = torch.randint(
        0, dim_size, size=(num_samples * dim_size,), device=tensor.device
    )
    bs_samples = torch.index_select(tensor, -1, index=bs_indices)
    bs_samples = bs_samples.view(
        tensor.size()[:-1] + (num_samples, dim_size)
    ).transpose(-1, -2)
    return bs_samples


def example_documented_function(arg1, arg2):
    r"""Summary line: this function is not used anywhere, it's just an example.

    Extended description of function from the docstrings tutorial :ref:`write
    docstrings-extended`.

    Refer to

    * functions with :py:func:`gflownet.utils.common.set_device`
    * classes with :py:class:`gflownet.gflownet.GFlowNetAgent`
    * methods with :py:meth:`gflownet.envs.base.GFlowNetEnv.get_action_space`
    * constants with :py:const:`gflownet.envs.base.CMAP`

    Prepenend with ``~`` to refer to the name of the object only instead of the full
    path -> :py:func:`~gflownet.utils.common.set_device` will display as ``set_device``
    instead of the full path.

    Great maths:

    .. math::

        \int_0^1 x^2 dx = \frac{1}{3}

    .. important::

        A docstring with **math** MUST be a raw Python string (a string prepended with
        an ``r``: ``r"raw"``) to avoid backslashes being treated as escape characters.

        Alternatively, you can use double backslashes.

    .. warning::

        Display a warning. See :ref:`learn by example`. (<-- this is a cross reference,
        learn about it `here
        <https://www.sphinx-doc.org/en/master/usage/referencing.html#ref-rolel>`_)


    Examples
    --------
    >>> function(1, 'a')
    True
    >>> function(1, 2)
    True

    >>> function(1, 1)
    Traceback (most recent call last):
        ...

    Notes
    -----
    This block uses ``$ ... $`` for inline maths -> $e^{\frac{x}{2}}$.

    Or ``$$ ... $$`` for block math instead of the ``.. math:`` directive above.

    $$\int_0^1 x^2 dx = \frac{1}{3}$$


    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        Description of return value
    """
    if arg1 == arg2:
        raise ValueError("arg1 must not be equal to arg2")
    return True
