from collections.abc import MutableMapping
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from hydra.utils import get_original_cwd
from torchtyping import TensorType


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


def tfloat(x, device, float_type):
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).type(float_type).to(device)
    if torch.is_tensor(x):
        return x.type(float_type).to(device)
    else:
        return torch.tensor(x, dtype=float_type, device=device)


def tlong(x, device):
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).type(torch.long).to(device)
    if torch.is_tensor(x):
        return x.type(torch.long).to(device)
    else:
        return torch.tensor(x, dtype=torch.long, device=device)


def tint(x, device):
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).type(torch.int).to(device)
    if torch.is_tensor(x):
        return x.type(torch.int).to(device)
    else:
        return torch.tensor(x, dtype=torch.int, device=device)


def tbool(x, device):
    if isinstance(x, list) and torch.is_tensor(x[0]):
        return torch.stack(x).type(torch.bool).to(device)
    if torch.is_tensor(x):
        return x.type(torch.bool).to(device)
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


def copy(x: Union[List, TensorType["..."]]):
    if torch.is_tensor(x):
        return x.clone().detach()
    else:
        return x.copy()
