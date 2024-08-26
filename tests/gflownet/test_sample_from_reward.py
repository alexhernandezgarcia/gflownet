import os
from pathlib import Path

import numpy as np
import yaml
from hydra import compose, initialize
from omegaconf import OmegaConf

from gflownet.utils.common import gflownet_from_config


def test_nested_sampling_simple_check():
    ROOT = Path(__file__).resolve().parent.parent.parent
    command = "+experiments=icml23/ctorus logger.do.online=False"
    overrides = command.split()
    with initialize(
        version_base="1.1",
        config_path=os.path.relpath(
            str(ROOT / "config"), start=str(Path(__file__).parent)
        ),
        job_name="xxx",
    ):
        config = compose(config_name="main", overrides=overrides)

    gfn = gflownet_from_config(config)
    samples = gfn.sample_from_reward(100, method="nested")
    assert samples.shape[0] == 100
    assert (samples[:, 0] < np.pi * 2).all()
    assert (samples[:, 1] < np.pi * 2).all()
    assert (samples[:, 0] > 0).all()
    assert (samples[:, 1] > 0).all()
