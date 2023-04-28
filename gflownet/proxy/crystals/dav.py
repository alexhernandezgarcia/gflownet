from pathlib import Path

import pandas as pd
import torch
from torchtyping import TensorType
from gflownet.proxy.base import Proxy
import git
import sys

ROOT = Path(__file__).resolve().parent.parent.parent.parent
REPO_PATH = ROOT / "external" / "repos" / "ActiveLearningMaterials"
REPO_URL = "https://github.com/sh-divya/ActiveLearningMaterials.git"


def checkout_tag(tag):
    repo = git.Repo(str(REPO_PATH))
    assert (
        tag in repo.tags
    ), f"Tag {tag} not found in repo {REPO_PATH}. Verify the `release` config."
    repo.git.checkout(repo.tags[tag].path)


class DAV(Proxy):
    def __init__(self, normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        if not REPO_PATH.exists():
            REPO_PATH.parent.mkdir(exist_ok=True, parents=True)
            git.Repo.clone_from(REPO_URL, str(REPO_PATH))
        assert REPO_PATH.exists()
        assert checkout_tag(kwargs["release"])
        sys.path.append(str(REPO_PATH))

        from proxies.models import make_model

        ckpt_path = Path(kwargs["ckpt_path"]).resolve()
        assert ckpt_path.exists(), f"Checkpoint {str(ckpt_path)} not found."
        ckpt = torch.load(str(ckpt_path))
        self.model_config = ckpt["hyper_parameters"]
        self.model = make_model(self.model_config)
        self.model.load_state_dict(ckpt["state_dict"])

    def __call__(
        self, states: TensorType["batch", "96"]  # noqa: F821
    ) -> TensorType["batch"]:  # noqa: F821
        assert states.shape[-1] == self.model.pred_inp_size
        comp = states[:, : self.model.n_elements]
        sg = states[:, self.model.n_elements :].int()
        lat_params = states[:, -6:]
        assert comp.shape[-1] + sg.shape[-1] + lat_params.shape[-1] == states.shape[-1]
        x = (comp, sg, lat_params)
        return self.model(x).squeeze(-1)
