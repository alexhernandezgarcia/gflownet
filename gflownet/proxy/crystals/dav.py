import os
from pathlib import Path

import torch
from torchtyping import TensorType
from gflownet.proxy.base import Proxy
import git
import sys

# gflownet/ repo root
ROOT = Path(__file__).resolve().parent.parent.parent.parent
# where to clone / find the external code
REPO_PATH = ROOT / "external" / "repos" / "ActiveLearningMaterials"
# remote repo url to clone from
REPO_URL = "https://github.com/sh-divya/ActiveLearningMaterials.git"


def checkout_tag(tag):
    """
    Changes the proxy's repo state to the specified tag.

    Args:
        tag (str): Tag/release to checkout
    """
    repo = git.Repo(str(REPO_PATH))
    assert (
        tag in repo.tags
    ), f"Tag {tag} not found in repo {str(REPO_PATH)}. Verify the `release` config."
    repo.git.checkout(repo.tags[tag].path)


def resolve(path: str) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


def find_ckpt(ckpt_path: dict) -> Path:
    loc = os.environ.get("SLURM_CLUSTER_NAME", os.environ["USER"])
    if loc not in ckpt_path:
        raise ValueError(f"DAV proxy checkpoint path not found for location {loc}.")
    path = resolve(ckpt_path[loc])
    if not path.exists():
        raise ValueError(f"DAV proxy checkpoint not found at {str(path)}.")
    if path.is_file():
        return path
    ckpts = list(path.glob("*.ckpt"))
    if len(ckpts) == 0:
        raise ValueError(f"No DAV proxy checkpoint found at {str(path)}.")
    if len(ckpts) > 1:
        raise ValueError(
            f"Multiple DAV proxy checkpoints found at {str(path)}. "
            "Please specify the checkpoint explicitly."
        )
    return ckpts[0]


class DAV(Proxy):
    def __init__(self, ckpt_path=None, release=None, **kwargs):
        """
        Wrapper class around the Divya-Alexandre-Victor proxy.

        * git clone the repo
        * checkout the appropriate tag/release as per ``release``
        * import the proxy build function ``make_model`` by updating ``sys.path``
        * load the checkpoint from ``ckpt_path`` and build the proxy model

        The checkpoint path is resolved as follows:
        * if ``ckpt_path`` is a dict, it is assumed to be a mapping from cluster
            or $USER to path (e.g. {mila: /path/ckpt.ckpt, victor: /path/ckpt.ckpt})
        * on the cluster, the path to the ckpt is public so everyone resolves to
            "mila". For local dev you need to specify a path in dav.yaml that maps
            to your local $USER.
        * if the resulting path is a dir, it must contain exactly one .ckpt file
        * if the resulting path is a file, it must be a .ckpt file

        Args:
            ckpt_path (dict, optional): Mapping from cluster / ``$USER`` to checkpoint.
                Defaults to None.
            release (str, optional): Tag to checkout in the DAV repo. Defaults to None.
        """
        super().__init__(**kwargs)
        if not REPO_PATH.exists():
            # creatre $root/external/repos
            REPO_PATH.parent.mkdir(exist_ok=True, parents=True)
            # clone remote proxy code
            git.Repo.clone_from(REPO_URL, str(REPO_PATH))
        # at this point the repo must exist
        assert REPO_PATH.exists()
        # checkout the appropriate tag/release
        checkout_tag(release)

        # import the proxu build funcion
        sys.path.append(str(REPO_PATH))
        from proxies.models import make_model

        # load the checkpoint
        ckpt_path = find_ckpt(ckpt_path)
        assert ckpt_path.exists(), f"Checkpoint {str(ckpt_path)} not found."
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        # extract config
        self.model_config = ckpt["hyper_parameters"]
        # make model from ckpt config
        self.model = make_model(self.model_config)
        # load state dict and remove potential leading `model.` in the keys
        self.model.load_state_dict(
            {
                k[6:] if k.startswith("model.") else k: v
                for k, v in ckpt["state_dict"].items()
            }
        )
        assert hasattr(self.model, "pred_inp_size")
        self.model.n_elements = 89  # TEMPORARY for release `v0-dev-embeddings`
        assert hasattr(self.model, "n_elements")
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def __call__(self, states: TensorType["batch", "96"]) -> TensorType["batch"]:
        # state shape and model expected input shape must match
        assert states.shape[-1] == self.model.pred_inp_size
        # split state in individual tensors
        comp = states[:, : self.model.n_elements]
        sg = states[:, self.model.n_elements].int()
        lat_params = states[:, -6:]
        # check that the split is correct
        assert comp.shape[-1] + sg.shape[-1] + lat_params.shape[-1] == states.shape[-1]
        x = (comp, sg, lat_params)
        # model forward
        return self.model(x).squeeze(-1)
