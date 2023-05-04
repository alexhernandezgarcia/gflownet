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
    for remote in repo.remotes:
        remote.fetch()
    if tag not in repo.tags:
        raise ValueError(
            f"Tag {tag} not found in repo {str(REPO_PATH)}. Available tags: {repo.tags}"
            + "\nVerify the `release` config."
        )
    repo.git.checkout(repo.tags[tag].path)


def resolve(path: str) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


def find_ckpt(ckpt_path: dict) -> Path:
    loc = os.environ.get(
        "SLURM_CLUSTER_NAME", os.environ.get("SLURM_JOB_ID", os.environ["USER"])
    )
    if all(s.isdigit() for s in loc):
        loc = "mila"
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
    def __init__(self, ckpt_path=None, release=None, rescale_outputs=True, **kwargs):
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
            rescale_outputs (bool, optional): Whether to rescale the proxy outputs
                using its training mean and std. Defaults to True.
        """
        super().__init__(**kwargs)
        self.rescale_outputs = rescale_outputs
        self.scaled = False

        print("Initializing DAV proxy:")
        print("  Fetching proxy Github code...")
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

        print("  Making model...")
        # load the checkpoint
        ckpt_path = find_ckpt(ckpt_path)
        assert ckpt_path.exists(), f"Checkpoint {str(ckpt_path)} not found."
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        # extract config
        self.model_config = ckpt["hyper_parameters"]
        self.scales = self.model_config.get("scales")
        if self.rescale_outputs:
            assert self.scales is not None
            assert all(t in self.scales for t in ["x", "y"])
            assert all(u in self.scales[t] for t in ["x", "y"] for u in ["mean", "std"])
        # make model from ckpt config
        self.model = make_model(self.model_config)
        # load state dict and remove potential leading `model.` in the keys
        print("  Loading proxy checkpoint...")
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
        print("Proxy ready.")

    def _set_scales(self):
        if self.scaled:
            return
        if self.rescale_outputs:
            if self.model_config["scales"]["x"]["mean"].device != self.device:
                self.scales["x"]["mean"] = self.scales["x"]["mean"].to(self.device)
                self.scales["x"]["std"] = self.scales["x"]["std"].to(self.device)
                self.scales["y"]["mean"] = self.scales["y"]["mean"].to(self.device)
                self.scales["y"]["std"] = self.scales["y"]["std"].to(self.device)
            if self.scales["x"]["mean"].ndim == 1:
                self.scales["x"]["mean"] = self.scales["x"]["mean"][None, :].float()
                self.scales["x"]["std"] = self.scales["x"]["std"][None, :].float()
                self.scales["y"]["mean"] = self.scales["y"]["mean"][None].float()
                self.scales["y"]["std"] = self.scales["y"]["std"][None].float()
        self.scaled = True

    @torch.no_grad()
    def __call__(self, states: TensorType["batch", "96"]) -> TensorType["batch"]:
        self._set_scales()

        comp = states[:, :-7]
        sg = states[:, -7] - 1
        lat_params = states[:, -6:]

        n_env = comp.shape[-1]
        if n_env != self.model.n_elements:
            missing = torch.zeros(
                (len(comp), self.model.n_elements - n_env), device=comp.device
            )
            comp = torch.cat([comp, missing], dim=-1)

        if self.rescale_outputs:
            lat_params = (lat_params - self.scales["x"]["mean"]) / self.scales["x"][
                "std"
            ]

        # model forward
        x = (comp.long(), sg.long(), lat_params.float())
        y = self.model(x).squeeze(-1)

        if self.rescale_outputs:
            y = y * self.scales["y"]["std"] + self.scales["y"]["mean"]

        return y
