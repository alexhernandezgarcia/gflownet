import os
import re
import sys
from copy import deepcopy
from pathlib import Path

import git
import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy

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
    repo = git.Repo(REPO_PATH)
    major_revison = re.findall(r"v(\d+)", tag)[0]
    if int(major_revison) < 2:
        raise ValueError("Version is too old. Please use Dave v2.0.0 or later.")
    if repo.git.describe("--tags") == tag:
        return
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


class DAVE(Proxy):
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
            "mila". For local dev you need to specify a path in dave.yaml that maps
            to your local $USER.
        * if the resulting path is a dir, it must contain exactly one .ckpt file
        * if the resulting path is a file, it must be a .ckpt file

        Args:
            ckpt_path (dict, optional): Mapping from cluster / ``$USER`` to checkpoint.
                Defaults to None.
            release (str, optional): Tag to checkout in the DAVE repo. Defaults to None.
            rescale_outputs (bool, optional): Whether to rescale the proxy outputs
                using its training mean and std. Defaults to True.
        """
        super().__init__(**kwargs)
        self.rescale_outputs = rescale_outputs
        self.scaled = False

        print("Initializing DAVE proxy:")
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
        from dave import prepare_for_gfn

        self.model, self.proxy_loaders, self.scales = prepare_for_gfn(
            ckpt_path, release, self.rescale_outputs
        )

        self.model.to(self.device)

    def _set_scales(self):
        if self.scaled:
            return
        if self.rescale_outputs:
            if self.scales["x"]["mean"].device != self.device:
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

    @torch.no_grad()
    def infer_on_train_set(self):
        """
        Infer on the training set and return the ground-truth and proxy values.

        Returns:
        --------
        energy: torch.Tensor
            Ground-truth energies in the proxy's training set.
        proxy: torch.Tensor
            Proxy inference on its training set.
        """
        rso = deepcopy(self.rescale_outputs)
        self.rescale_outputs = False
        y_mean = self.scales["y"]["mean"]
        y_std = self.scales["y"]["std"]

        energy = []
        proxy = []

        for b in self.proxy_loaders["train"]:
            x, e_normed = b
            for k, t in enumerate(x):
                if t.ndim == 1:
                    x[k] = t[:, None]
                if t.ndim > 2:
                    x[k] = t.squeeze()
                assert (
                    x[k].ndim == 2
                ), f"t.ndim = {x[k].ndim} != 2 (t.shape: {x[k].shape})"
            p_normed = self.proxy(torch.cat(x, dim=-1))
            e = e_normed * y_std + y_mean
            p = p_normed * y_std + y_mean
            energy.append(e)
            proxy.append(p)

        self.rescale_outputs = rso

        energy = torch.cat(energy).cpu()
        proxy = torch.cat(proxy).cpu()

        return energy, proxy
