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


class MXtalNetD(Proxy):
    def __init__(self, ckpt_path=None, scaling_func='score', **kwargs):
        """
        Wrapper class around the standalone Molecular Crystal stability proxy.

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

        self.rescaling_func = scaling_func


        print("Initializing MXtalNetD proxy:")
        print("  Fetching proxy Github code...")
        if not REPO_PATH.exists():
            # creatre $root/external/repos
            REPO_PATH.parent.mkdir(exist_ok=True, parents=True)
            # clone remote proxy code
            git.Repo.clone_from(REPO_URL, str(REPO_PATH))
        # at this point the repo must exist
        assert REPO_PATH.exists()
        # checkout the appropriate tag/release

        from MCryGAN.standalone import load_standalone_discriminator  # TODO make this function
        self.model = load_standalone_discriminator(ckpt_path, self.rescaling_func, self.device)


    @torch.no_grad()
    def __call__(self, cell_params, mol_data) -> TensorType["batch"]:

        return self.model(cell_params, mol_data)

    @torch.no_grad()
    def infer_on_train_set(self):
        """
        Infer on the training set and return the ground-truth and proxy values.
        No proxy here - this is more or less a dummy

        Returns:
        --------
        energy: torch.Tensor
            Ground-truth energies in the proxy's training set.
        proxy: torch.Tensor
            Proxy inference on its training set.
        """
        energy = []
        proxy = []

        for b in self.proxy_loaders["train"]:
            y = self.model(b.cell_params, b)
            e = y
            p = y
            energy.append(e)
            proxy.append(p)

        energy = torch.cat(energy).cpu()
        proxy = torch.cat(proxy).cpu()

        return energy, proxy
