from copy import deepcopy
from pathlib import Path

import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


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
        print("  Checking out release:", release)

        from importlib.metadata import PackageNotFoundError, version

        pip_url = f"https://github.com/sh-divya/ActiveLearningMaterials.git@{release}"

        try:
            dave_version = version("dave")
        except PackageNotFoundError:
            print("  `dave` cannot be imported.")
            print(f"  Install with: `pip install git+{pip_url}`\n")

            raise PackageNotFoundError("DAVE not found")

        if dave_version != release:
            print("  💥 DAVE version mismatch: ")
            print(f"    current ({dave_version}) != requested ({release})")
            print(f"    Install the requested version with:")
            print(f"    `pip install --upgrade git+{pip_url}`\n")
            raise ImportError("Wrong DAVE version")

        print("  Found version:", dave_version)
        print("  Loading model from:", str(Path(ckpt_path) / release))

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
