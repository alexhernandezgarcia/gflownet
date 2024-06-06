from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version

import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy

REPO_URL = "https://github.com/sh-divya/ActiveLearningMaterials.git"
"""
URL to the proxy's code repository. It is used to provide a link to the
appropriate release link in case of version mismatch between requested
and installed ``dave`` release.
"""


class DAVE(Proxy):
    def __init__(
        self,
        ckpt_path=None,
        release=None,
        rescale_outputs=True,
        **kwargs,
    ):
        """
        Wrapper class around the Divya-Alexandre-Victor proxy.

        * git clone the repo
        * checkout the appropriate tag/release as per ``release``
        * import the proxy build function ``make_model`` by updating ``sys.path``
        * load the checkpoint from ``ckpt_path`` and build the proxy model

        The checkpoint path is resolved as follows: * if ``ckpt_path`` is a dict, it is
        assumed to be a mapping from cluster or
             ``$USER`` to path (e.g. ``{mila: /path/ckpt.ckpt, victor:
             /path/ckpt.ckpt}``)
        * on the cluster, the path to the ckpt is public so everyone resolves to
            ``"mila"``. For local dev you need to specify a path in ``dave.yaml`` that
            maps to your local ``$USER``.
        * if the resulting path is a dir, it must contain exactly one ``.ckpt`` file
        * if the resulting path is a file, it must be a ``.ckpt`` file

        Parameters
        ----------
        ckpt_path : dict, optional
            Mapping from cluster / ``$USER`` to checkpoint, by default None
        release : str, optional
            Tag to checkout in the DAVE repo, by default None
        rescale_outputs : bool, optional
            Whether to rescale the proxy outputs using its training mean and std, by
            default True
        """
        super().__init__(**kwargs)
        self.rescale_outputs = rescale_outputs
        self.release = release

        self.is_eform = self.is_bandgap = False

        if release.startswith("0."):
            self.is_eform = True
        elif release.startswith("1."):
            assert self.reward_function.startswith("rbf_exp"), (
                "The RBF exponential reward function must be used with band gap models "
                "in order for the reward to reflect proximity to a target value. "
                f"{self.reward_function} has been used instead."
            )
            assert (
                "center" in self.reward_function_kwargs
                and self.reward_function_kwargs["center"] is not None
            ), (
                "A target band gap (reward_function_kwargs.center must be specified "
                "for releases 1.x.x (i.e. band gap models)"
            )
            bandgap_target = self.reward_function_kwargs["center"]
            assert (
                torch.is_tensor(bandgap_target) and bandgap_target.dtype == self.float
            ), (
                "reward_function_kwargs.center must be a float (received "
                f"{bandgap_target}: {type(bandgap_target)})"
            )
            self.is_bandgap = True
        else:
            raise ValueError(f"Unknown release: {release}. Allowed: 0.x.x or 1.x.x")

        self.scaled = False
        if "clip" in kwargs:
            self.clip = kwargs["clip"]
        else:
            self.clip = False

        print("Initializing DAVE proxy:")
        print("  Checking out release:", release)

        pip_url = f"{REPO_URL}@{release}"

        try:
            dave_version = version("dave")
        except PackageNotFoundError:
            print("  💥 `dave` cannot be imported.")
            print("    Install with:")
            print(f"    $ pip install git+{pip_url}\n")

            raise PackageNotFoundError("DAVE not found")

        if dave_version != release:
            print("  💥 `dave` version mismatch: ")
            print(f"    current ({dave_version}) != requested ({release})")
            print("    Install the requested version with:")
            print(f"    $ pip install --upgrade git+{pip_url}\n")
            raise ImportError("Wrong DAVE version")

        print("  Found version:", dave_version)
        print("  Loading model weights...")

        from dave import prepare_for_gfn

        self.model, self.proxy_loaders, self.scales = prepare_for_gfn(
            ckpt_path, release, self.rescale_outputs
        )

        self.model.to(self.device)

    def _set_scales(self):
        """
        Sets the scales to the device and converts them to float if needed.
        """
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
    def __call__(self, states: TensorType["batch", "102"]) -> TensorType["batch"]:
        """
        Forward pass of the proxy.

        The proxy will decompose the state as:
        * composition: ``states[:, :-7]`` -> length 95 (dummy 0 then 94 elements)
        * space group: ``states[:, -7] - 1``
        * lattice parameters: ``states[:, -6:]``

        >>> composition MUST be a list of ATOMIC NUMBERS, prepended with a 0.
        >>> dummy padding value at comp[0] MUST be 0.
        ie -> comp[i] -> element Z=i
        ie -> LiO2 -> [0, 0, 0, 1, 0, 0, 2, 0, ...] up until Z=94 for the MatBench proxy
        ie -> len(comp) = 95 (0 then 94 elements)

        >>> sg MUST be a list of ACTUAL space group numbers (1-230)

        >>> lat_params MUST be a list of lattice parameters in the following order:
        [a, b, c, alpha, beta, gamma] as floats.

        >>> the states tensor MUST already be on the device.

        Parameters
        ----------
        states : torch.Tensor
            States to infer on. Shape: ``(batch, [6 + 1 + n_elements])``.

        Returns
        -------
        torch.Tensor
            Proxy energies. Shape: ``(batch,)``.
        """
        self._set_scales()

        comp = states[:, :-7]
        sg = states[:, -7]
        lat_params = states[:, -6:]

        if self.rescale_outputs:
            lat_params = (lat_params - self.scales["x"]["mean"]) / self.scales["x"][
                "std"
            ]

        # model forward
        x = (comp.long(), sg.long(), lat_params.float())
        y = self.model(x).squeeze(-1)

        if self.rescale_outputs:
            y = y * self.scales["y"]["std"] + self.scales["y"]["mean"]

        if self.clip and self.clip.do:
            if self.rescale_outputs:
                if self.clip.min_stds:
                    y_min = -1.0 * self.clip.min_stds * self.scales["y"]["std"]
                else:
                    y_min = None
                if self.clip.max_stds:
                    y_max = self.clip.max_stds * self.scales["y"]["std"]
                else:
                    y_max = None
            else:
                y_min = self.clip.min
                y_max = self.clip.max

            y = torch.clamp(min=y_min, max=y_max)

        return y

    @torch.no_grad()
    def infer_on_train_set(self):
        """
        Infer on the training set and return the ground-truth and proxy values.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(energy, proxy)`` representing 1/ ground-truth energies and 2/
                proxy inference on the proxy's training set as 1D tensors.
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
