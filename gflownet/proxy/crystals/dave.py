from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version

import torch
from dave import DavePredictor
from torchtyping import TensorType

from gflownet.proxy.base import Proxy

REPO_URL = "https://github.com/sh-divya/crystalproxies.git"
RELEASE = "2.0.6"
"""
URL to the proxy's code repository. It is used to provide a link to the appropriate
release link in case of version mismatch between requested and installed ``dave``
release.
"""


class DAVE(Proxy):
    def __init__(
        self,
        ckpt_path=str,
        **kwargs,
    ):
        """
        Wrapper class around the Dave (Divya-Alexandre-Victor) proxy.

        Parameters
        ----------
        ckpt_path : str
            Path to a directory containing the checkpoint of a pre-trained model.
        """
        super().__init__(**kwargs)

        print("Initializing DAVE proxy:")
        try:
            dave_version = version("dave")
        except PackageNotFoundError:
            print("  💥 `dave` cannot be imported.")
            print("    Install with:")
            print(f"    $ pip install git+{REPO_URL}@{RELEASE}\n")
            raise PackageNotFoundError("DAVE not found")

        if dave_version != RELEASE:
            print("  💥 `dave` version mismatch: ")
            print(f"    current ({dave_version}) != requested ({RELEASE})")
            print("    Install the requested version with:")
            print(f"    $ pip install --upgrade git+{REPO_URL}@{RELEASE}\n")
            raise ImportError("Wrong DAVE version")

        print("  Found version:", dave_version)
        print("  Loading model weights...")

        # Initialize Dave model and load weights from checkpoint
        self.model = DavePredictor(path_to_weights=ckpt_path, device=self.device)

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
        comp = states[:, :-7]
        sg = states[:, -7]
        lat_params = states[:, -6:]

        # model forward
        x = (comp.long(), sg.long(), lat_params.float())
        y = self.model(x, scale_input=True).squeeze(-1)

        return y

    # TODO: review whether rescaling is done as expected
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
        energy = []
        proxy = []

        for b in self.proxy_loaders["train"]:
            x, e = b
            for k, t in enumerate(x):
                if t.ndim == 1:
                    x[k] = t[:, None]
                if t.ndim > 2:
                    x[k] = t.squeeze()
                assert (
                    x[k].ndim == 2
                ), f"t.ndim = {x[k].ndim} != 2 (t.shape: {x[k].shape})"
            p = self.proxy(torch.cat(x, dim=-1), scale_input=True)
            energy.append(e)
            proxy.append(p)

        energy = torch.cat(energy).cpu()
        proxy = torch.cat(proxy).cpu()

        return energy, proxy
