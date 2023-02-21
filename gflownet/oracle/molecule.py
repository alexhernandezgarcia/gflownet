from typing import Union

import numpy as np
import numpy.typing as npt
import torch
import torchani

from gflownet.proxy.base import Proxy


MODELS = {
    "ANI1x": torchani.models.ANI1x,
    "ANI1ccx": torchani.models.ANI1ccx,
    "ANI2x": torchani.models.ANI2x,
}


class MoleculeEnergy(Proxy):
    def __init__(self, model: str = "ANI2x", use_ensemble: bool = True, **kwargs):
        """
        Parameters
        ----------
        model : str
            The name of the pretrained model to be used for prediction.

        use_ensemble : bool
            Whether to use whole ensemble of the models for prediction or only the first one.
        """
        if MODELS.get(model) is None:
            raise ValueError(
                f'Tried to use model "{model}", '
                f"but only {set(MODELS.keys())} are available."
            )

        self.model = MODELS[model](
            periodic_table_index=True, model_index=None if use_ensemble else 0
        )

        super().__init__(**kwargs)

    def set_device(self, device):
        super().set_device(device)

        self.model.to(self.device)

    @torch.no_grad()
    def __call__(
        self,
        elements: Union[npt.NDArray[np.int64], torch.LongTensor],
        coordinates: Union[npt.NDArray[np.float32], torch.FloatTensor],
    ) -> npt.NDArray[np.float32]:
        """
        Args
        ----
        elements : tensor
            Either numpy or torch tensor with dimensionality (batch_size, n_atoms),
            with values indicating atomic number of individual atoms.

        coordinates : tensor
            Either numpy or torch tensor with dimensionality (batch_size, n_atoms, 3),
            with values indicating 3D positions of individual atoms.

        Returns
        ----
        energies : tensor
            Numpy array with dimensionality (batch_size,), containing energies
            predicted by a TorchANI model (in Hartree).
        """
        if isinstance(elements, np.ndarray):
            elements = torch.from_numpy(elements)
        if isinstance(coordinates, np.ndarray):
            coordinates = torch.from_numpy(coordinates)

        elements = elements.long().to(self.device)
        coordinates = coordinates.float().to(self.device)

        return self.model((elements, coordinates)).energies.cpu().numpy()
