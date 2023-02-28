import pickle
from typing import Union

import numpy as np
import numpy.typing as npt
import torch
import torchani
from sklearn.ensemble import RandomForestRegressor

from gflownet.proxy.base import Proxy
from gflownet.utils.common import download_file_if_not_exists


TORCHANI_MODELS = {
    "ANI1x": torchani.models.ANI1x,
    "ANI1ccx": torchani.models.ANI1ccx,
    "ANI2x": torchani.models.ANI2x,
}


class RFMoleculeEnergy(Proxy):
    def __init__(self, path_to_model, url_to_model, **kwargs):
        super().__init__(**kwargs)
        self.min = -np.log(105)
        path_to_model = download_file_if_not_exists(path_to_model, url_to_model)
        if path_to_model is not None:
            with open(path_to_model, "rb") as inp:
                self.model = pickle.load(inp)

    def set_n_dim(self, n_dim):
        # self.n_dim is never used in this env,
        # this is just to make molecule env work with htorus
        self.n_dim = n_dim

    def __call__(self, states_proxy):
        # output of the model is exp(-energy) / 100
        x = states_proxy % (2 * np.pi)
        rewards = -np.log(self.model.predict(x) * 100)
        return torch.tensor(
            rewards,
            dtype=self.float,
            device=self.device,
        )

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        new_obj.model = self.model
        return new_obj


class TorchANIMoleculeEnergy(Proxy):
    def __init__(self, model: str = "ANI2x", use_ensemble: bool = True, **kwargs):
        """
        Parameters
        ----------
        model : str
            The name of the pretrained model to be used for prediction.

        use_ensemble : bool
            Whether to use whole ensemble of the models for prediction or only the first one.
        """
        super().__init__(**kwargs)

        if TORCHANI_MODELS.get(model) is None:
            raise ValueError(
                f'Tried to use model "{model}", '
                f"but only {set(TORCHANI_MODELS.keys())} are available."
            )

        self.model = TORCHANI_MODELS[model](
            periodic_table_index=True, model_index=None if use_ensemble else 0
        ).to(self.device)

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
