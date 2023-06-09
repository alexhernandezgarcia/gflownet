from tblite.interface import Calculator, Structure
import pickle
from typing import Iterable, List, Optional
import ray
import numpy as np
import torch
import torchani
from sklearn.ensemble import RandomForestRegressor
from torch import FloatTensor, LongTensor, Tensor

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
        new_obj.__dict__.update(self.__dict__)
        return new_obj


@ray.remote
def _get_energy(numbers, positions):
    calc = Calculator("GFN2-xTB", numbers, positions * 1.8897259886)
    res = calc.singlepoint()
    return res.get("energy").item()


def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class XTBMoleculeEnergy(Proxy):
    def __init__(self, batch_size=100, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.min = -5
        self.max = 0
        self.conformer = None

    def setup(self, env=None):
        self.conformer = env.conformer

    def _sync_conformer_with_state(self, state: List):
        for idx, ta in enumerate(self.conformer.freely_rotatable_tas):
            self.conformer.set_torsion_angle(ta, state[idx])
        return self.conformer

    def __call__(self, states: List) -> Tensor:
        energies = []

        for batch in _chunks(states, self.batch_size):
            structures = []

            for state in batch:
                conf = self._sync_conformer_with_state(state)
                structures.append(
                    (conf.get_atomic_numbers(), conf.get_atom_positions())
                )

            tasks = [_get_energy.remote(s[0], s[1]) for s in structures]
            energies.extend(ray.get(tasks))

        return torch.tensor(energies, dtype=self.float, device=self.device)


class TorchANIMoleculeEnergy(Proxy):
    def __init__(
        self,
        model: str = "ANI2x",
        use_ensemble: bool = True,
        batch_size: Optional[int] = None,
        divider: float = 100.0,
        **kwargs,
    ):
        """
        Parameters
        ----------
        model : str
            The name of the pretrained model to be used for prediction.

        use_ensemble : bool
            Whether to use whole ensemble of the models for prediction or only the first one.

        batch_size : int
            Batch size for TorchANI. If none, will process all states as a single batch.

        divider : float
            The value by which the output of TorchANI will be divided. Necessary for Boltzmann
            reward function with high betas, for which the values can explode without division.
        """
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.divider = divider
        self.min = -5

        if TORCHANI_MODELS.get(model) is None:
            raise ValueError(
                f'Tried to use model "{model}", '
                f"but only {set(TORCHANI_MODELS.keys())} are available."
            )

        self.model = TORCHANI_MODELS[model](
            periodic_table_index=True, model_index=None if use_ensemble else 0
        ).to(self.device)
        self.conformer = None

    def setup(self, env=None):
        self.conformer = env.conformer

    def _sync_conformer_with_state(self, state: List):
        for idx, ta in enumerate(self.conformer.freely_rotatable_tas):
            self.conformer.set_torsion_angle(ta, state[idx])
        return self.conformer

    @torch.no_grad()
    def __call__(self, states: Iterable) -> Tensor:
        """
        Args
        ----
        states
            An iterable of states in AlanineDipeptide environment format (torsion angles).

        Returns
        ----
        energies : tensor
            Torch tensor with dimensionality (batch_size,), containing energies
            predicted by a TorchANI model (in Hartree).
        """
        elements = []
        coordinates = []

        for st in states:
            conf = self._sync_conformer_with_state(st)

            elements.append(conf.get_atomic_numbers())
            coordinates.append(conf.get_atom_positions())

        elements = LongTensor(np.array(elements)).to(self.device)
        coordinates = FloatTensor(np.array(coordinates)).to(self.device)

        if self.batch_size is not None:
            energies = []
            for elements_batch, coordinates_batch in zip(
                torch.split(elements, self.batch_size),
                torch.split(coordinates, self.batch_size),
            ):
                energies.append(
                    self.model((elements_batch, coordinates_batch)).energies
                )
            energies = torch.cat(energies).float()
        else:
            energies = self.model((elements, coordinates)).energies.float()

        return energies / self.divider

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        new_obj.batch_size = self.batch_size
        new_obj.min = self.min
        new_obj.model = self.model
        new_obj.conformer = self.conformer
        return new_obj
