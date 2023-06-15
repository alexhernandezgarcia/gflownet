from typing import Iterable, List, Optional

import numpy as np
import torch
import torchani
from torch import FloatTensor, LongTensor, Tensor

from gflownet.proxy.base import Proxy

TORCHANI_MODELS = {
    "ANI1x": torchani.models.ANI1x,
    "ANI1ccx": torchani.models.ANI1ccx,
    "ANI2x": torchani.models.ANI2x,
}


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
