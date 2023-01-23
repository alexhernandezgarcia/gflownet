import numpy as np
import numpy.typing as npt
import pickle
import torch

from xtb.interface import Calculator, Param, XTBException
from xtb.libxtb import VERBOSITY_MUTED


from gflownet.proxy.base import Proxy
from sklearn.ensemble import RandomForestRegressor


class MoleculeEnergyProxy(Proxy):
    def __init__(self, path_to_model=None):
        super().__init__()
        self.min = -np.log(105)
        if path_to_model is not None:
            with open(path_to_model, 'rb') as inp:
                self.model = pickle.load(inp)
    
    def set_device(self, device):
        self.device = device

    def set_float_precision(self, dtype):
        self.float = dtype

    def __call__(self, states_proxy):
        # output of the model is exp(-energy) / 100
        x = states_proxy % (2*np.pi)
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