import pickle

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor

from gflownet.proxy.base import Proxy
from gflownet.utils.common import download_file_if_not_exists


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
