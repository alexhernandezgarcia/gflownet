import sys

import torch

from gflownet.proxy.base import Proxy

from standalone.crystal_discriminator import StandaloneDiscriminator

class MXtalNetD(Proxy):
    def __init__(self, scaling_func='score', **kwargs):
        """
        Wrapper class around the standalone Molecular Crystal stability proxy.
        """
        super().__init__(**kwargs)

        self.rescaling_func = scaling_func
        self.higher_is_better = True

        print("Initializing MXtalNetD proxy:")

        self.model = StandaloneDiscriminator(self.device, self.rescaling_func)

    @torch.no_grad()
    def __call__(self, cell_params, mol_data):
        """
        input cell params and molecule conditioning data
        model automatically builds and scores supercells

        cell params (sg_num, (a,b,c), (alpha,beta,gamma), (xbar,ybar,zbar), (theta,phi,r))
        """
        return self.model(cell_params, mol_data)
