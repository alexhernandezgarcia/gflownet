import torch
from torch import nn

from gflownet.utils.common import set_device, set_float_precision
from gflownet.policy.base import ModelBase

class StateFlow(ModelBase):
    """
    Takes state in the policy format and predicts its flow (a scalar)
    """
    def __init__(self, config, env, device, float_precision, base=None):
        super().__init__(config, env.policy_input_dim, device, float_precision, base)
        
        # output dim
        self.output_dim = 1
        
        # Instantiate neural network
        self.instantiate()

    def instantiate(self):
        if self.type == "mlp":
            self.model = self.make_mlp(nn.LeakyReLU()).to(self.device)
            self.is_model = True
        else:
            raise "StateFlow model type not defined"