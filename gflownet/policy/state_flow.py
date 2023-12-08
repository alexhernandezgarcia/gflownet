import torch
from torch import nn

from gflownet.policy.base import ModelBase
from gflownet.utils.common import set_device, set_float_precision


class StateFlow(ModelBase):
    """
    Takes state in the policy format and predicts its flow (a scalar)
    """

    def __init__(self, config, env, device, float_precision, base=None):
        super().__init__(config, env, device, float_precision, base)
        # Output dimension
        self.output_dim = 1

        # Instantiate neural network
        self.instantiate()

    def instantiate(self):
        if self.type == "mlp":
            self.model = self.make_mlp(nn.LeakyReLU()).to(self.device)
            self.is_model = True
        else:
            raise "StateFlow model type not defined"

    def __call__(self, states):
        """
        Returns a tensor of the state flows of the shape (batch_size, )
        """
        return super().__call__(states).squeeze()
