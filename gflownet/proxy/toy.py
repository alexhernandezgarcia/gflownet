"""
Reward for the Toy environment.

The class allows for the assignment of proxy values for each state independently.

Arbitrarily, the default values are approximately equal to the number of particles
reaching the final states in Figure 2 of the GFlowNet Foundations paper.
"""

from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat


class ToyScorer(Proxy):
    def __init__(
        self,
        values: dict = {3: 30, 4: 14, 6: 23, 8: 10, 9: 30, 10: 5},
        **kwargs,
    ):
        """
        Parameters
        ----------
        values : dict
            The proxy values for each state. The keys are the indidices of each state
            and the values are the proxy values assigned to them.
        """
        super().__init__(**kwargs)
        self.values_dict = values

    def setup(self, env=None):
        """
        Builds a tensor of scores for each state of the environment.

        Parameters
        ----------
        env : GFlowNetEnv
           An instance of the Toy environment
        """
        if env is None:
            return
        self.scores = tfloat(
            [
                self.values_dict[idx] if idx in self.values_dict else 0
                for idx in env.connections.keys()
            ],
            float_type=self.float,
            device=self.device,
        )

    def __call__(self, states: TensorType["batch", 1]) -> TensorType["batch"]:
        if states.shape[1] != 1:
            raise ValueError(
                "Inputs to the Toy function must be 1-dimensional, "
                f"but inputs with {states.shape[1]} dimensions were passed."
            )
        return self.scores[states.squeeze()]
