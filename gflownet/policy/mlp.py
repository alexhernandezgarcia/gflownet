from omegaconf import OmegaConf
from torch import nn

from gflownet.policy.base import Policy


class MLPPolicy(Policy):
    def __init__(self, **kwargs):
        config = self._get_config(kwargs["config"])
        # Shared weights, defaults to False
        self.shared_weights = config.get("shared_weights", False)
        # Reload checkpoint, defaults to False
        self.reload_ckpt = config.get("reload_ckpt", False)
        # MLP features: number of layers, number of hidden units, tail, etc.
        self.n_layers = config.get("n_layers", 2)
        self.n_hid = config.get("n_hid", 128)
        self.tail = config.get("tail", [])
        # Base init
        super().__init__(**kwargs)

    def make_model(self, activation: nn.Module = nn.LeakyReLU()):
        """
        Instantiates an MLP with no top layer activation as the policy model.

        If self.shared_weights is True, the base model with which weights are to be
        shared must be provided.

        Parameters
        ----------
        activation : nn.Module
            Activation function of the MLP layers

        Returns
        -------
        model : torch.tensor or torch.nn.Module
            A torch model containing the MLP.
        is_model : bool
            True because an MLP is a model.
        """

        if self.shared_weights == True and self.base is not None:
            mlp = nn.Sequential(
                self.base.model[:-1],
                nn.Linear(
                    self.base.model[-1].in_features, self.base.model[-1].out_features
                ),
            )
            return mlp, True
        elif self.shared_weights == False:
            layers_dim = (
                [self.state_dim] + [self.n_hid] * self.n_layers + [(self.output_dim)]
            )
            mlp = nn.Sequential(
                *(
                    sum(
                        [
                            [nn.Linear(idim, odim)]
                            + ([activation] if n < len(layers_dim) - 2 else [])
                            for n, (idim, odim) in enumerate(
                                zip(layers_dim, layers_dim[1:])
                            )
                        ],
                        [],
                    )
                    + self.tail
                )
            )
            return mlp.to(self.device), True
        else:
            raise ValueError(
                "Base Model must be provided when shared_weights is set to True"
            )

    def __call__(self, states):
        return self.model(states)
