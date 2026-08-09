import torch
from torch import nn

from gflownet.policy.trainablebase import TrainablePolicy


class MLPPolicy(TrainablePolicy):
    """
    A class for MLP policies.

    Attributes
    ----------
    n_layers : int
        The number of hidden layers.
    n_hid : int
        The number of units per hidden layer.
    """

    def __init__(
        self,
        n_layers: int = 2,
        n_hid: int = 128,
        **kwargs,
    ):
        # MLP attributes
        self.n_layers = n_layers
        self.n_hid = n_hid
        # Base init
        super().__init__(**kwargs)

    def make_model(self) -> torch.nn.Module:
        """
        Instantiates an MLP with no top layer activation as the policy model.

        If ``self.shared_weights`` is True, the backbone model with which weights are
        to be shared must be provided.

        Returns
        -------
        model : torch.tensor or torch.nn.Module
            A torch model containing the MLP.
        """

        if self.shared_weights == True and self.base is not None:
            mlp = nn.Sequential(
                self.base.model[:-1],
                nn.Linear(
                    self.base.model[-1].in_features, self.base.model[-1].out_features
                ),
            )
            return mlp.to(self.device)
        elif self.shared_weights == False:
            layers_dim = (
                [self.state_dim] + [self.n_hid] * self.n_layers + [self.output_dim]
            )
            mlp = nn.Sequential(
                *(
                    sum(
                        [
                            [nn.Linear(idim, odim)]
                            + ([self.activation] if n < len(layers_dim) - 2 else [])
                            for n, (idim, odim) in enumerate(
                                zip(layers_dim, layers_dim[1:])
                            )
                        ],
                        [],
                    )
                )
            )
            return mlp.to(self.device)
        else:
            raise ValueError(
                "Base Model must be provided when shared_weights is set to True"
            )
