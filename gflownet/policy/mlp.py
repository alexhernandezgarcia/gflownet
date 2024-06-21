from torch import nn
from omegaconf import OmegaConf
from gflownet.policy.base import Policy


class MLPPolicy(Policy):
    def __init__(self, config, env, device, float_precision, base=None):
        super().__init__(
            config=config,
            env=env,
            device=device,
            float_precision=float_precision,
            base=base,
        )
        self.is_model = True

    def make_mlp(self, activation):
        """
        Defines an MLP with no top layer activation
        If share_weight == True,
            baseModel (the model with which weights are to be shared) must be provided
        Args
        ----
        layers_dim : list
            Dimensionality of each layer
        activation : Activation
            Activation function
        """
        if self.shared_weights == True and self.base is not None:
            mlp = nn.Sequential(
                self.base.model[:-1],
                nn.Linear(
                    self.base.model[-1].in_features, self.base.model[-1].out_features
                ),
            )
            return mlp
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
            return mlp
        else:
            raise ValueError(
                "Base Model must be provided when shared_weights is set to True"
            )

    def parse_config(self, config):
        if config is None:
            config = OmegaConf.create()
            config.type = "mlp"
        self.checkpoint = config.get("checkpoint", None)
        self.shared_weights = config.get("shared_weights", False)
        self.n_hid = config.get("n_hid", 128)
        self.n_layers = config.get("n_layers", 2)
        self.tail = config.get("tail", [])
        self.reload_ckpt = config.get("reload_ckpt", False)

    def instantiate(self):
        self.model = self.make_mlp(nn.LeakyReLU()).to(self.device)

    def __call__(self, states):
        return self.model(states)
