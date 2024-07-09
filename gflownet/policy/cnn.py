import torch
from omegaconf import OmegaConf
from torch import nn

from gflownet.policy.base import Policy


class CNNPolicy(Policy):
    def __init__(self, config, env, device, float_precision, base=None):
        self.env = env
        super().__init__(
            config=config,
            env=env,
            device=device,
            float_precision=float_precision,
            base=base,
        )

    def make_cnn(self):
        """
        Defines an CNN with no top layer activation
        """
        if self.shared_weights and self.base is not None:
            layers = list(self.base.model.children())[:-1]
            last_layer = nn.Linear(
                self.base.model[-1].in_features, self.base.model[-1].out_features
            )

            model = nn.Sequential(*layers, last_layer).to(self.device)
            return model

        current_channels = 1
        conv_module = nn.Sequential()

        if len(self.kernel_sizes) != self.n_layers:
            raise ValueError(
                f"Inconsistent dimensions kernel_sizes != n_layers, {len(self.kernel_sizes)} != {self.n_layers}"
            )

        for i in range(self.n_layers):
            conv_module.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=self.channels[i],
                    kernel_size=tuple(self.kernel_sizes[i]),
                    stride=tuple(self.strides[i]),
                    padding=0,
                    padding_mode="zeros",  # Constant zero padding
                ),
            )
            conv_module.add_module(f"relu_{i}", nn.ReLU())
            current_channels = self.channels[i]

        dummy_input = torch.ones(
            (1, 1, self.env.height, self.env.width)
        )  # (batch_size, channels, height, width)
        try:
            in_channels = conv_module(dummy_input).numel()
            if in_channels >= 500_000:  # TODO: this could better be handled
                raise RuntimeWarning(
                    "Input channels for the dense layer are too big, this will increase number of parameters"
                )
        except RuntimeError as e:
            raise RuntimeError(
                "Failed during convolution operation. Ensure that the kernel sizes and strides are appropriate for the input dimensions."
            ) from e

        model = nn.Sequential(
            conv_module, nn.Flatten(), nn.Linear(in_channels, self.output_dim)
        )
        return model.to(self.device)

    def parse_config(self, config):
        super().parse_config(config)
        if config is None:
            config = OmegaConf.create()
        self.checkpoint = config.get("checkpoint", None)
        self.shared_weights = config.get("shared_weights", False)
        self.reload_ckpt = config.get("reload_ckpt", False)
        self.n_layers = config.get("n_layers", 3)
        self.channels = config.get("channels", [16] * self.n_layers)
        self.kernel_sizes = config.get("kernel_sizes", [(3, 3)] * self.n_layers)
        self.strides = config.get("strides", [(1, 1)] * self.n_layers)

    def instantiate(self):
        self.model = self.make_cnn()
        self.is_model = True

    def __call__(self, states):
        states = states.unsqueeze(1)  # (batch_size, channels, height, width)
        return self.model(states)
