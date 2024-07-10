import torch
from omegaconf import OmegaConf
from torch import nn

from gflownet.policy.base import Policy


class CNNPolicy(Policy):
    def __init__(
        self,
        n_layers: int = 2,
        channels: Union[int, List] = [16, 32],
        kernels: Union[int, List] =  [[3, 3], [2, 2]],
        strides: Union[int, List] = [[1, 1], [1, 1]],
        **kwargs,
    ):
        """
        CNN Policy class for a :class:`GFlowNetAgent`.

        Parameters
        ----------
        n_layers : int
            The number of layers in the CNN architecture.
        channels : int or list
            The number of channels in the convolutional layers or a list of number of
            channels for each layer.
        kernels : int or list
            The kernel size of the convolutions or a list of kernel sizes for each
            layer.
        strides : int or list
            The stride of the convolutions or a list of strides for each layer.
        """
        # CNN features: number of layers, number of channels, kernel sizes, strides
        self.n_layers = n_layers
        if isinstance(channels, int):
            self.channels = [channels] * self.n_layers
        else:
            # TODO: check if valid
            self.channels = channels
        if isinstance(kernels, int):
            self.kernels = [(kernels, kernels)] * self.n_layers
        else:
            # TODO: check if valid
            self.kernels = kernels
        if isinstance(strides, int):
            self.strides = [(stride, stride)] * self.n_layers
        else:
            # TODO: check if valid
            self.strides = strides
        # Environment
        # TODO: rethink whether storing the whole environment is needed
        self.env = env
        # Base init
        super().__init__(**kwargs)

    def make_model(self):
        """
        Instantiates a CNN with no top layer activation.

        Returns
        -------
        model : torch.nn.Module
            A torch model containing the CNN.
        is_model : bool
            True because a CNN is a model.
        """
        if self.shared_weights and self.base is not None:
            layers = list(self.base.model.children())[:-1]
            last_layer = nn.Linear(
                self.base.model[-1].in_features, self.base.model[-1].out_features
            )

            model = nn.Sequential(*layers, last_layer).to(self.device)
            return model, True

        current_channels = 1
        conv_module = nn.Sequential()

        if len(self.kernels) != self.n_layers:
            raise ValueError(
                f"Inconsistent dimensions kernels != n_layers, "
                "{len(self.kernels)} != {self.n_layers}"
            )

        for i in range(self.n_layers):
            conv_module.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=self.channels[i],
                    kernel_size=tuple(self.kernels[i]),
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
                    "Input channels for the dense layer are too big, this will "
                    "increase number of parameters"
                )
        except RuntimeError as e:
            raise RuntimeError(
                "Failed during convolution operation. Ensure that the kernel sizes "
                "and strides are appropriate for the input dimensions."
            ) from e

        model = nn.Sequential(
            conv_module, nn.Flatten(), nn.Linear(in_channels, self.output_dim)
        )
        return model.to(self.device), True

    def __call__(self, states):
        states = states.unsqueeze(1)  # (batch_size, channels, height, width)
        return self.model(states)
