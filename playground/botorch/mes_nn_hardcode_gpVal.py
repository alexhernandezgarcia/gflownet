import torch

# from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions import Hartmann
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam
from torch.nn import Linear
from torch.nn import MSELoss
from torch.nn import Sequential, ReLU, Dropout
from torch import tensor
import numpy as np
from abc import ABC


"""
The input to the mvn is mean of dim 1x20 and covar of dim 1 x 20 x 20
explicitly mentioning the batch size to be 1 as opposed to mes_nn_like_gp wherein the batch size is impliitly assumed to be []

If I try this explicit thing for the posterior of Y as well, then I get the following error:
Expected the output shape to match either the t-batch shape of X, or the `model.batch_shape` in the case of acquisition functions using batch models; but got output with shape torch.Size([1]) for X with shape torch.Size([10, 1, 6]).
(on line mes = qMES(test_X))
"""

neg_hartmann6 = Hartmann(dim=6, negate=True)

train_x = torch.rand(10, 6)
train_y = neg_hartmann6(train_x).unsqueeze(-1)

mlp = Sequential(
    Linear(6, 1024),
    ReLU(),
    Dropout(0.5),
    Linear(1024, 1024),
    Dropout(0.5),
    ReLU(),
    Linear(1024, 1),
)

NUM_EPOCHS = 10

mlp.train()
optimizer = Adam(mlp.parameters())
criterion = MSELoss()

seed = 123
torch.manual_seed(seed)

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    output = mlp(train_x)
    loss = criterion(output, train_y)
    loss.backward()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} ")
    optimizer.step()


from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior


class NN_Model(Model):
    def __init__(self, nn):
        super().__init__()
        self.model = nn

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        super().posterior(X, observation_noise, posterior_transform)

        if X.ndim == 2:
            mean = tensor(
                [
                    0.2971,
                    0.1443,
                    0.1065,
                    0.2114,
                    0.0825,
                    0.0843,
                    0.0645,
                    0.1047,
                    0.3363,
                    0.2164,
                ]
            )
            covar = tensor(
                [
                    [
                        0.3971,
                        0.1432,
                        0.0752,
                        0.1190,
                        0.0968,
                        0.0285,
                        0.0539,
                        0.1331,
                        0.1840,
                        0.1136,
                    ],
                    [
                        0.1432,
                        0.4083,
                        0.1268,
                        0.1114,
                        0.1529,
                        0.0644,
                        0.1024,
                        0.0922,
                        0.0565,
                        0.0720,
                    ],
                    [
                        0.0752,
                        0.1268,
                        0.4519,
                        0.0488,
                        0.0111,
                        0.1868,
                        0.0466,
                        0.0032,
                        0.0749,
                        0.0554,
                    ],
                    [
                        0.1190,
                        0.1114,
                        0.0488,
                        0.4502,
                        0.0554,
                        0.1033,
                        0.0423,
                        0.0672,
                        0.0561,
                        0.0927,
                    ],
                    [
                        0.0968,
                        0.1529,
                        0.0111,
                        0.0554,
                        0.3883,
                        0.0018,
                        0.1694,
                        0.2801,
                        0.0347,
                        0.0518,
                    ],
                    [
                        0.0285,
                        0.0644,
                        0.1868,
                        0.1033,
                        0.0018,
                        0.4678,
                        0.0353,
                        -0.0046,
                        0.0198,
                        0.0572,
                    ],
                    [
                        0.0539,
                        0.1024,
                        0.0466,
                        0.0423,
                        0.1694,
                        0.0353,
                        0.4511,
                        0.0652,
                        0.0329,
                        0.1015,
                    ],
                    [
                        0.1331,
                        0.0922,
                        0.0032,
                        0.0672,
                        0.2801,
                        -0.0046,
                        0.0652,
                        0.4125,
                        0.0540,
                        0.0325,
                    ],
                    [
                        0.1840,
                        0.0565,
                        0.0749,
                        0.0561,
                        0.0347,
                        0.0198,
                        0.0329,
                        0.0540,
                        0.4292,
                        0.2044,
                    ],
                    [
                        0.1136,
                        0.0720,
                        0.0554,
                        0.0927,
                        0.0518,
                        0.0572,
                        0.1015,
                        0.0325,
                        0.2044,
                        0.4287,
                    ],
                ]
            )
        elif X.ndim == 4:
            # 10 x 1
            mean = tensor(
                [
                    [[0.0656]],
                    [[0.1681]],
                    [[0.1414]],
                    [[0.1150]],
                    [[0.0942]],
                    [[0.1507]],
                    [[0.0739]],
                    [[0.0861]],
                    [[0.1082]],
                    [[0.2361]],
                ]
            )
            # 10 x 1 x1 x1
            covar = tensor(
                [
                    [[[0.6380]]],
                    [[[0.5570]]],
                    [[[0.5677]]],
                    [[[0.4907]]],
                    [[[0.5122]]],
                    [[[0.5068]]],
                    [[[0.5931]]],
                    [[[0.6051]]],
                    [[[0.5914]]],
                    [[[0.5140]]],
                ]
            )

        if observation_noise == True:
            mean = tensor(
                [
                    [[0.0656]],
                    [[0.1681]],
                    [[0.1414]],
                    [[0.1150]],
                    [[0.0942]],
                    [[0.1507]],
                    [[0.0739]],
                    [[0.0861]],
                    [[0.1082]],
                    [[0.2361]],
                ]
            )
            covar = tensor(
                [
                    [[[0.6380]]],
                    [[[0.5570]]],
                    [[[0.5677]]],
                    [[[0.4907]]],
                    [[[0.5122]]],
                    [[[0.5068]]],
                    [[[0.5931]]],
                    [[[0.6051]]],
                    [[[0.5914]]],
                    [[[0.5140]]],
                ]
            )

        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)
        return posterior

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):
        """
        This is a batch shape from an I/O perspective. For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.

        """
        return torch.Size([])


proxy = NN_Model(mlp)

qMES = qMaxValueEntropy(proxy, candidate_set=train_x, num_fantasies=1, use_gumbel=False)
test_X = tensor(
    [
        [[8.8019e-01, 9.3754e-01, 6.6175e-01, 9.9731e-01, 9.8766e-01, 1.2758e-01]],
        [[8.3748e-01, 7.6972e-01, 8.5658e-01, 7.8703e-02, 5.1260e-04, 5.1586e-01]],
        [[8.4287e-01, 5.5077e-01, 7.1622e-01, 3.7767e-02, 8.5246e-01, 4.8746e-01]],
        [[8.2035e-01, 5.1586e-01, 6.9733e-01, 8.1488e-01, 5.0108e-01, 7.0884e-01]],
        [[7.2564e-01, 9.0265e-01, 2.0046e-01, 8.2284e-01, 6.1629e-01, 6.7683e-01]],
        [[3.3603e-01, 9.6553e-01, 1.7575e-01, 2.5865e-01, 3.3263e-01, 9.7570e-01]],
        [[7.8309e-01, 1.6451e-01, 2.7006e-01, 2.4255e-01, 8.8796e-01, 9.6889e-01]],
        [[9.6394e-01, 9.9281e-01, 1.8663e-02, 6.1858e-01, 4.5968e-01, 2.2169e-01]],
        [[8.9832e-01, 5.9954e-01, 9.5246e-01, 8.5052e-01, 6.3701e-01, 1.9874e-01]],
        [[6.0580e-01, 5.4446e-01, 9.3030e-01, 4.5117e-01, 4.6516e-01, 1.4582e-01]],
    ]
)

mes = qMES(test_X)
print(mes)
