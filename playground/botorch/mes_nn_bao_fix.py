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


neg_hartmann6 = Hartmann(dim=6, negate=True)

train_x = torch.rand(10, 6)
train_y = neg_hartmann6(train_x).unsqueeze(-1)

"""
If we are doing DropoutRegressor technique, then it must have a dropout layer
"""
mlp = Sequential(
    Linear(6, 1024),
    ReLU(),
    Dropout(0.5),
    Linear(1024, 1024),
    Dropout(0.5),
    ReLU(),
    Linear(1024, 1),
)

"""
For a very sparse network, with hidden_dim = 8, the expected improvement is very less even though the final loss of the mlp is similar.
EI dec as MLP dim dec
"""

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

# from botorch.posteriors.
from torch.distributions import Normal


class NN_Model(Model):
    def __init__(self, nn):
        super().__init__()
        self.model = nn
        self._num_outputs = 1
        self.nb_samples = 20
        """
        train_inputs: A `n_train x d` Tensor that the model has been fitted on.
                Not required if the model is an instance of a GPyTorch ExactGP model.
        """
        # self.train_inputs = train_x

    # def _shaped_noise_covar(self, base_shape: torch.Size):
    #     return self.noise_covar(*params, shape=base_shape, **kwargs)

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        super().posterior(X, observation_noise, posterior_transform)
        dim_input = X.dim()
        self.model.train()
        with torch.no_grad():
            outputs = torch.hstack([self.model(X) for _ in range(self.nb_samples)])
        mean = torch.mean(outputs, axis=1)
        var = torch.var(outputs, axis=1)
        covar = [torch.diag(var[i]) for i in range(X.shape[0])]
        covar = torch.stack(covar, axis=0)
        if dim_input == 3:
            mean = mean.unsqueeze(1)
            covar = covar.unsqueeze(1)

        if dim_input == 4:
            mean = mean.unsqueeze(1).squeeze(-1)
            covar = covar.unsqueeze(1).unsqueeze(1)
        mvn = MultivariateNormal(mean=mean, covariance_matrix=covar)
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

qMES = qMaxValueEntropy(proxy, candidate_set=train_x.unsqueeze(-2), num_fantasies=1)
test_X = torch.tensor(
    [
        [[0.8754, 0.9025, 0.5862, 0.1580, 0.3266, 0.7930]],
        [[0.1407, 0.2835, 0.0574, 0.7165, 0.2836, 0.8033]],
        [[0.1043, 0.4672, 0.7695, 0.5995, 0.2715, 0.7897]],
        [[0.6130, 0.8399, 0.3882, 0.2005, 0.5959, 0.5445]],
        [[0.5849, 0.9051, 0.8367, 0.1182, 0.3853, 0.9588]],
        [[0.4114, 0.7935, 0.0299, 0.3348, 0.1985, 0.3097]],
        [[0.0172, 0.8890, 0.6926, 0.1963, 0.3057, 0.2855]],
        [[0.6131, 0.9267, 0.6613, 0.1429, 0.3706, 0.3486]],
        [[0.5914, 0.8657, 0.4393, 0.6715, 0.7866, 0.7446]],
        [[0.6269, 0.9950, 0.0640, 0.4415, 0.1140, 0.6024]],
    ]
)

# because in the forward call, qMES adds a dimension but it also
# does not accept textX in shape b x 1 xd
with torch.no_grad():
    mes = qMES(test_X)
print(mes)


"""
Always input dim to model must be of dim 3. By model I do not mean inside the posterior call, I mean mean wehn we are giving input to qMES
"""
