from abc import ABC

import numpy as np
import torch
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

# from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.test_functions import Hartmann
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

# from botorch.posteriors.
from torch import distributions, tensor
from torch.nn import Dropout, Linear, MSELoss, ReLU, Sequential
from torch.optim import Adam

"""
Initialise the Dataset 
"""
neg_hartmann6 = Hartmann(dim=6, negate=True)

train_x = torch.rand(10, 6)
train_y = neg_hartmann6(train_x).unsqueeze(-1)


"""
Initialise and train the NN
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

"""
Derived class of botorch.Models that implements posterior
"""


class NN_Model(Model):
    def __init__(self, nn):
        super().__init__()
        self.model = nn
        self._num_outputs = 1
        self.nb_samples = 20

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        super().posterior(X, observation_noise, posterior_transform)
        self.model.train()

        with torch.no_grad():
            outputs = torch.hstack([self.model(X) for _ in range(self.nb_samples)])
        mean = torch.mean(outputs, axis=1)
        var = torch.var(outputs, axis=1)

        if len(X.shape) == 2:
            covar = torch.diag(var)
        elif len(X.shape) == 4:
            covar = [torch.diag(var[i][0]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis=0)
            covar = covar.unsqueeze(-1)
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


"""
Create 10k batches of test data points to check whether the MES gives negative values for any of those configurations.
"""
for num in range(10000):
    test_x = torch.rand(10, 6)
    test_x = test_x.unsqueeze(-2)
    with torch.no_grad():
        mes = qMES(test_x)
        mes_arr = mes.detach().numpy()
        verdict = np.all(mes_arr > 0)
    if not verdict:
        print(mes)

"""
Randomly predicted MES vals with negative elements: 
tensor([ 1.3287e-04,  2.7284e-02, -5.9893e-03,  4.2737e-05,  1.0618e-01,
         4.2081e-05,  1.2386e-04,  3.4879e-02,  5.4181e-05,  7.9588e-03])
tensor([ 8.3982e-03, -1.9492e-02,  9.3261e-02,  4.2140e-05,  1.1625e-01,
         4.2021e-05,  6.3142e-03,  8.2910e-04,  4.2707e-05,  1.0034e-04])
tensor([ 6.0743e-02, -6.0380e-02,  1.1793e-01,  4.3750e-05,  1.6773e-01,
         7.5281e-05,  2.9400e-02,  6.6042e-05,  4.2379e-05,  4.2036e-05])
tensor([ 1.4783e-03, -5.4467e-04,  1.3299e-01,  4.9710e-05,  2.4702e-02,
         4.1842e-05,  8.1807e-05,  1.2018e-02,  4.2468e-05,  2.6655e-02])
tensor([ 1.1917e-02, -2.6156e-02,  2.2992e-01,  4.4107e-05,  1.6128e-01,
         4.2021e-05,  4.4733e-05,  4.6474e-04,  4.2200e-05,  4.2727e-02])
tensor([ 3.2218e-02,  4.7796e-03,  2.1847e-01,  4.4882e-05, -1.0960e-01,
         4.1842e-05,  3.1891e-04,  4.2439e-05,  1.7229e-04,  1.8108e-04])
"""


"""
Always input dim to model must be of dim 3. By model I do not mean inside the posterior call, I mean mean wehn we are giving input to qMES
"""
