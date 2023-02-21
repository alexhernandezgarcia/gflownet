import torch

# from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions import Hartmann, AugmentedHartmann
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam
from torch.nn import Linear
from torch.nn import MSELoss
from torch.nn import Sequential, ReLU, Dropout
from torch import tensor
import numpy as np
from abc import ABC
import gpytorch
from gpytorch.priors.torch_priors import GammaPrior
from botorch.models.utils import check_no_nans
from gpytorch.utils.cholesky import psd_safe_cholesky
from botorch.models.model import Model
import math

CLAMP_LB = 1.0e-8

# neg_hartmann6 = Hartmann(dim=6, negate=True)
train_x = torch.linspace(0, 1, 100)
train_y = torch.stack(
    [
        torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
        torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    ],
    -1,
)

# We will use the simplest form of GP model, exact inference
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
gp = MultitaskGPModel(train_x, train_y, likelihood)


gp.train()
likelihood.train()
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

optimizer = Adam([{"params": gp.parameters()}], lr=0.1)


NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    output = gp(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} ")
    optimizer.step()


from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
    qMultiFidelityLowerBoundMaxValueEntropy,
)

# proxy = myGPModel(gp, train_x, train_y)

from botorch.acquisition.utils import project_to_target_fidelity

target_fidelities = {6: 1.0}


def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)


proxy = gp
qMES = qMultiFidelityLowerBoundMaxValueEntropy(
    proxy, candidate_set=train_x, project=project
)

for num in range(10):
    test_seq = torch.rand(10, 6)
    test_f = fidelities[torch.randint(3, (10, 1))]
    test_x = torch.cat((test_seq, test_f), dim=1)
    test_x = test_x.unsqueeze(-2)
    with torch.no_grad():
        mes = qMES(test_x)
        mes_arr = mes.detach().numpy()
        verdict = np.all(mes_arr > 0)
    if not verdict:
        print("Negative MES", mes)
        # print(mes)
