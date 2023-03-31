"""
This works.
"""

import math
import os
import urllib.request
from math import floor

import gpytorch

# import tqdm
import torch
from botorch.test_functions import Hartmann
from scipy.io import loadmat
from tqdm.notebook import tqdm

"""
Initialise the dataset 
"""
neg_hartmann6 = Hartmann(dim=6, negate=True)
train_x = torch.rand(10, 6)
train_y = neg_hartmann6(train_x)
test_x = torch.rand(10, 6)
test_y = neg_hartmann6(test_x)

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = (
        train_x.cuda(),
        train_y.cuda(),
        test_x.cuda(),
        test_y.cuda(),
    )
data_dim = train_x.size(-1)

"""
Initialise the deep learning module. 
"""


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module("linear1", torch.nn.Linear(data_dim, 1024))
        self.add_module("relu1", torch.nn.ReLU())
        self.add_module("linear2", torch.nn.Linear(1024, 512))
        self.add_module("relu2", torch.nn.ReLU())
        self.add_module("linear3", torch.nn.Linear(512, 256))


feature_extractor = LargeFeatureExtractor()

"""
Initialise the Deep Kernel Modeule which combines the above defined deep learning module with a GP layer
"""


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel()
        )
        self.feature_extractor = feature_extractor
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    def forward(self, x):
        projected_x = self.feature_extractor(x)  # projected_x = (10, 256)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
        mean_x = self.mean_module(projected_x)  # 10
        covar_x = self.covar_module(projected_x)  # 10, 10
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

training_iterations = 10

model.train()
likelihood.train()

"""Optimize both the deep nn and the GP"""
optimizer = torch.optim.Adam(
    [
        {"params": model.feature_extractor.parameters()},
        {"params": model.covar_module.parameters()},
        {"params": model.mean_module.parameters()},
        {"params": model.likelihood.parameters()},
    ],
    lr=0.01,
)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


def train():
    iterator = tqdm(range(training_iterations))
    for i in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
    print(loss.item())


train()


model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.use_toeplitz(
    False
), gpytorch.settings.fast_pred_var():
    preds = model(test_x)

# calculate mse between preds.mean and test_y
print("Test MSE: {}".format(torch.sqrt(torch.mean((preds.mean - test_y) ** 2))))


""""
Acquistion Function
"""

from botorch.models import SingleTaskGP
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal


class myGPModel(SingleTaskGP):
    def __init__(self, gp, trainX, trainY):
        super().__init__(trainX, trainY)
        self.model = gp

    @property
    def num_outputs(self) -> int:
        return super().num_outputs

    @property
    def batch_shape(self):
        return super().batch_shape

    def posterior(
        self, X, output_indices=None, observation_noise=False, posterior_transform=None
    ):
        self.eval()  # make sure model is in eval mode
        X = X.to("cuda")
        mvn = self.model(X)
        posterior = GPyTorchPosterior(mvn=mvn)
        return posterior


from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

proxy = myGPModel(model, train_x, train_y.unsqueeze(-1))
qMES = qMaxValueEntropy(proxy, candidate_set=train_x, num_fantasies=1, use_gumbel=True)

"""
Run it for 10k different batches of test ponts. 
"""
import numpy as np

for num in range(10000):
    test_x = torch.rand(10, 6)
    with torch.no_grad():
        mes = qMES(test_x.to("cuda"))
        mes_arr = mes.cpu().detach().numpy()
        verdict = np.all(mes_arr > 0)
    if not verdict:
        print(mes)
