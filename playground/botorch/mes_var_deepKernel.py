"""
Vanilla variational strategy gives negative acq function values
Grid variational strategy doesn't work, the input dimension to the gp layer ends up being (64, 1) even in the IndependentMultiTask variational case. 
The mean then ends up ebing (1) which leads to an error (ofc) while computing the loss
Tutorial somewhat inspored by: https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html
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
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm


class Data(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


# this is for running the notebook in our testing framework
smoke_test = True
neg_hartmann6 = Hartmann(dim=6, negate=True)

train_x = torch.rand(1000, 6)
train_y = neg_hartmann6(train_x)

# define a testloader similar to how the trainloader is defined
test_x = torch.rand(10, 6)
test_y = neg_hartmann6(test_x)

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = (
        train_x.cuda(),
        train_y.cuda(),
        test_x.cuda(),
        test_y.cuda(),
    )

train_dataset = Data(train_x, train_y)
test_dataset = Data(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

data_dim = train_x.shape[-1]
final_dim_nn = 256


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module("linear1", torch.nn.Linear(data_dim, 1024))
        self.add_module("relu1", torch.nn.ReLU())
        self.add_module("linear2", torch.nn.Linear(1024, 512))
        self.add_module("relu2", torch.nn.ReLU())
        self.add_module("linear3", torch.nn.Linear(512, final_dim_nn))


feature_extractor = LargeFeatureExtractor()


class GPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(
        self, num_dim=256, grid_bounds=(-1.0, 1.0), grid_size=64, inducing_points=None
    ):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKLModel(gpytorch.Module):
    def __init__(
        self, num_dim=256, grid_bounds=(-1.0, 1.0), grid_size=64, inducing_points=None
    ):
        super(DKLModel, self).__init__()
        self.gp_layer = GPRegressionModel(
            num_dim=num_dim,
            grid_bounds=grid_bounds,
            grid_size=grid_size,
            inducing_points=inducing_points,
        )
        self.feature_extractor = LargeFeatureExtractor()
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(
            grid_bounds[0], grid_bounds[1]
        )

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # (batch, 256)
        return self.gp_layer(projected_x)


grid_size = 64
likelihood = gpytorch.likelihoods.GaussianLikelihood()
inducing_points = torch.rand(grid_size, final_dim_nn)
model = DKLModel(grid_size=grid_size, inducing_points=inducing_points)

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

training_iterations = 10

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(
    [
        {"params": model.feature_extractor.parameters()},
        {"params": model.gp_layer.hyperparameters()},
        {"params": model.gp_layer.variational_parameters()},
        {"params": likelihood.parameters()},
    ],
    lr=0.01,
)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.VariationalELBO(
    likelihood, model.gp_layer, num_data=train_x.shape[0]
)


def train(epoch):
    model.train()
    likelihood.train()

    minibatch_iter = tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
    with gpytorch.settings.num_likelihood_samples(64):
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = -mll(output, target)
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())


n_epochs = 1
for epoch in range(1, n_epochs + 1):
    with gpytorch.settings.use_toeplitz(False):
        train(epoch)


model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.use_toeplitz(
    False
), gpytorch.settings.fast_pred_var():
    preds = model(test_x)

# calculate mse between preds.mean and test_y
print("Test RMSE: {}".format(torch.sqrt(torch.mean((preds.mean - test_y) ** 2))))


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
        # X = X.to('cuda')
        mvn = self.model(X)
        posterior = GPyTorchPosterior(mvn=mvn)
        return posterior


from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
)

proxy = myGPModel(model, train_x, train_y.unsqueeze(-1))
qMES = qLowerBoundMaxValueEntropy(proxy, candidate_set=train_x, use_gumbel=True)
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

import numpy as np

for num in range(10000):
    test_x = torch.rand(10, 6)
    with torch.no_grad():
        mes = qMES(test_X)  # to('cuda)
        mes_arr = mes.cpu().detach().numpy()
        verdict = np.all(mes_arr > 0)
        # print(num)
    if not verdict:
        print(mes)
    else:
        print(num)
