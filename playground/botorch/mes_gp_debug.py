from abc import ABC

import gpytorch
import numpy as np
import torch

# from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions import Hartmann
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior
from torch import tensor
from torch.nn import Dropout, Linear, MSELoss, ReLU, Sequential
from torch.optim import Adam

neg_hartmann6 = Hartmann(dim=6, negate=True)

train_x = torch.rand(10, 6)
train_y = neg_hartmann6(train_x).unsqueeze(-1)


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=train_x.shape[-1],
                batch_shape=torch.Size([]),
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            batch_shape=torch.Size([]),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)  # 101
        covar_x = self.covar_module(x)  # (train+test, train+test)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp = ExactGPModel(train_x, train_y, likelihood)

gp.train()
likelihood.train()

from botorch.models.utils import add_output_dim
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood


class myGPModel(SingleTaskGP):
    def __init__(self, gp, trainX=None, trainY=None):
        super().__init__(trainX, trainY)
        self.model = gp

    @property
    def num_outputs(self) -> int:
        return super().num_outputs

    @property
    def batch_shape(self):
        """
        This is a batch shape from an I/O perspective. For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.

        """
        return super().batch_shape

    def posterior(
        self, X, output_indices=None, observation_noise=False, posterior_transform=None
    ):
        """
        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.
        Returns:
            A `GPyTorchPosterior` object, representing `batch_shape` joint
            distributions over `q` points and the outputs selected by
            `output_indices` each. Includes observation noise if specified.
        """
        self.eval()  # make sure model is in eval mode
        X = self.transform_inputs(X)
        if self._num_outputs > 1:
            X, output_dim_idx = add_output_dim(
                X=X, original_batch_shape=self._input_batch_shape
            )
        mvn = self(X)
        posterior = GPyTorchPosterior(mvn=mvn)
        return posterior


from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

proxy = myGPModel(gp, train_x, train_y)

qMES = qMaxValueEntropy(proxy, candidate_set=train_x, num_fantasies=1, use_gumbel=True)

for num in range(10000):
    test_x = torch.rand(10, 6)
    test_x = test_x.unsqueeze(-2)
    with torch.no_grad():
        mes = qMES(test_x)
        mes_arr = mes.detach().numpy()
        verdict = np.all(mes_arr > 0)
    if not verdict:
        print(mes)
