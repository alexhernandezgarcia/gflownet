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
import gpytorch
from gpytorch.priors.torch_priors import GammaPrior
from botorch.models.utils import check_no_nans
from gpytorch.utils.cholesky import psd_safe_cholesky
from botorch.models.model import Model

CLAMP_LB = 1.0e-8

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
        mean_x = self.mean_module(x) #101
        covar_x = self.covar_module(x) #(train+test, train+test)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp = ExactGPModel(train_x, train_y, likelihood)

gp.train()
likelihood.train()
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

optimizer = Adam([{'params': gp.parameters()}], lr=0.1)


NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    output = gp(train_x)
    loss = - mll(output, gp.train_targets).mean()
    loss.backward()
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} " 
         )
    optimizer.step()

"""
Let's test whether the GP gives same mean and covar for the same test point,"""

# gp.eval()
# likelihood.eval()
# test_x = torch.rand(1, 6)
# output1 = likelihood(gp(test_x))
# output2 =  likelihood(gp(test_x))

"""
Yes it does."""

from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils import add_output_dim
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from gpytorch.utils.broadcasting import _mul_broadcast_shape



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

    def posterior(self, X, output_indices = None, observation_noise= False, posterior_transform= None):
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
        # X = self.transform_inputs(X)
        if self._num_outputs > 1:
            X, output_dim_idx = add_output_dim(
                X=X, original_batch_shape=self._input_batch_shape
            )
        
        mvn = self(X) #self.forward() does not work here.
        # I mean it does not give any error, but it gives the same mean and
        # covariance matrix for all the test points. I don't know why.
        posterior = GPyTorchPosterior(mvn=mvn)
        if np.all(posterior.mean.detach().numpy()<0):
            print(posterior.mean.numpy())
        if np.all(posterior.mvn.covariance_matrix.detach().numpy()<0):
            print(posterior.mvn.covariance_matrix.numpy())
        return posterior

from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
proxy = myGPModel(gp, train_x, train_y)

qMES = qLowerBoundMaxValueEntropy(proxy, candidate_set = train_x, use_gumbel=True)

for num in range(10000):
    test_x = torch.rand(10, 6)
    # NEG
    # test_x = tensor([[[0.2305, 0.8453, 0.3189, 0.2432, 0.5595, 0.6100]]])
    # POS
    # test_x = tensor([[[0.9086, 0.6311, 0.5129, 0.0237, 0.8402, 0.3696]]])
    test_x = test_x.unsqueeze(-2)
    with torch.no_grad():
        mes = qMES(test_x)
        mes_arr = mes.detach().numpy()
        verdict = np.all(mes_arr>0)
    if not verdict:
        print("Negative MES", mes)
        # print(mes)