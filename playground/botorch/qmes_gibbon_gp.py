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
        mean_x = self.mean_module(x)  # 101
        covar_x = self.covar_module(x)  # (train+test, train+test)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
gp = ExactGPModel(train_x, train_y, likelihood)

gp.train()
likelihood.train()
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

optimizer = Adam([{"params": gp.parameters()}], lr=0.1)


NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    output = gp(train_x)
    loss = -mll(output, gp.train_targets).mean()
    loss.backward()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} ")
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
        # X = self.transform_inputs(X)
        if self._num_outputs > 1:
            X, output_dim_idx = add_output_dim(
                X=X, original_batch_shape=self._input_batch_shape
            )

        mvn = self(X)  # self.forward() does not work here.
        # I mean it does not give any error, but it gives the same mean and
        # covariance matrix for all the test points. I don't know why.
        posterior = GPyTorchPosterior(mvn=mvn)
        if np.all(posterior.mean.detach().numpy() < 0):
            print(posterior.mean.numpy())
        if np.all(posterior.mvn.covariance_matrix.detach().numpy() < 0):
            print(posterior.mvn.covariance_matrix.numpy())
        return posterior

    def get_mvn(self, X):
        return self(X)


from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
)
from gpytorch.functions import inv_quad

proxy = myGPModel(gp, train_x, train_y)


class myGIBBON(qLowerBoundMaxValueEntropy):
    def _compute_information_gain(self, X, mean_M, variance_M, covar_mM):
        posterior_m = self.model.posterior(
            X, observation_noise=True, posterior_transform=self.posterior_transform
        )
        mean_m = self.weight * posterior_m.mean.squeeze(-1)
        # batch_shape x 1
        variance_m = posterior_m.variance.clamp_min(CLAMP_LB).squeeze(-1)
        # batch_shape x 1
        check_no_nans(variance_m)

        # get stdv of noiseless variance
        stdv = variance_M.sqrt()
        # batch_shape x 1

        # define normal distribution to compute cdf and pdf
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        # prepare max value quantities required by GIBBON
        mvs = torch.transpose(self.posterior_max_values, 0, 1)
        # 1 x s_M
        normalized_mvs = (mvs - mean_m) / stdv
        # batch_shape x s_M

        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        pdf_mvs = torch.exp(normal.log_prob(normalized_mvs))
        ratio = pdf_mvs / cdf_mvs
        check_no_nans(ratio)

        # prepare squared correlation between current and target fidelity
        rhos_squared = torch.pow(covar_mM.squeeze(-1), 2) / (variance_m * variance_M)
        # batch_shape x 1
        check_no_nans(rhos_squared)

        # calculate quality contribution to the GIBBON acqusition function
        inner_term = 1 - rhos_squared * ratio * (normalized_mvs + ratio)
        acq = -0.5 * inner_term.clamp_min(CLAMP_LB).log()
        # average over posterior max samples
        acq = acq.mean(dim=1).unsqueeze(0)

        if self.X_pending is None:
            # for q=1, no replusion term required
            return acq

        X_batches = torch.cat(
            [X, self.X_pending.unsqueeze(0).repeat(X.shape[0], 1, 1)], 1
        )
        # batch_shape x (1 + m) x d
        # NOTE: This is the blocker for supporting posterior transforms.
        # We would have to process this MVN, applying whatever operations
        # are typically applied for the corresponding posterior, then applying
        # the posterior transform onto the resulting object.
        V = self.model(X_batches)
        # Evaluate terms required for A
        A = V.lazy_covariance_matrix[:, 0, 1:].unsqueeze(1)
        # batch_shape x 1 x m
        # Evaluate terms required for B
        B = self.model.posterior(
            self.X_pending,
            observation_noise=True,
            posterior_transform=self.posterior_transform,
        ).mvn.covariance_matrix.unsqueeze(0)
        # 1 x m x m

        # use determinant of block matrix formula
        V_determinant = variance_m - inv_quad(B, A.transpose(1, 2)).unsqueeze(1)
        # batch_shape x 1

        # Take logs and convert covariances to correlations.
        r = V_determinant.log() - variance_m.log()
        r = 0.5 * r.transpose(0, 1)
        return acq + r


GIBBON = myGIBBON(proxy, candidate_set=train_x, use_gumbel=True)
qMES = qMaxValueEntropy(proxy, candidate_set=train_x, num_fantasies=1, use_gumbel=True)

test_x = torch.rand(10, 6)
test_x = test_x.unsqueeze(-2)
with torch.no_grad():
    mes = qMES(test_x)
    gibbon = GIBBON(test_x)
    # covar = proxy.get_mvn(test_x).covariance_matrix
    # find determinant of var
    # det = torch.det(covar)
    # find log of determinant
    # log_det = 0.5 * torch.log(det)
    # subtract two arrays mes and log_det
    # mes_gibbon = gibbon - log_det
    print("MES", mes)
    # print("MES-GIBBON", mes_gibbon)
    print("GIBBON", gibbon)

"""
MES-GIBBON and MES values differ a lot for a batch of 10 samples:
Epoch  10/10 - Loss: 0.991 
MES tensor([0.1673, 0.1717, 0.1661, 0.1674, 0.1547, 0.1674, 0.1494, 0.1757, 0.1369,
        0.1817])
MES-GIBBON tensor([0.4329, 0.4313, 0.4157, 0.4237, 0.4578, 0.4375, 0.4566, 0.3970, 0.4750,
        0.3858])
GIBBON tensor([0.1402, 0.1470, 0.1398, 0.1406, 0.1342, 0.1417, 0.1311, 0.1478, 0.1228,
        0.1495])

As well as for a batch of 1 sample:
Epoch  10/10 - Loss: 0.985 
MES tensor([0.1315])
MES-GIBBON tensor([0.3766])
GIBBON tensor([0.0998])
"""
