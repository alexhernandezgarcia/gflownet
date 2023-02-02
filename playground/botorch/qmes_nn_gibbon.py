import torch
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
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior
from torch import distributions
from botorch.models.utils import check_no_nans
from gpytorch.utils.cholesky import psd_safe_cholesky

CLAMP_LB = 1.0e-8

"""
Initialise the Dataset 
"""
neg_hartmann6 = Hartmann(dim=6, negate=True)

train_x = torch.rand(50, 6)
train_y = neg_hartmann6(train_x).unsqueeze(-1)


"""
Initialise and train the NN
"""
mlp = Sequential(Linear(6, 1024), ReLU(), Dropout(0.1), Linear(1024, 1024), Dropout(0.1), ReLU(), Linear(1024, 1))
NUM_EPOCHS = 0

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
        print(
            f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} "
         )
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

    def posterior(self, X, observation_noise = False, posterior_transform = None):
        super().posterior(X, observation_noise, posterior_transform)
        self.model.train()

        with torch.no_grad():
             outputs = torch.hstack([self.model(X) for _ in range(self.nb_samples)])
        mean = torch.mean(outputs, axis=1)
        var = torch.var(outputs, axis=1)

        if len(X.shape)==2:
            # candidate set
            covar = torch.diag(var)
        elif len(X.shape)==4:
            # fidelity
            covar = [torch.diag(var[i][0]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis = 0)
            covar = covar.unsqueeze(-1)
        elif len(X.shape)==3:
            # max samples
            covar = [torch.diag(var[i]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis = 0)
        
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

from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy


class myGIBBON(qLowerBoundMaxValueEntropy):

    def _compute_information_gain(
        self, X, mean_M, variance_M, covar_mM):
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

        """"
        Below is not used because we don't have a pending point"""

        # X_batches = torch.cat(
        #     [X, self.X_pending.unsqueeze(0).repeat(X.shape[0], 1, 1)], 1
        # )
        # # batch_shape x (1 + m) x d
        # # NOTE: This is the blocker for supporting posterior transforms.
        # # We would have to process this MVN, applying whatever operations
        # # are typically applied for the corresponding posterior, then applying
        # # the posterior transform onto the resulting object.
        # V = self.model(X_batches)
        # # Evaluate terms required for A
        # A = V.lazy_covariance_matrix[:, 0, 1:].unsqueeze(1)
        # # batch_shape x 1 x m
        # # Evaluate terms required for B
        # B = self.model.posterior(
        #     self.X_pending,
        #     observation_noise=True,
        #     posterior_transform=self.posterior_transform,
        # ).mvn.covariance_matrix.unsqueeze(0)
        # # 1 x m x m

        # # use determinant of block matrix formula
        # V_determinant = variance_m - inv_quad(B, A.transpose(1, 2)).unsqueeze(1)
        # # batch_shape x 1

        # # Take logs and convert covariances to correlations.
        # r = V_determinant.log() - variance_m.log()
        # r = 0.5 * r.transpose(0, 1)
        # return acq + r


qMES = myGIBBON(proxy, candidate_set = train_x, use_gumbel=True, num_mv_samples=32)

# qMES = qLowerBoundMaxValueEntropy(proxy, candidate_set = train_x, use_gumbel=True)

for num in range(10000):
    test_x = torch.rand(10, 6)
    test_x = test_x.unsqueeze(-2)
    with torch.no_grad():
        mes = qMES(test_x)
        mes_arr = mes.detach().numpy()
        print(mes_arr)
        verdict = np.all(mes_arr>0)
    if not verdict:
        print("Negative MES", mes)