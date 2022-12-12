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

train_x = torch.rand(10, 6)
train_y = neg_hartmann6(train_x).unsqueeze(-1)


"""
Initialise and train the NN
"""
mlp = Sequential(Linear(6, 1024), ReLU(), Dropout(0.5), Linear(1024, 1024), Dropout(0.5), ReLU(), Linear(1024, 1))
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
            # covar = torch.cov(outputs)
            covar = torch.diag(var)
        elif len(X.shape)==4:
            covar = [torch.diag(var[i][0]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis = 0)
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

 
class myQMES(qMaxValueEntropy):
    def __init__(self,
        model: Model,
        candidate_set,
        force_equality,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        posterior_transform = None,
        use_gumbel: bool = True,
        maximize = True,
        X_pending = None,
        train_inputs = None):
        self.force_equality = force_equality
        super().__init__(model, candidate_set, num_fantasies, num_mv_samples, num_y_samples, posterior_transform, use_gumbel, maximize, X_pending, train_inputs)

    def forward(self, X):
        r"""Compute max-value entropy at the design points `X`.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of MVE values at the given design points `X`.
        """
        # Compute the posterior, posterior mean, variance and std
        posterior = self.model.posterior(
            X.unsqueeze(-3),
            observation_noise=False,
            posterior_transform=self.posterior_transform,
        )
        # batch_shape x num_fantasies x (m) x 1
        mean = self.weight * posterior.mean.squeeze(-1).squeeze(-1)
        variance = posterior.variance.clamp_min(CLAMP_LB).view_as(mean)
        ig = self._compute_information_gain(
            X=X, mean_M=mean, variance_M=variance, covar_mM=variance.unsqueeze(-1), posterior_M= posterior
        )
        return ig.mean(dim=0) 

    def _compute_information_gain(self, X,  mean_M, variance_M, covar_mM, posterior_M):
        r"""Computes the information gain at the design points `X`.

        Approximately computes the information gain at the design points `X`,
        for both MES with noisy observations and multi-fidelity MES with noisy
        observation and trace observations.

        The implementation is inspired from the papers on multi-fidelity MES by
        [Takeno2020mfmves]_. The notation in the comments in this function follows
        the Appendix C of [Takeno2020mfmves]_.

        `num_fantasies = 1` for non-fantasized models.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design point each.
            mean_M: A `batch_shape x num_fantasies x (m)`-dim Tensor of means.
            variance_M: A `batch_shape x num_fantasies x (m)`-dim Tensor of variances.
            covar_mM: A
                `batch_shape x num_fantasies x (m) x (1 + num_trace_observations)`-dim
                Tensor of covariances.

        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of information gains at the
            given design points `X` (`num_fantasies=1` for non-fantasized models).
        """
        # compute the std_m, variance_m with noisy observation
        posterior_m = self.model.posterior(
            X.unsqueeze(-3),
            observation_noise=True,
            posterior_transform=self.posterior_transform,
        )
        if self.force_equality == True:
            posterior_m = posterior_M
        # batch_shape x num_fantasies x (m) x (1 + num_trace_observations)
        mean_m = self.weight * posterior_m.mean.squeeze(-1)
        # mean_M = mean_m
        # batch_shape x num_fantasies x (m) x (1 + num_trace_observations)
        variance_m = posterior_m.mvn.covariance_matrix
        # variance_M = variance_m
        check_no_nans(variance_m)

        # compute mean and std for fM|ym, x, Dt ~ N(u, s^2)
        samples_m = self.weight * self.sampler(posterior_m).squeeze(-1)
        # s_m x batch_shape x num_fantasies x (m) (1 + num_trace_observations)
        L = psd_safe_cholesky(variance_m)
        temp_term = torch.cholesky_solve(covar_mM.unsqueeze(-1), L).transpose(-2, -1)
        # equivalent to torch.matmul(covar_mM.unsqueeze(-2), torch.inverse(variance_m))
        # batch_shape x num_fantasies (m) x 1 x (1 + num_trace_observations)

        mean_pt1 = torch.matmul(temp_term, (samples_m - mean_m).unsqueeze(-1))
        mean_new = mean_pt1.squeeze(-1).squeeze(-1) + mean_M
        # s_m x batch_shape x num_fantasies x (m)
        variance_pt1 = torch.matmul(temp_term, covar_mM.unsqueeze(-1))
        # variance_pt1 = variance_m
        variance_new = variance_M - variance_pt1.squeeze(-1).squeeze(-1)
        # batch_shape x num_fantasies x (m)
        stdv_new = variance_new.clamp_min(CLAMP_LB).sqrt()
        # batch_shape x num_fantasies x (m)

        # define normal distribution to compute cdf and pdf
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        # Compute p(fM <= f* | ym, x, Dt)
        view_shape = torch.Size(
            [
                self.posterior_max_values.shape[0],
                # add 1s to broadcast across the batch_shape of X
                *[1 for _ in range(X.ndim - self.posterior_max_values.ndim)],
                *self.posterior_max_values.shape[1:],
            ]
        )  # s_M x batch_shape x num_fantasies x (m)
        max_vals = self.posterior_max_values.view(view_shape).unsqueeze(1)
        # s_M x 1 x batch_shape x num_fantasies x (m)
        normalized_mvs_new = (max_vals - mean_new) / stdv_new
        # s_M x s_m x batch_shape x num_fantasies x (m)  =
        #   s_M x 1 x batch_shape x num_fantasies x (m)
        #   - s_m x batch_shape x num_fantasies x (m)
        cdf_mvs_new = normal.cdf(normalized_mvs_new).clamp_min(CLAMP_LB)

        # Compute p(fM <= f* | x, Dt)
        stdv_M = variance_M.sqrt()
        normalized_mvs = (max_vals - mean_M) / stdv_M
        # s_M x 1 x batch_shape x num_fantasies  x (m) =
        # s_M x 1 x 1 x num_fantasies x (m) - batch_shape x num_fantasies x (m)
        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        # s_M x 1 x batch_shape x num_fantasies x (m)

        # Compute log(p(ym | x, Dt))
        log_pdf_fm = posterior_m.mvn.log_prob(self.weight * samples_m).unsqueeze(0)
        # 1 x s_m x batch_shape x num_fantasies x (m)

        # H0 = H(ym | x, Dt)
        H0 = posterior_m.mvn.entropy()  # batch_shape x num_fantasies x (m)

        # regression adjusted H1 estimation, H1_hat = H1_bar - beta * (H0_bar - H0)
        # H1 = E_{f*|x, Dt}[H(ym|f*, x, Dt)]
        Z = cdf_mvs_new / cdf_mvs  # s_M x s_m x batch_shape x num_fantasies x (m)
        # s_M x s_m x batch_shape x num_fantasies x (m)
        h1 = -Z * Z.log() - Z * log_pdf_fm
        check_no_nans(h1)
        dim = [0, 1]  # dimension of fm samples, fM samples
        H1_bar = h1.mean(dim=dim)
        h0 = -log_pdf_fm
        H0_bar = h0.mean(dim=dim)
        cov = ((h1 - H1_bar) * (h0 - H0_bar)).mean(dim=dim)
        beta = cov / (h0.var(dim=dim) * h1.var(dim=dim)).sqrt()
        H1_hat = H1_bar - beta * (H0_bar - H0)
        ig = H0 - H1_hat  # batch_shape x num_fantasies x (m)
        if self.posterior_max_values.ndim == 2:
            permute_idcs = [-1, *range(ig.ndim - 1)]
        else:
            permute_idcs = [-2, *range(ig.ndim - 2), -1]
        ig = ig.permute(*permute_idcs)  # num_fantasies x batch_shape x (m)
        verdict = np.all(ig.detach().numpy()>0)
        if not verdict:
            print("Ooops")
        return ig

qMES = myQMES(proxy, candidate_set = train_x, force_equality=True, num_fantasies=1)

"""
Create 10k batches of test data points to check whether the MES gives negative values for any of those configurations.
"""
for num in range(100000):
    test_x = torch.rand(1, 6)
    test_x = test_x.unsqueeze(-2)
    with torch.no_grad():
        mes = qMES(test_x)
        mes_arr = mes.detach().numpy()
        verdict = np.all(mes_arr>0)
    if not verdict:
        print(mes)
