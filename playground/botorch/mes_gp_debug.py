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

"""   def __call__(self, *args):
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            if self.settings.debug.on():
                if not all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                    raise RuntimeError("You must train on the training inputs!")
            res = super().__call__(*inputs)
            return res

        # Prior mode
        # elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
        #     full_inputs = args
        #     full_output = super(ExactGP, self).__call__(*full_inputs)
        #     if settings.debug().on():
        #         if not isinstance(full_output, MultivariateNormal):
        #             raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
        #     return full_output

        # Posterior mode
        else:
            # if True : #settings.debug.on():
                # if all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                #     warnings.warn(
                #         "The input matches the stored training data. Did you forget to call model.train()?",
                #         GPInputWarning,
                #     )

            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                train_output = super().__call__(*train_inputs)

                # Create the prediction strategy for
                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )

            # Concatenate the input to the training input
            full_inputs = []
            batch_shape = train_inputs[0].shape[:-2]
            for train_input, input in zip(train_inputs, inputs):
                # Make sure the batch shapes agree for training/test data
                if batch_shape != train_input.shape[:-2]:
                    batch_shape = _mul_broadcast_shape(batch_shape, train_input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                if batch_shape != input.shape[:-2]:
                    batch_shape = _mul_broadcast_shape(batch_shape, input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                    input = input.expand(*batch_shape, *input.shape[-2:])
                full_inputs.append(torch.cat([train_input, input], dim=-2))

            # Get the joint distribution for training/test data
            full_output = super(myGPModel, self).__call__(*full_inputs)
            # if settings.debug().on():
                # if not isinstance(full_output, MultivariateNormal):
                    # raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

            # Make the prediction
            # with settings._use_eval_tolerance():
            predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(full_mean, full_covar)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)
 """
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
proxy = myGPModel(gp, train_x, train_y)

# qMES = qMaxValueEntropy(proxy, candidate_set = train_x, num_fantasies=1, use_gumbel=True)

class myQMES(qMaxValueEntropy):
    def __init__(self,
        model,
        candidate_set,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        posterior_transform = None,
        use_gumbel: bool = True,
        maximize = True,
        X_pending = None,
        train_inputs = None):
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
            X=X, mean_M=mean, variance_M=variance, covar_mM=variance.unsqueeze(-1)
        )
        return ig.mean(dim=0)

    def _compute_information_gain(self, X,  mean_M, variance_M, covar_mM):
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
        # batch_shape x num_fantasies x (m) x (1 + num_trace_observations)
        mean_m = self.weight * posterior_m.mean.squeeze(-1)
        # batch_shape x num_fantasies x (m) x (1 + num_trace_observations)
        variance_m = posterior_m.mvn.covariance_matrix
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
        if ig<0:
            print(ig)
        if variance_new!=0:
            print(variance_new)
        return ig


qMES = myQMES(proxy, candidate_set = train_x, num_fantasies=1, use_gumbel=True)

for num in range(10000):
    # test_x = torch.rand(2, 6)
    # NEG
    # test_x = tensor([[[0.2305, 0.8453, 0.3189, 0.2432, 0.5595, 0.6100]]])
    # POS
    test_x = tensor([[[0.9086, 0.6311, 0.5129, 0.0237, 0.8402, 0.3696]]])
    test_x = test_x.unsqueeze(-2)
    with torch.no_grad():
        mes = qMES(test_x)
        mes_arr = mes.detach().numpy()
        verdict = np.all(mes_arr>0)
    if not verdict:
        print("Negative MES", mes)
        # print(mes)