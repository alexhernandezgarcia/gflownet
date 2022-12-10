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

qMES = qMaxValueEntropy(proxy, candidate_set = train_x, num_fantasies=1, use_gumbel=True)

for num in range(10000):
    test_x = torch.rand(2, 6)
    test_x = test_x.unsqueeze(-2)
    with torch.no_grad():
        mes = qMES(test_x)
        mes_arr = mes.detach().numpy()
        verdict = np.all(mes_arr>0)
    if not verdict:
        print("Negative MES", mes)
        # print(mes)