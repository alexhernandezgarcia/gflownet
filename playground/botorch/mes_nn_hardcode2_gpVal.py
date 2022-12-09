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


"""
The input to the mvn is mean of dim 1x20 and covar of dim 1 x 20 x 20
explicitly mentioning the batch size to be 1 as opposed to mes_nn_like_gp wherein the batch size is impliitly assumed to be []

If I try this explicit thing for the posterior of Y as well, then I get the following error:
Expected the output shape to match either the t-batch shape of X, or the `model.batch_shape` in the case of acquisition functions using batch models; but got output with shape torch.Size([1]) for X with shape torch.Size([3, 1, 6]).
(on line mes = qMES(test_X))
"""

neg_hartmann6 = Hartmann(dim=6, negate=True)

train_x = torch.rand(3, 6)
train_y = neg_hartmann6(train_x).unsqueeze(-1)

mlp = Sequential(Linear(6, 1024), ReLU(), Dropout(0.5), Linear(1024, 1024), Dropout(0.5), ReLU(), Linear(1024, 1))

NUM_EPOCHS = 10
NUM_EPOCHS = 1

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



from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior

class NN_Model(Model):
    def __init__(self, nn, hc=True):
        super().__init__()
        self.model = nn
        self._num_outputs = 1
        self.nb_samples = 20
        self.hc = hc

    def posterior(self, X, observation_noise = False, posterior_transform = None):
        super().posterior(X, observation_noise, posterior_transform)
        with torch.no_grad():
             outputs = torch.hstack([self.model(X) for _ in range(self.nb_samples)])
        mean_data = torch.mean(outputs, axis=1)
        var_data = torch.var(outputs, axis=1)
        if outputs.ndim == 2:
            covar_data = torch.cov(outputs)
        elif outputs.ndim == 4:
            covar = [torch.diag(var_data[i][0]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis = 0)
            covar_data = covar.unsqueeze(-1)

        if X.ndim==2:
            mean_hc = tensor([0.2971, 0.1443, 0.1065])
            covar_hc = tensor([
                [ 0.3971,  0.1432,  0.0752], 
                [ 0.1432,  0.4083,  0.1268], 
                [ 0.0752,  0.1268,  0.4519], 
                ])
        elif X.ndim==4:
            #3 x 1
            mean_hc = tensor([[[0.0656]], [[0.1681]], [[0.1414]], [[0.1150]], [[0.0942]], [[0.1507]], [[0.0739]], [[0.0861]], [[0.1082]], [[0.2361]]])
            #3 x 1 x1 x1
            covar_hc = tensor([[[[0.6380]]], [[[0.5570]]], [[[0.5677]]], [[[0.4907]]], [[[0.5122]]], [[[0.5068]]], [[[0.5931]]], [[[0.6051]]], [[[0.5914]]], [[[0.5140]]]])

        if observation_noise == True:
            mean_hc = tensor([[[0.0656]], [[0.1681]], [[0.1414]], [[0.1150]], [[0.0942]], [[0.1507]], [[0.0739]], [[0.0861]], [[0.1082]], [[0.2361]]])
            covar_hc = tensor([[[[0.6380]]], [[[0.5570]]], [[[0.5677]]], [[[0.4907]]], [[[0.5122]]], [[[0.5068]]], [[[0.5931]]], [[[0.6051]]], [[[0.5914]]], [[[0.5140]]]]) 
        mvn_hc = MultivariateNormal(mean_hc, covar_hc)
        mvn_data = MultivariateNormal(mean_data, covar_data)
        posterior_hc = GPyTorchPosterior(mvn_hc)
        posterior_data = GPyTorchPosterior(mvn_data)
        if self.hc:
            return posterior_hc
        else:
            if outputs.ndim == 2:
                print("Covariance matrix - F*")
            elif outputs.ndim == 4 and observation_noise == False:
                print("Covariance matrix - Y - obs noise False")
            elif outputs.ndim == 4 and observation_noise == True:
                print("Covariance matrix - Y - obs noise True")
            print(covar_data.squeeze())
            return posterior_data

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

hc = False
proxy = NN_Model(mlp, hc=hc)

qMES = qMaxValueEntropy(proxy, candidate_set = train_x, num_fantasies=1, use_gumbel=False)
test_X=tensor([[[8.8019e-01, 9.3754e-01, 6.6175e-01, 9.9731e-01, 9.8766e-01,
          1.2758e-01]],

        [[8.3748e-01, 7.6972e-01, 8.5658e-01, 7.8703e-02, 5.1260e-04,
          5.1586e-01]],

        [[8.4287e-01, 5.5077e-01, 7.1622e-01, 3.7767e-02, 8.5246e-01,
          4.8746e-01]],

        [[8.2035e-01, 5.1586e-01, 6.9733e-01, 8.1488e-01, 5.0108e-01,
          7.0884e-01]],

        [[7.2564e-01, 9.0265e-01, 2.0046e-01, 8.2284e-01, 6.1629e-01,
          6.7683e-01]],

        [[3.3603e-01, 9.6553e-01, 1.7575e-01, 2.5865e-01, 3.3263e-01,
          9.7570e-01]],

        [[7.8309e-01, 1.6451e-01, 2.7006e-01, 2.4255e-01, 8.8796e-01,
          9.6889e-01]],

        [[9.6394e-01, 9.9281e-01, 1.8663e-02, 6.1858e-01, 4.5968e-01,
          2.2169e-01]],

        [[8.9832e-01, 5.9954e-01, 9.5246e-01, 8.5052e-01, 6.3701e-01,
          1.9874e-01]],

        [[6.0580e-01, 5.4446e-01, 9.3030e-01, 4.5117e-01, 4.6516e-01,
          1.4582e-01]]])

mes = qMES(test_X)
print("----")
print(mes)
