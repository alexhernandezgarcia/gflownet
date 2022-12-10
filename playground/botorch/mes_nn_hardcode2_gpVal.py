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

neg_hartmann6 = Hartmann(dim=6, negate=True)

train_x = torch.rand(3, 6)
train_y = neg_hartmann6(train_x).unsqueeze(-1)

mlp = Sequential(Linear(6, 1024), ReLU(), Dropout(0.5), Linear(1024, 1024), Dropout(0.5), ReLU(), Linear(1024, 1))

NUM_EPOCHS = 10
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
        self.model.train()
        with torch.no_grad():
             outputs = torch.hstack([self.model(X) for _ in range(self.nb_samples)])
        mean_data = torch.mean(outputs, axis=1)
        var_data = torch.var(outputs, axis=1)
        if outputs.ndim == 2:
            covar_bogg = torch.cov(outputs)
            covar_data = torch.diag(var_data)
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
            covar_data = covar_hc
            mean_data = mean_hc

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
            # if outputs.ndim == 2:
                # print("Covariance matrix - F*")
            # elif outputs.ndim == 4 and observation_noise == False:
                # print("Covariance matrix - Y - obs noise False")
            # elif outputs.ndim == 4 and observation_noise == True:
                # print("Covariance matrix - Y - obs noise True")
            # print(covar_data.squeeze())
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

# mes = qMES(test_X)
# print("----")
# print(mes)

for num in range(10000):
    test_x = torch.rand(10, 6)
    test_x = test_x.unsqueeze(-2)
    qMES = qMaxValueEntropy(proxy, candidate_set = train_x, use_gumbel=False)

    with torch.no_grad():
        mes = qMES(test_x)
        mes_arr = mes.detach().numpy()
        verdict = np.all(mes_arr>0)
    if not verdict:
        print(mes)