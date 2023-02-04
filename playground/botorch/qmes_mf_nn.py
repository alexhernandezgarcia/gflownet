import torch
from botorch.models import SingleTaskGP
from botorch.test_functions import AugmentedHartmann
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam
from torch.nn import Linear
from torch.nn import MSELoss, Module
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
neg_hartmann6 = AugmentedHartmann(negate=True)
fidelities = torch.tensor([0.5, 0.75, 1.0])


train_seq = torch.rand(50, 6)
train_f = fidelities[torch.randint(3, (50,1))]
train_x = torch.cat((train_seq, train_f), dim=1)
train_y = neg_hartmann6(train_x).unsqueeze(-1)


"""
Initialise and train the NN
"""
class MLP(Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(Linear(6, 1024), ReLU(), Dropout(0.5), Linear(1024, 1024), Dropout(0.5), ReLU(), Linear(1024, 1))
    def forward(self, x):
        x = self.preprocess(x)
        output = self.model(x)
        return output
    def preprocess(self, x):
        # write code to tkae the first elment along the last axis of x
        # x = x
        x = x[..., :-1]
        return x

mlp = MLP()
mlp.train()
optimizer = Adam(mlp.parameters())
criterion = MSELoss()

seed = 123
torch.manual_seed(seed)
NUM_EPOCHS = 1

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
            covar = torch.diag(var)
        elif len(X.shape)==4:
            mean = mean.squeeze(-1).unsqueeze(1)
            outputs = outputs.squeeze(-1)
            outputs = outputs.view(X.shape[0], -1, self.nb_samples)
            covar = [torch.cov(outputs[i]) for i in range(X.shape[0])]
            # covar = [torch.diag(var[i].squeeze(-1)) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis = 0)
            covar = covar.unsqueeze(1)
        elif len(X.shape)==3:
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

from botorch.acquisition.utils import project_to_target_fidelity
target_fidelities = {6: 1.0}
def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)


from botorch.acquisition.max_value_entropy_search import qMultiFidelityLowerBoundMaxValueEntropy
qMES = qMultiFidelityLowerBoundMaxValueEntropy(proxy, candidate_set = train_x, project = project, num_mv_samples=32)

for num in range(10000):
    test_seq = torch.rand(10, 6)
    test_f = fidelities[torch.randint(3, (10,1))]
    test_x = torch.cat((test_seq, test_f), dim=1)
    test_x = test_x.unsqueeze(-2)
    with torch.no_grad():
        mes = qMES(test_x)
        mes_arr = mes.detach().numpy()
        verdict = np.all(mes_arr>=0)
    if not verdict:
        print("Negative MES", mes)
