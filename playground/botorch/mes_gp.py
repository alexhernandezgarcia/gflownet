"""
Tutorial: https://botorch.org/tutorials/max_value_entropy
"""

import torch

# from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions import Hartmann
from botorch.test_functions import Branin
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize
from torch.optim import Adam
from torch.nn import Linear
from torch.nn import MSELoss
from torch.nn import Sequential, ReLU, Dropout
from torch import tensor
import numpy as np
from abc import ABC

train_X = torch.rand(10, 6)
neg_hartmann6 = Hartmann(dim=6, negate=True)
train_Y = neg_hartmann6(train_X).unsqueeze(-1)
model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)


NUM_EPOCHS = 10

model.train()
optimizer = Adam([{'params': model.parameters()}], lr=0.1)


for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    output = model(train_X)
    loss = - mll(output, model.train_targets)
    loss.backward()
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} " 
         )
    optimizer.step()

from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

qMES = qMaxValueEntropy(model, train_X, num_fantasies=1)
test_X=torch.tensor([[[0.8754, 0.9025, 0.5862, 0.1580, 0.3266, 0.7930]],

        [[0.1407, 0.2835, 0.0574, 0.7165, 0.2836, 0.8033]],

        [[0.1043, 0.4672, 0.7695, 0.5995, 0.2715, 0.7897]],

        [[0.6130, 0.8399, 0.3882, 0.2005, 0.5959, 0.5445]],

        [[0.5849, 0.9051, 0.8367, 0.1182, 0.3853, 0.9588]],

        [[0.4114, 0.7935, 0.0299, 0.3348, 0.1985, 0.3097]],

        [[0.0172, 0.8890, 0.6926, 0.1963, 0.3057, 0.2855]],

        [[0.6131, 0.9267, 0.6613, 0.1429, 0.3706, 0.3486]],

        [[0.5914, 0.8657, 0.4393, 0.6715, 0.7866, 0.7446]],

        [[0.6269, 0.9950, 0.0640, 0.4415, 0.1140, 0.6024]]])

for num in range(10000):
    test_x = torch.rand(10, 6)
    test_x = test_x.unsqueeze(-2)
    with torch.no_grad():
        mes = qMES(test_X)
        mes_arr = mes.detach().numpy()
        verdict = np.all(mes_arr>0)
    if not verdict:
        print(mes)
        
