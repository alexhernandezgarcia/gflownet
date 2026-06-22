"""
Tutorial: https://botorch.org/tutorials/max_value_entropy
"""

from abc import ABC

import numpy as np
import torch

# from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.test_functions import Branin, Hartmann
from botorch.utils.transforms import normalize, standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import tensor
from torch.nn import Dropout, Linear, MSELoss, ReLU, Sequential
from torch.optim import Adam

bounds = torch.tensor(Branin._bounds).T
# train_X = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(10, 2)
# train_Y = Branin(negate=True)(train_X).unsqueeze(-1)

# train_X = normalize(train_X, bounds=bounds)
# train_Y = standardize(train_Y + 0.05 * torch.randn_like(train_Y))

# trainX.shape = (10, 2)
# trainY.shape = (10, 1)

train_X = torch.rand(10, 6)
neg_hartmann6 = Hartmann(dim=6, negate=True)
train_Y = neg_hartmann6(train_X).unsqueeze(-1)
model = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)


NUM_EPOCHS = 10

model.train()
optimizer = Adam([{"params": model.parameters()}], lr=0.1)


for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    output = model(train_X)
    loss = -mll(output, model.train_targets)
    loss.backward()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:>3}/{NUM_EPOCHS} - Loss: {loss.item():>4.3f} ")
    optimizer.step()

from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy

"""
Before Bound Thingy
tensor([[0.8834, 0.1320],
        [0.4236, 0.5213],
        [0.0094, 0.6924],
        [0.1341, 0.2716],
        [0.6784, 0.8944],
        [0.3258, 0.1364],
        [0.7805, 0.6763],
        [0.1010, 0.7995],
        [0.5122, 0.7684],
        [0.6269, 0.8049]])
After Bound Thingy
tensor([[ 8.2503,  1.9803],
        [ 1.3537,  7.8199],
        [-4.8592, 10.3860],
        [-2.9892,  4.0742],
        [ 5.1756, 13.4161],
        [-0.1128,  2.0460],
        [ 6.7082, 10.1445],
        [-3.4848, 11.9918],
        [ 2.6837, 11.5260],
        [ 4.4040, 12.0742]])
"""

qMES = qMaxValueEntropy(model, train_X, num_fantasies=1)
test_X = torch.tensor(
    [
        [[0.8754, 0.9025, 0.5862, 0.1580, 0.3266, 0.7930]],
        [[0.1407, 0.2835, 0.0574, 0.7165, 0.2836, 0.8033]],
        [[0.1043, 0.4672, 0.7695, 0.5995, 0.2715, 0.7897]],
        [[0.6130, 0.8399, 0.3882, 0.2005, 0.5959, 0.5445]],
        [[0.5849, 0.9051, 0.8367, 0.1182, 0.3853, 0.9588]],
        [[0.4114, 0.7935, 0.0299, 0.3348, 0.1985, 0.3097]],
        [[0.0172, 0.8890, 0.6926, 0.1963, 0.3057, 0.2855]],
        [[0.6131, 0.9267, 0.6613, 0.1429, 0.3706, 0.3486]],
        [[0.5914, 0.8657, 0.4393, 0.6715, 0.7866, 0.7446]],
        [[0.6269, 0.9950, 0.0640, 0.4415, 0.1140, 0.6024]],
    ]
)
# with torch.no_grad():
#     mes = qMES(test_X)
# print(mes)

for num in range(10000):
    test_x = torch.rand(10, 6)
    test_x = test_x.unsqueeze(-2)
    with torch.no_grad():
        mes = qMES(test_X)
        mes_arr = mes.detach().numpy()
        verdict = np.all(mes_arr > 0)
    if not verdict:
        print(mes)

# candidate_set = torch.rand(10, 6)
# qMES = qMaxValueEntropy(model, candidate_set, num_fantasies=1)
# with torch.no_grad():
#     mes = qMES(test_X)
# print(mes)

# print("Gen batch Initial Conditions")
# from botorch.optim.initializers import gen_batch_initial_conditions

# Xinit = gen_batch_initial_conditions(
#              qMES, bounds, q=1, num_restarts=1, raw_samples=500
#         )
# with torch.no_grad():
#     mes = qMES(Xinit)
# print(mes)

# from botorch.optim import optimize_acqf
# print("Optimize AF")
# # for q = 1
# candidates, acq_value = optimize_acqf(
#     acq_function=qMES,
#     bounds=bounds,
#     q=1,
#     num_restarts=10,
#     raw_samples=512,
#     return_best_only=False
# )
# print(acq_value)
