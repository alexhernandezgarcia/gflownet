import torch
from gflownet.proxy.iam.scenario_scripts.Scenario_Models import initialize_fairy

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from torchtyping import TensorType

from typing import Dict, Iterable, List, Set, Tuple

from gflownet.proxy.base import Proxy

class FAIRY(Proxy):
    def __init__(
        self,
        **kwargs,
    ):
        """
        Wrapper class around the fairy proxy.
        """
        super().__init__(**kwargs)

        print("Initializing fairy proxy:")
        try:
            fairy, data = initialize_fairy()
            self.fairy = fairy
            self.data = data
        except PackageNotFoundError:
            print("  💥 `fairy` cannot be initialized.")

    @torch.no_grad()
    def __call__(self, states: List[List[Dict]], contexts = None, keys_context = None, budgets = None) -> TensorType["batch"]:
        """
        Forward pass of the proxy.

        The proxy, for each state in the batch:
            -initializes an empty vector of subsidies
            -scan the investment dictionaries one by one
            -read the associated tech and amounts, and fill in the subsidies vector

            -catch a context, if none is selected get a random one
            -compute the projection

            -read the emission budget, if none is selected get the context dependant one
            -compute the reward

        Parameters
        ----------
        states : List of List[Dict]
            Each List[Dict] represents a single state. Each Dict is an investment configuration, a list defines a plan.
            List of plans is a batch.
        contexts: Either List of vectors or none. MAX-SCALED
            If not none, needs to have the same number of elements as batch.
            Contains the context associated with each state. The starting situation for which the investment plan is designed.
        keys_context: List of dictionaries, if present, else none
            if context is present, it is ignored.
            If not none, and context is not present, needs to have the same number of elements as batch.
            Contains the keys associated with each investment plan's context, extracted from scenario data
        budgets: either None, single signed integer or list of signed integers. MAX-SCALED [to-do pass original scale budget]
            If not none, needs to have the same number of elements as batch, or be a single signed integer..
            Contains the emission budget associated with each investment plan.
            if is none, and context is passed via keys, targets actual emission reduction. else random number


        Returns
        -------
        torch.Tensor
            Proxy energies. Shape: ``(batch,)``.
        """
        if contexts is not None:
            assert len(states) == len(contexts)
            #assert all contexts are proper length
            assert all([len(c) == self.fairy.variables_dim for c in contexts])
        elif keys_context is not None:
            assert len(states) == len(keys_context)
            contexts = [self.data.variables_df[self.data.index_map.get((key['gdx'], key['year'], key['region']))] for key in keys_context]
            if budgets is None:
                raise ValueError("Implement budget reading from keys")
            elif isinstance(budgets, float):
                budgets = [budgets] * len(states)
        else:
            contexts = [torch.rand(self.fairy.variables_dim) for _ in range(len(states))]

        contexts = torch.tensor(contexts)
        budgets = torch.tensor(budgets)

        plan = torch.zeros(len(states),self.fairy.subsidies_dim).float()

        for i, s in enumerate(states):
            for inv in s:
                amount = self.get_invested_amount(inv['AMOUNT'])
                tech = self.fairy.subsidies_names.index(inv['TECH'])
                plan[i, tech] = amount

        developments = self.fairy(contexts, plan)

        y = developments[:,self.fairy.variables_names.index("CONSUMPTION")]# CONSUMPTION, or GDP, or something else

        y[developments[:,self.fairy.variables_names.index("EMI_total_CO2")]>budgets] = 0

        return y

    # TODO: review whether rescaling is done as expected
    @torch.no_grad()
    def infer_on_train_set(self):
        """
        Infer on the training set and return the ground-truth and proxy values.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(energy, proxy)`` representing 1/ ground-truth energies and 2/
                proxy inference on the proxy's training set as 1D tensors.
        """
        energy = []
        proxy = []

        for b in self.proxy_loaders["train"]:
            x, e = b
            for k, t in enumerate(x):
                if t.ndim == 1:
                    x[k] = t[:, None]
                if t.ndim > 2:
                    x[k] = t.squeeze()
                assert (
                    x[k].ndim == 2
                ), f"t.ndim = {x[k].ndim} != 2 (t.shape: {x[k].shape})"
            p = self.proxy(torch.cat(x, dim=-1), scale_input=True)
            energy.append(e)
            proxy.append(p)

        energy = torch.cat(energy).cpu()
        proxy = torch.cat(proxy).cpu()

        return energy, proxy

    def get_invested_amount(self, amount: str) -> float:
        if amount == 'NONE':
            return 0.0
        if amount == 'LOW':
            return 0.1
        if amount == 'MEDIUM':
            return 0.5
        if amount == 'HIGH':
            return 1.0
