from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, Iterable, List, Set, Tuple

import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.proxy.iam.scenario_scripts.Scenario_Models import initialize_fairy


class FAIRY(Proxy):
    def __init__(
        self,
        key_gdx,
        key_year,
        key_region,
        budget,
        SCC,
        **kwargs,
    ):
        """
        Wrapper class around the fairy proxy.
        """
        super().__init__(**kwargs)
        self.key_gdx = key_gdx
        self.key_year = key_year
        self.key_region = key_region
        self.budget = torch.tensor(budget)

        self.SCC = SCC

        print("Initializing fairy proxy:")
        try:
            fairy, data = initialize_fairy()
            self.fairy = fairy
            context = data.variables_df[
                data.index_map.get((self.key_gdx, self.key_year, self.key_region))
            ]
            self.context = context.unsqueeze(dim=0)

            # Convert subsidies_names to list if it's a pandas Index
            self.subsidies_names = (
                list(self.fairy.subsidies_names)
                if hasattr(self.fairy.subsidies_names, "__iter__")
                else self.fairy.subsidies_names
            )
            self.variables_names = (
                list(self.fairy.variables_names)
                if hasattr(self.fairy.variables_names, "__iter__")
                else self.fairy.variables_names
            )

            # Get device from the model once
            self.device = next(self.fairy.parameters()).device

            # Set model to eval mode (required for BatchNorm with batch_size=1)
            self.fairy.eval()

        except PackageNotFoundError:
            print("  💥 `fairy` cannot be initialized.")

    @torch.no_grad()
    def __call__(self, states: List[List[Dict]]) -> TensorType["batch"]:
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
        Returns
        -------
        torch.Tensor
            Proxy energies. Shape: ``(batch,)``.
        """

        contexts = self.context.repeat(len(states), 1)
        budgets = self.budget.unsqueeze(0).expand(len(states))

        plan = torch.zeros(
            len(states),
            self.fairy.subsidies_dim,
            dtype=torch.float32,
            device=self.device,
        )

        for i, s in enumerate(states):
            for inv in s:
                amount = self.get_invested_amount(inv["AMOUNT"])
                tech = self.subsidies_names.index(inv["TECH"])
                plan[i, tech] = amount

        developments = self.fairy(contexts, plan)

        y = developments[
            :, self.variables_names.index("CONSUMPTION")
        ]  # CONSUMPTION, or GDP, or something else

        # Apply budget constraint: zero out consumption where emissions exceed budget
        emissions = developments[:, self.variables_names.index("EMI_total_CO2")]
        y = y - self.SCC + emissions

        return y

    def get_invested_amount(self, amount: str) -> float:
        if amount == "NONE":
            return 0.0
        if amount == "LOW":
            return 0.1
        if amount == "MEDIUM":
            return 0.5
        if amount == "HIGH":
            return 1.0
        else:
            raise ValueError("Invalid amount")
