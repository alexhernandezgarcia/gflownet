from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, Iterable, List, Set, Tuple

import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.proxy.iam.scenario_scripts.Scenario_Models import \
    initialize_fairy


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
    def __call__(
        self, states: List[List[Dict]], contexts=None, keys_context=None, budgets=None
    ) -> TensorType["batch"]:
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
            # assert all contexts are proper length
            assert all([len(c) == self.fairy.variables_dim for c in contexts])
        elif keys_context is not None:
            assert len(states) == len(keys_context)
            contexts = [
                self.data.variables_df[
                    self.data.index_map.get((key["gdx"], key["year"], key["region"]))
                ]
                for key in keys_context
            ]
            if budgets is None:
                raise ValueError("Implement budget reading from keys")
            elif isinstance(budgets, float):
                budgets = [budgets] * len(states)
        else:
            contexts = [
                torch.rand(self.fairy.variables_dim, device=self.device)
                for _ in range(len(states))
            ]

        # Stack contexts properly - convert all to tensors first
        contexts_list = []
        for ctx in contexts:
            if isinstance(ctx, torch.Tensor):
                contexts_list.append(ctx.to(self.device))
            else:
                contexts_list.append(
                    torch.tensor(ctx, dtype=torch.float32, device=self.device)
                )
        contexts = torch.stack(contexts_list)

        # Handle budgets: convert to 1D tensor matching batch size
        if isinstance(budgets, (int, float)):
            budgets = torch.tensor([float(budgets)] * len(states), device=self.device)
        elif isinstance(budgets, torch.Tensor):
            budgets = budgets.to(self.device)
        elif isinstance(budgets, list):
            budgets = torch.tensor(budgets, device=self.device)
        else:
            budgets = torch.tensor(budgets, device=self.device)

        # Ensure budgets is 1D
        if budgets.dim() == 0:
            budgets = budgets.unsqueeze(0).expand(len(states))

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
        y[emissions > budgets] = 0

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
