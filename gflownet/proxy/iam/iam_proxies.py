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
        key_gdx = None,
        key_year = None,
        key_region = None,
        budget = None,
        SCC = None,
        device='cpu',
        **kwargs,
    ):
        """
        Wrapper class around the fairy proxy.

        Parameters
        ----------
        key_gdx : optional
            GDX key for context selection
        key_year : int, optional
            Year for context selection
        key_region : optional
            Region name for context selection
        budget : float, optional
            Emission budget
        SCC : float, optional
            Social cost of carbon
        device : str or torch.device, optional
            Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
            If None, uses the device of the loaded model
        """
        super().__init__(**kwargs)


        print("Initializing fairy proxy:")
        try:
            fairy, data = initialize_fairy()
            self.fairy = fairy
            self.precomputed_scaling_params = data.precomputed_scaling_params

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

            if device is None:
                # Use the device the model is currently on
                self.device = next(self.fairy.parameters()).device
            else:
                # Convert string to torch.device if needed
                if isinstance(device, str):
                    self.device = torch.device(device)
                else:
                    self.device = device

                # Move model to the specified device
                self.fairy = self.fairy.to(self.device)

            print(f"Using device: {self.device}")

            if key_gdx is None:
                self.key_gdx = data.keys_df.iloc[0,2]
            else:
                self.key_gdx = key_gdx
            if key_year is None:
                self.key_year = int(data.keys_df.iloc[0,3])
            else:
                self.key_year = int(key_year)
                #consistency: convert year index to calendar year
                if self.key_year < 2000:
                    self.key_year = 2010 + 5*self.key_year

            if key_region is None:
                self.key_region = data.keys_df.iloc[0,1]
            else:
                assert key_region in ['europe', 'mexico', 'laca', 'brazil', 'southafrica', 'ssa', 'seasia', 'oceania', 'te', 'jpnkor', 'india', 'mena', 'usa', 'indonesia', 'canada', 'china', 'sasia']
                self.key_region = key_region
            if budget is None:
                emission_t0 = data.variables_df[0,self.variables_names.index("EMI_total_CO2")]
                index_t1 = data.index_map.get((self.key_gdx, self.key_year+5, self.key_region))
                emission_t1 = data.variables_df[index_t1, self.variables_names.index("EMI_total_CO2")]
                budget = emission_t1 - emission_t0

            if isinstance(budget, torch.Tensor):
                self.budget = budget.clone().detach().to(self.device)
            else:
                self.budget = torch.tensor(budget, dtype=torch.float32, device=self.device)

            if SCC is None:
                if self.key_year > 2050:
                    late_year = self.key_year
                else:
                    late_year = 2100

                index = data.index_map.get((self.key_gdx, late_year, self.key_region))
                SCC_guess = data.variables_df[index, self.variables_names.index("SHADOWPRICE_carbon")]
                self.SCC = SCC_guess*(self.precomputed_scaling_params["SHADOWPRICE_carbon"]['max'] - self.precomputed_scaling_params["SHADOWPRICE_carbon"]['min']) + self.precomputed_scaling_params["SHADOWPRICE_carbon"]['min']
            else:
                self.SCC = SCC

            self.SCC = self.SCC.to(self.device)

            context_index = data.index_map.get((self.key_gdx, self.key_year, self.key_region))
            try:
                assert context_index is not None
            except AssertionError:
                print(f"Context index {context_index} not found for {self.key_gdx, self.key_year, self.key_region}")
                exit(1)
            context = data.variables_df[context_index,:]
            context = context.squeeze()

            context = context.to(self.device)

            # Add batch dimension: (features,) -> (1, features)
            self.context = context.unsqueeze(0)

            # Set model to eval mode (required for BatchNorm with batch_size=1)
            self.fairy.eval()

        except PackageNotFoundError:
            print("  💥 `fairy` cannot be initialized.")

    def to(self, device):
        """
        Move the proxy to a different device.

        Parameters
        ----------
        device : str or torch.device
            Target device ('cpu', 'cuda')

        Returns
        -------
        self
            Returns self for chaining
        """
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.fairy = self.fairy.to(device)
        self.context = self.context.to(device)
        self.budget = self.budget.to(device)
        self.SCC = self.SCC.to(device)

        return self

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
        y = y*(self.precomputed_scaling_params["CONSUMPTION"]['max'] - self.precomputed_scaling_params["CONSUMPTION"]['min']) + self.precomputed_scaling_params["CONSUMPTION"]['min']

        # Apply budget constraint: zero out consumption where emissions exceed budget
        emissions = developments[:, self.variables_names.index("EMI_total_CO2")]
        emissions = emissions*(self.precomputed_scaling_params["EMI_total_CO2"]['max'] - self.precomputed_scaling_params["EMI_total_CO2"]['min']) + self.precomputed_scaling_params["EMI_total_CO2"]['min']
        y = y - self.SCC * emissions

        return y

    def get_invested_amount(self, amount: str) -> float:
        #todo - update based on scaling parameters
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
