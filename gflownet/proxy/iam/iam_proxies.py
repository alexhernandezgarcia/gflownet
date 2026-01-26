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
        key_gdx=None,
        key_year=None,
        key_region=None,
        SCC=None,
        device="cpu",
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
        SCC : float, optional
            Social cost of carbon #Trillion USD over GtCO2
        device : str or torch.device, optional
            Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
            If None, uses the device of the loaded model (likely cuda)
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
                self.key_gdx = data.keys_df.iloc[0, 2]
            else:
                self.key_gdx = key_gdx
            if key_year is None:
                self.key_year = int(data.keys_df.iloc[0, 3])
            else:
                self.key_year = int(key_year)
                # consistency: eventually convert year index to calendar year
                if self.key_year < 2000:
                    self.key_year = 2010 + 5 * self.key_year

            if key_region is None:
                self.key_region = "europe"  # data.keys_df.iloc[0, 1]
            else:
                assert key_region in [
                    "europe",
                    "mexico",
                    "laca",
                    "brazil",
                    "southafrica",
                    "ssa",
                    "seasia",
                    "oceania",
                    "te",
                    "jpnkor",
                    "india",
                    "mena",
                    "usa",
                    "indonesia",
                    "canada",
                    "china",
                    "sasia",
                ]
                self.key_region = key_region

            if SCC is None:
                if self.key_year > 2050:
                    late_year = self.key_year
                else:
                    late_year = 2075

                index = data.index_map.get((self.key_gdx, late_year, self.key_region))
                SCC_guess = data.variables_df[
                    index, self.variables_names.index("SHADOWPRICE_carbon")
                ]
                self.SCC = (
                    SCC_guess
                    * (
                        self.precomputed_scaling_params["SHADOWPRICE_carbon"]["max"]
                        - self.precomputed_scaling_params["SHADOWPRICE_carbon"]["min"]
                    )
                    + self.precomputed_scaling_params["SHADOWPRICE_carbon"]["min"]
                )
            else:
                self.SCC = (
                    torch.tensor(SCC) if not isinstance(SCC, torch.Tensor) else SCC
                )

            self.SCC = self.SCC.to(self.device)
            self.SCC = torch.clamp(self.SCC, 1e-3, 1)

            context_index = data.index_map.get(
                (self.key_gdx, self.key_year, self.key_region)
            )
            try:
                assert context_index is not None
            except AssertionError:
                print(
                    f"Context index {context_index} not found for (gdx, year, region): {self.key_gdx, self.key_year, self.key_region}"
                )
                exit(1)
            context = data.variables_df[context_index, :]
            context = context.squeeze()

            context = context.to(self.device)

            # Add batch dimension: (features,) -> (1, features)
            self.context = context.unsqueeze(0)

            # Set model to eval mode (required for BatchNorm with batch_size=1)
            self.fairy.eval()

            # precompute amounts for call
            self.tech2idx = {name: i for i, name in enumerate(self.subsidies_names)}
            self.var_CONS = self.variables_names.index("CONSUMPTION")
            self.var_EMI = self.variables_names.index("EMI_total_CO2")

            self.amount_map = {
                "NONE": 0.0,
                "LOW": 0.1,
                "MEDIUM": 0.3,
                "HIGH": 0.75,
            }

            # cache scaling params
            self.cons_min = torch.tensor(self.precomputed_scaling_params["CONSUMPTION"]["min"], device=self.device)
            self.cons_max = torch.tensor(self.precomputed_scaling_params["CONSUMPTION"]["max"], device=self.device)
            self.cons_scale = self.cons_max - self.cons_min

        except PackageNotFoundError:
            print("  💥 `fairy` cannot be initialized.")

    def to(self, device):
        """
        Move the proxy to a different device.

        Parameters
        ----------
        device : str or torch.device
            Target device ('cpu', 'cuda', ...)

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
        self.SCC = self.SCC.to(device)

        return self

    @torch.no_grad()
    def __call__(self, states: List[List[Dict]]) -> TensorType:
        """
        Forward pass of the proxy.

        The proxy, for each state in the batch:
            -initializes an empty vector of subsidies
            -scan the investment dictionaries one by one
            -read the associated tech and amounts, and fill in the subsidies vector
            -compute the projection
            -read projected consumption and emissions
            -compute the reward as consumption - emissions*SCC, in Trillion USD

        Parameters
        ----------
        states : List of List[Dict]
            Each List[Dict] represents a single state. Each Dict is an investment configuration, a list defines a plan.
            List of plans is a batch.
        Returns
        -------
        torch.Tensor
            Proxy energies
        """
        batch_size = len(states)
        contexts = self.context.expand(batch_size, -1)

        plan = torch.zeros(
            batch_size,
            self.fairy.subsidies_dim,
            dtype=torch.float32,
            device=self.device,
        )

        #for i, s in enumerate(states):
        #    for inv in s:
        #        plan[i, self.tech2idx[inv["TECH"]]] = self.amount_map[inv["AMOUNT"]]

        # vector version
        batch_indices = []
        tech_indices = []
        amounts = []
        for i, state in enumerate(states):
            for inv in state:
                batch_indices.append(i)
                tech_indices.append(self.tech2idx[inv["TECH"]])
                amounts.append(self.amount_map[inv["AMOUNT"]])

        plan[batch_indices, tech_indices] = torch.tensor(amounts, device=self.device)

        developments = self.fairy(contexts, plan)

        # variable indices
        c_idx = self.var_CONS
        #e_idx = self.var_EMI

        # scaling params
        # cons_min = self.precomputed_scaling_params["CONSUMPTION"]["min"]
        # cons_max = self.precomputed_scaling_params["CONSUMPTION"]["max"]
        #emi_min = self.precomputed_scaling_params["EMI_total_CO2"]["min"]
        #emi_max = self.precomputed_scaling_params["EMI_total_CO2"]["max"]

        y = developments[:, c_idx]  # CONSUMPTION, or GDP, or something else
        #y = y * (cons_max - cons_min) + cons_min
        y = torch.addcmul(self.cons_min, y, self.cons_scale)

        # Apply budget constraint: zero out consumption where emissions exceed budget
        #emissions = developments[:, e_idx]
        #emissions = emissions * (emi_max - emi_min) + emi_min
        #y = y - self.SCC * emissions

        return y
