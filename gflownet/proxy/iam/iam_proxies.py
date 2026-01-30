from importlib.metadata import PackageNotFoundError, version
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.proxy.iam.scenario_scripts.Scenario_Models import initialize_fairy


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
            self.cons_min = torch.tensor(
                self.precomputed_scaling_params["CONSUMPTION"]["min"],
                device=self.device,
            )
            self.cons_max = torch.tensor(
                self.precomputed_scaling_params["CONSUMPTION"]["max"],
                device=self.device,
            )
            self.cons_scale = self.cons_max - self.cons_min

            # Permutation index for reordering (computed on first call)
            self._permutation_idx: Optional[torch.Tensor] = None
            self._env_n_techs: Optional[int] = None

        except PackageNotFoundError:
            print("  💥 `fairy` cannot be initialized.")

    def _initialize_permutation(self, tech_names: List[str]) -> None:
        """
        Compute and cache the permutation index for reordering environment
        tech ordering to proxy tech ordering.

        Called once on first forward pass.
        """
        env_tech_set = set(tech_names)
        proxy_tech_set = set(self.tech_names_ordered)

        # Validate tech sets match
        missing_in_proxy = env_tech_set - proxy_tech_set
        missing_in_env = proxy_tech_set - env_tech_set

        if missing_in_proxy:
            raise ValueError(
                f"Technologies in environment but not in proxy: {missing_in_proxy}"
            )
        if missing_in_env:
            raise ValueError(
                f"Technologies in proxy but not in environment: {missing_in_env}"
            )

        # Build permutation: for each proxy position, which env position to read from
        env_name2idx = {name: i for i, name in enumerate(tech_names)}
        permutation = [env_name2idx[tech] for tech in self.tech_names_ordered]

        self._permutation_idx = torch.tensor(
            permutation, dtype=torch.long, device=self.device
        )
        self._env_n_techs = len(tech_names)

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
        self.cons_min = self.cons_min.to(device)
        self.cons_max = self.cons_max.to(device)
        self.cons_scale = self.cons_scale.to(device)

        if self._permutation_idx is not None:
            self._permutation_idx = self._permutation_idx.to(device)

        return self

    @torch.no_grad()
    def __call__(self, states: Tuple[torch.Tensor, List[str]]) -> TensorType:
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
        plans_tensor, tech_names = states

        # Initialize permutation on first call
        if self._permutation_idx is None:
            self._initialize_permutation(tech_names)

        # Move to device and apply permutation
        plans_tensor = plans_tensor.to(self.device)
        plan = plans_tensor[:, self._permutation_idx]

        batch_size = plan.shape[0]
        contexts = self.context.expand(batch_size, -1)

        # Forward pass
        developments = self.fairy(contexts, plan)

        # Extract and rescale consumption
        y = developments[:, self.var_CONS]
        y = torch.addcmul(self.cons_min, y, self.cons_scale)

        return y
