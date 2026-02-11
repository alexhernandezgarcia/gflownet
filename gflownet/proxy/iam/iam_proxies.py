from importlib.metadata import PackageNotFoundError
from typing import List, Optional, Tuple

import torch
from torchtyping import TensorType

from gflownet.envs.iam.full_plan import TECHS
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
        Wrapper class around the FAIRY proxy model.

        The proxy takes investment plans (subsidies for each technology) and predicts
        economic outcomes like consumption. It expects inputs as a tuple of
        (plans_tensor, tech_names) where plans_tensor is (batch, n_techs) and
        tech_names is an ordered list matching the tensor columns.

        On first call, validates that environment tech names match proxy tech names
        and caches a permutation index for efficient reordering.

        Parameters
        ----------
        key_gdx : str, optional
            GDX key for context selection. If None, uses first available.
        key_year : int, optional
            Year for context selection. If None, uses first available.
            Values < 2000 are treated as indices and converted: year = 2010 + 5 * key_year
        key_region : str, optional
            Region name for context selection. If None, defaults to "europe".
            Valid regions: europe, mexico, laca, brazil, southafrica, ssa, seasia,
            oceania, te, jpnkor, india, mena, usa, indonesia, canada, china, sasia
        SCC : float, optional
            Social cost of carbon (Trillion USD / GtCO2). If None, estimated from data.
        device : str or torch.device, optional
            Device to run on ('cpu', 'cuda', 'cuda:0', etc.). Default 'cpu'.
        """
        super().__init__(**kwargs)

        print("Initializing FAIRY proxy:")

        try:
            fairy, data = initialize_fairy()
            self.fairy = fairy
            self.precomputed_scaling_params = data.precomputed_scaling_params

            # Technology and variable names from the model
            self.tech_names_ordered = (
                list(self.fairy.subsidies_names)
                if hasattr(self.fairy.subsidies_names, "__iter__")
                else self.fairy.subsidies_names
            )
            self.n_techs = len(self.tech_names_ordered)

            # Import tech names ordering from Plan
            self.tech_names = ["SUBS_"+TECHS[idx] for idx in range(0, self.n_techs)]


            self.variables_names = (
                list(self.fairy.variables_names)
                if hasattr(self.fairy.variables_names, "__iter__")
                else self.fairy.variables_names
            )

            # Device setup
            if device is None:
                self.device = next(self.fairy.parameters()).device
            else:
                self.device = (
                    torch.device(device) if isinstance(device, str) else device
                )
                self.fairy = self.fairy.to(self.device)

            print(f"  Device: {self.device}")
            print(f"  Technologies: {self.n_techs}")

            # Context selection: GDX key
            if key_gdx is None:
                self.key_gdx = data.keys_df.iloc[0, 2]
            else:
                self.key_gdx = key_gdx

            # Context selection: Year
            if key_year is None:
                self.key_year = int(data.keys_df.iloc[0, 3])
            else:
                self.key_year = int(key_year)
                if self.key_year < 2000:
                    self.key_year = 2010 + 5 * self.key_year

            # Context selection: Region
            if key_region is None:
                self.key_region = "europe"
            else:
                valid_regions = [
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
                assert key_region in valid_regions, f"Invalid region: {key_region}"
                self.key_region = key_region

            print(
                f"  Context: gdx={self.key_gdx}, year={self.key_year}, region={self.key_region}"
            )

            # Social cost of carbon
            if SCC is None:
                late_year = self.key_year if self.key_year > 2050 else 2075
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

            # Load context vector
            context_index = data.index_map.get(
                (self.key_gdx, self.key_year, self.key_region)
            )
            assert context_index is not None, (
                f"Context not found for (gdx, year, region): "
                f"({self.key_gdx}, {self.key_year}, {self.key_region})"
            )

            context = data.variables_df[context_index, :].squeeze().to(self.device)
            self.context = context.unsqueeze(0)  # (1, n_variables)

            # Set model to eval mode (required for BatchNorm with batch_size=1)
            self.fairy.eval()

            # Output variable indices
            self.var_CONS = self.variables_names.index("CONSUMPTION")
            self.var_EMI = self.variables_names.index("EMI_total_CO2")

            # Cached scaling parameters as tensors for efficient rescaling
            self.cons_min = torch.tensor(
                self.precomputed_scaling_params["CONSUMPTION"]["min"],
                device=self.device,
            )
            self.cons_max = torch.tensor(
                self.precomputed_scaling_params["CONSUMPTION"]["max"],
                device=self.device,
            )
            self.cons_scale = self.cons_max - self.cons_min

            self.cons_current = torch.addcmul(
                self.cons_min,
                self.context[:, self.var_CONS],
                self.cons_scale,
            )  # shape (1,)

            # Permutation index for reordering env techs -> proxy techs
            # Computed on first __call__ and cached
            self._permutation_idx: Optional[torch.Tensor] = None

            print("  FAIRY proxy initialized successfully")

        except PackageNotFoundError:
            print("  💥 FAIRY cannot be initialized: package not found")
            raise

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
    def __call__(
        self,
        states: torch.Tensor,
    ) -> TensorType:
        """
        Forward pass of the proxy.

        Parameters
        ----------
        states : Tuple[torch.Tensor, List[str]]
            - plans_tensor: (batch, n_techs) with subsidy values
            - tech_names: ordered list of tech names matching tensor columns

        Returns
        -------
        torch.Tensor
            Proxy energies (consumption values), shape (batch,)
        """
        plans_tensor = states

        # Initialize permutation on first call
        if self._permutation_idx is None:
            self._initialize_permutation(self.tech_names)

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

        # Compute delta
        y = y - self.cons_current

        return y
