"""
Climate-economics environment:
Discrete investment options
"""

import random
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from sympy.physics.quantum import state
from torch.distributions import Categorical
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tfloat, tlong

CHOICES = tuple(["SECTOR", "TAG", "TECH", "AMOUNT", "LOCK"])

SECTORS = tuple(
    [
        "POWER",
        "ENERGY",
        "VEHICLES",
        "STORAGE",
        "DAC",
    ]
)

TAGS = tuple(["GREEN", "BROWN", "CCS"])

TECHS = tuple(
    [
        "power_COAL_noccs",
        "power_COAL_ccs",
        "power_NUCLEAR",
        "power_OIL",
        "power_GAS_noccs",
        "power_GAS_ccs",
        "power_HYDRO",
        "power_BIOMASS_noccs",
        "power_BIOMASS_ccs",
        "power_WIND_onshore",
        "power_WIND_offshore",
        "power_SOLAR",
        "thermal_SOLAR",
        "enduse_COAL_ccs",
        "power_STORAGE",
        "production_HYDROGEN",
        "refueling_station_HYDROGEN",
        "pipelines_HYDROGEN",
        "DAC_liquid_sorbents",
        "DAC_solid_sorbents",
        "DAC_calcium_oxide",
        "CARS_trad",
        "CARS_hybrid",
        "CARS_electric",
        "CARS_fuelcell",
        "HEAVYDUTY_trad",
        "HEAVYDUTY_hybrid",
        "HEAVYDUTY_electric",
        "HEAVYDUTY_fuelcell",
    ]
)

AMOUNTS = tuple(
    [
        "HIGH",
        "MEDIUM",
        "LOW",
        "NONE",
    ]
)

ALLOWED_SECTOR2TAGS = {
    "POWER": ["GREEN", "BROWN", "CCS"],
    "ENERGY": ["GREEN", "CCS"],
    "VEHICLES": ["GREEN", "BROWN"],
    "STORAGE": ["GREEN"],
    "DAC": ["CCS"],
}
ALLOWED_TAG2SECTOR = {
    "GREEN": ["POWER", "ENERGY", "VEHICLES", "STORAGE"],
    "BROWN": ["POWER", "VEHICLES"],
    "CCS": ["POWER", "ENERGY", "DAC"],
}
ALLOWED_SECTOR2TECH = {
    "POWER": [
        "power_COAL_noccs",
        "power_COAL_ccs",
        "power_NUCLEAR",
        "power_OIL",
        "power_GAS_noccs",
        "power_GAS_ccs",
        "power_HYDRO",
        "power_BIOMASS_noccs",
        "power_BIOMASS_ccs",
        "power_WIND_onshore",
        "power_WIND_offshore",
        "power_SOLAR",
    ],
    "ENERGY": ["thermal_SOLAR", "enduse_COAL_ccs"],
    "VEHICLES": [
        "CARS_trad",
        "CARS_hybrid",
        "CARS_electric",
        "CARS_fuelcell",
        "HEAVYDUTY_trad",
        "HEAVYDUTY_hybrid",
        "HEAVYDUTY_electric",
        "HEAVYDUTY_fuelcell",
    ],
    "STORAGE": [
        "power_STORAGE",
        "production_HYDROGEN",
        "refueling_station_HYDROGEN",
        "pipelines_HYDROGEN",
    ],
    "DAC": ["DAC_liquid_sorbents", "DAC_solid_sorbents", "DAC_calcium_oxide"],
}
ALLOWED_TAG2TECH = {
    "GREEN": [
        "power_NUCLEAR",
        "power_HYDRO",
        "power_WIND_onshore",
        "power_WIND_offshore",
        "power_SOLAR",
        "thermal_SOLAR",
        "power_STORAGE",
        "production_HYDROGEN",
        "refueling_station_HYDROGEN",
        "pipelines_HYDROGEN",
        "CARS_hybrid",
        "CARS_electric",
        "CARS_fuelcell",
        "HEAVYDUTY_hybrid",
        "HEAVYDUTY_electric",
        "HEAVYDUTY_fuelcell",
    ],
    "BROWN": [
        "power_COAL_noccs",
        "power_OIL",
        "power_GAS_noccs",
        "power_BIOMASS_noccs",
        "CARS_trad",
        "HEAVYDUTY_trad",
    ],
    "CCS": [
        "power_COAL_ccs",
        "power_GAS_ccs",
        "power_BIOMASS_ccs",
        "enduse_COAL_ccs",
        "DAC_liquid_sorbents",
        "DAC_solid_sorbents",
        "DAC_calcium_oxide",
    ],
}


class FullPlan(GFlowNetEnv):

    def __init__(
        self,
        sectors: Iterable = None,
        tags: Iterable = None,
        techs: Iterable = None,
        amounts: Iterable = None,
        **kwargs,
    ):
        # Main attributes
        self.choices = CHOICES

        if sectors is None:
            self.sectors = SECTORS
        else:
            self.sectors = sectors
        self.n_sectors = len(self.sectors)
        if tags is None:
            self.tags = TAGS
        else:
            self.tags = tags
        self.n_tags = len(self.tags)
        if techs is None:
            self.techs = TECHS
        else:
            self.techs = techs
        self.n_techs = len(self.techs)
        if amounts is None:
            self.amounts = AMOUNTS
        else:
            self.amounts = amounts
        self.n_amounts = len(self.amounts)
        # Dictionaries
        self.idx2token_choices = {
            idx + 1: token for idx, token in enumerate(self.choices)
        }
        self.token2idx_choices = {
            token: idx for idx, token in self.idx2token_choices.items()
        }

        self.idx2token_sectors = {
            idx + 1: token for idx, token in enumerate(self.sectors)
        }
        self.token2idx_sectors = {
            token: idx for idx, token in self.idx2token_sectors.items()
        }

        self.idx2token_tags = {idx + 1: token for idx, token in enumerate(self.tags)}
        self.token2idx_tags = {token: idx for idx, token in self.idx2token_tags.items()}

        self.idx2token_techs = {idx + 1: token for idx, token in enumerate(self.techs)}
        self.token2idx_techs = {
            token: idx for idx, token in self.idx2token_techs.items()
        }

        self.idx2token_amounts = {
            idx + 1: token for idx, token in enumerate(self.amounts)
        }
        self.token2idx_amounts = {
            token: idx for idx, token in self.idx2token_amounts.items()
        }
        # Source state: undefined investment
        self.source = {
            "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
            "plan": [0] * self.n_techs,
        }

        self.network_structure = {
            "sector2tag": ALLOWED_SECTOR2TAGS,
            "tag2sector": ALLOWED_TAG2SECTOR,
            "sector2tech": ALLOWED_SECTOR2TECH,
            "tag2tech": ALLOWED_TAG2TECH,
            "tech2tag": {
                tech: tag for tag, techs in ALLOWED_TAG2TECH.items() for tech in techs
            },
            "tech2sector": {
                tech: sector
                for sector, techs in ALLOWED_SECTOR2TECH.items()
                for tech in techs
            },
        }

        self.eos = (-1, -1)

        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs list with all possible actions, including eos.

        Actions are (choice_idx, value_idx) tuples:
        - (1, 1..n_sectors): SECTOR assignment
        - (2, 1..n_tags): TAG assignment
        - (3, 1..n_techs): TECH assignment
        - (4, 1..n_amounts): AMOUNT assignment
        - (5, 1..n_techs): LOCK actions
        - (-1, -1): EOS
        """
        all_actions = (
            # SECTOR actions (choice_idx = 1)
            [
                (self.token2idx_choices["SECTOR"], self.token2idx_sectors[s])
                for s in self.sectors
            ]
            # TAG actions (choice_idx = 2)
            + [
                (self.token2idx_choices["TAG"], self.token2idx_tags[t])
                for t in self.tags
            ]
            # TECH actions (choice_idx = 3)
            + [
                (self.token2idx_choices["TECH"], self.token2idx_techs[t])
                for t in self.techs
            ]
            # AMOUNT actions (choice_idx = 4)
            + [
                (self.token2idx_choices["AMOUNT"], self.token2idx_amounts[a])
                for a in self.amounts
            ]
            # LOCK actions (choice_idx = 5)
            + [
                (self.token2idx_choices["LOCK"], tech_idx)
                for tech_idx in range(1, self.n_techs + 1)
            ]
            # EOS
            + [self.eos]
        )
        return all_actions

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[Dict] = None,
        done: Optional[bool] = None,
    ) -> List[bool]:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.

        Parameters
        ----------
        state : dict
            Input state. If None, self.state is used.
        done : bool
            Whether the trajectory is done. If None, self.done is used.
        Returns
        -------
        A list of boolean values.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        if done:
            return [True for _ in range(self.action_space_dim)]

        partial = state["partial"]
        plan = state["plan"]

        # Initialize mask to all invalid
        mask = [True for _ in range(self.action_space_dim)]

        # Get available techs (not yet in plan)
        available_techs = self._get_available_techs(plan)

        # Check if partial is complete
        if self._partial_is_complete(partial):
            # Only valid action is LOCK for the tech in partial
            tech_idx = partial["TECH"]
            lock_action = (self.token2idx_choices["LOCK"], tech_idx)
            mask[self.action2index(lock_action)] = False
            return mask

        # Partial is not complete - determine valid assignment actions
        assigned = self._get_assigned_in_partial(partial)

        has_sector = "SECTOR" in assigned
        has_tag = "TAG" in assigned
        has_tech = "TECH" in assigned
        has_amount = "AMOUNT" in assigned

        # If TECH is set but SECTOR is not, force SECTOR assignment
        if has_tech and not has_sector:
            forced_sector_token = self.network_structure["tech2sector"][
                self.idx2token_techs[partial["TECH"]]
            ]
            forced_action = (
                self.token2idx_choices["SECTOR"],
                self.token2idx_sectors[forced_sector_token],
            )
            mask[self.action2index(forced_action)] = False
            return mask

        # If TECH is set but TAG is not, force TAG assignment
        if has_tech and not has_tag:
            forced_tag_token = self.network_structure["tech2tag"][
                self.idx2token_techs[partial["TECH"]]
            ]
            forced_action = (
                self.token2idx_choices["TAG"],
                self.token2idx_tags[forced_tag_token],
            )
            mask[self.action2index(forced_action)] = False
            return mask

        # Convert available_techs to token set for filtering
        available_tech_tokens = {self.idx2token_techs[idx] for idx in available_techs}

        # SECTOR actions
        if not has_sector:
            if has_tag:
                # Only sectors compatible with the tag and having available techs
                allowed_sectors = self.network_structure["tag2sector"][
                    self.idx2token_tags[partial["TAG"]]
                ]
                for sector_token in allowed_sectors:
                    # Check if this sector has available techs compatible with tag
                    sector_techs = set(
                        self.network_structure["sector2tech"][sector_token]
                    )
                    tag_techs = set(
                        self.network_structure["tag2tech"][
                            self.idx2token_tags[partial["TAG"]]
                        ]
                    )
                    compatible_available = (
                        sector_techs & tag_techs & available_tech_tokens
                    )
                    if compatible_available:
                        action = (
                            self.token2idx_choices["SECTOR"],
                            self.token2idx_sectors[sector_token],
                        )
                        mask[self.action2index(action)] = False
            else:
                # Any sector that has available techs
                for sector_token in self.sectors:
                    sector_techs = set(
                        self.network_structure["sector2tech"][sector_token]
                    )
                    if sector_techs & available_tech_tokens:
                        action = (
                            self.token2idx_choices["SECTOR"],
                            self.token2idx_sectors[sector_token],
                        )
                        mask[self.action2index(action)] = False

        # TAG actions
        if not has_tag:
            if has_sector:
                # Only tags compatible with the sector and having available techs
                allowed_tags = self.network_structure["sector2tag"][
                    self.idx2token_sectors[partial["SECTOR"]]
                ]
                for tag_token in allowed_tags:
                    # Check if this tag has available techs compatible with sector
                    sector_techs = set(
                        self.network_structure["sector2tech"][
                            self.idx2token_sectors[partial["SECTOR"]]
                        ]
                    )
                    tag_techs = set(self.network_structure["tag2tech"][tag_token])
                    compatible_available = (
                        sector_techs & tag_techs & available_tech_tokens
                    )
                    if compatible_available:
                        action = (
                            self.token2idx_choices["TAG"],
                            self.token2idx_tags[tag_token],
                        )
                        mask[self.action2index(action)] = False
            else:
                # Any tag that has available techs
                for tag_token in self.tags:
                    tag_techs = set(self.network_structure["tag2tech"][tag_token])
                    if tag_techs & available_tech_tokens:
                        action = (
                            self.token2idx_choices["TAG"],
                            self.token2idx_tags[tag_token],
                        )
                        mask[self.action2index(action)] = False

        # TECH actions
        if not has_tech:
            if has_sector and has_tag:
                allowed_techs_sector = self.network_structure["sector2tech"][
                    self.idx2token_sectors[partial["SECTOR"]]
                ]
                allowed_techs_tag = self.network_structure["tag2tech"][
                    self.idx2token_tags[partial["TAG"]]
                ]
                allowed_techs = (
                    set(allowed_techs_sector)
                    & set(allowed_techs_tag)
                    & available_tech_tokens
                )
            elif has_sector:
                allowed_techs = (
                    set(
                        self.network_structure["sector2tech"][
                            self.idx2token_sectors[partial["SECTOR"]]
                        ]
                    )
                    & available_tech_tokens
                )
            elif has_tag:
                allowed_techs = (
                    set(
                        self.network_structure["tag2tech"][
                            self.idx2token_tags[partial["TAG"]]
                        ]
                    )
                    & available_tech_tokens
                )
            else:
                allowed_techs = available_tech_tokens

            for tech_token in allowed_techs:
                action = (
                    self.token2idx_choices["TECH"],
                    self.token2idx_techs[tech_token],
                )
                mask[self.action2index(action)] = False

        # AMOUNT actions (always available if not set)
        if not has_amount and not self._plan_is_complete(plan):
            for amount_idx in range(1, self.n_amounts + 1):
                action = (self.token2idx_choices["AMOUNT"], amount_idx)
                mask[self.action2index(action)] = False

        # EOS only valid when plan complete and partial is empty
        if self._plan_is_complete(plan) and not any(partial[k] != 0 for k in partial):
            mask[self.action2index(self.eos)] = False

        return mask

    def _partial_is_complete(self, partial: Dict) -> bool:
        """Check if partial investment has all fields assigned."""
        return (
            partial["SECTOR"] != 0
            and partial["TAG"] != 0
            and partial["TECH"] != 0
            and partial["AMOUNT"] != 0
        )

    def _plan_is_complete(self, plan: List) -> bool:
        """Check if all technologies have been assigned."""
        return all(a != 0 for a in plan)

    def _get_available_techs(self, plan: List) -> set:
        """Get set of tech indices (1-indexed) not yet assigned in plan."""
        return {i + 1 for i, a in enumerate(plan) if a == 0}

    def _get_assigned_in_partial(self, partial: Dict) -> List[str]:
        """Get list of fields assigned in partial."""
        return [k for k in ["SECTOR", "TAG", "TECH", "AMOUNT"] if partial[k] != 0]

    def get_parents(
        self,
        state: Optional[Dict] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        Parameters
        ----------
        state : dict
            Input state. If None, self.state is used.
        done : bool
            Whether the trajectory is done. If None, self.done is used.
        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in environment format.
        actions : list
            List of actions that lead to state for each parent in parents.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        if done:
            return [state], [self.eos]

        if self.equal(state, self.source):
            return [], []

        parents = []
        actions = []

        partial = state["partial"]
        plan = state["plan"]

        assigned = self._get_assigned_in_partial(partial)

        has_sector = "SECTOR" in assigned
        has_tag = "TAG" in assigned
        has_tech = "TECH" in assigned
        has_amount = "AMOUNT" in assigned

        # Case 1: Partial has assignments - can undo certain assignments
        # Following InvestmentDiscrete's get_parents logic exactly

        for attr in assigned:
            # Skip if this would create an invalid intermediate state
            # (following InvestmentDiscrete logic)
            if attr == "TAG" and has_tech and not has_sector:
                # Can't undo TAG if TECH is set but SECTOR isn't
                # (would leave state where only SECTOR can be set, but TAG was undone)
                continue
            if attr == "AMOUNT" and has_tech and (not has_sector or not has_tag):
                # Can't undo AMOUNT if TECH is set but SECTOR or TAG isn't
                continue

            # Create parent by undoing this assignment
            parent = copy(state)
            parent["partial"][attr] = 0
            parents.append(parent)
            actions.append((self.token2idx_choices[attr], partial[attr]))

        # Case 2: Partial is empty but plan has entries - can undo a LOCK
        if not assigned:
            for tech_plan_idx, amount in enumerate(plan):
                if amount != 0:
                    tech_idx = tech_plan_idx + 1  # 1-indexed
                    tech_token = self.idx2token_techs[tech_idx]

                    # Parent had this tech in partial with full assignment
                    parent = copy(state)
                    parent["plan"][tech_plan_idx] = 0
                    parent["partial"] = {
                        "SECTOR": self.token2idx_sectors[
                            self.network_structure["tech2sector"][tech_token]
                        ],
                        "TAG": self.token2idx_tags[
                            self.network_structure["tech2tag"][tech_token]
                        ],
                        "TECH": tech_idx,
                        "AMOUNT": amount,
                    }
                    parents.append(parent)
                    actions.append((self.token2idx_choices["LOCK"], tech_idx))

        return parents, actions

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> [Dict, Tuple[int], bool]:
        """
        Executes step given an action.

        Parameters
        ----------
        action : tuple
            Action to be executed.
        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : dict
            The sequence after executing the action
        action : tuple
            Action executed
        valid : bool
            False, if the action is not allowed for the current state.
        """
        # Generic pre-step checks
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, False

            # Handle EOS
        if action == self.eos:
            if self._plan_is_complete(self.state["plan"]):
                partial = self.state["partial"]
                if not any(partial[k] != 0 for k in partial):
                    self.n_actions += 1
                    self.done = True
                    return self.state, action, True
            return self.state, action, False

        choice_idx, value_idx = action

        # Handle LOCK action
        if choice_idx == self.token2idx_choices["LOCK"]:
            partial = self.state["partial"]
            if not self._partial_is_complete(partial):
                return self.state, action, False
            if partial["TECH"] != value_idx:
                return self.state, action, False

            # Copy amount to plan
            tech_plan_idx = partial["TECH"] - 1  # 0-indexed for plan
            self.state["plan"][tech_plan_idx] = partial["AMOUNT"]

            # Reset partial
            self.state["partial"] = {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0}
            self.n_actions += 1
            return self.state, action, True

        # Handle assignment actions (SECTOR, TAG, TECH, AMOUNT)
        choice_token = self.idx2token_choices[choice_idx]
        if choice_token in ["SECTOR", "TAG", "TECH", "AMOUNT"]:
            if self.state["partial"][choice_token] != 0:
                return self.state, action, False

            self.state["partial"][choice_token] = value_idx
            self.n_actions += 1
            return self.state, action, True

        return self.state, action, False

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment.

        For each tech: up to 4 assignments (SECTOR, TAG, TECH, AMOUNT) + 1 LOCK = 5
        Total: n_techs * 5 + 1 (EOS)
        """
        return self.n_techs * 6 + 1

    def states2proxy(
        self, states: Union[List[Dict], List[TensorType["max_length"]]]
    ) -> Tuple[TensorType["batch", "n_techs"], List[str]]:
        """
        Prepares a batch of states for the FAIRY proxy.

        Returns a tensor of shape (batch, n_techs) containing amount values,
        along with the ordered list of technology names for validation.

        Parameters
        ----------
        states : list
            A batch of states in environment format.

        Returns
        -------
        Tuple[torch.Tensor, List[str]]
            - plans_tensor: (batch, n_techs) tensor with amount values [0.0, 0.1, 0.3, 0.75]
            - tech_names: ordered list of technology names matching tensor columns
        """
        batch_size = len(states)

        # Amount index to value mapping (index 0 = unassigned, 1-4 = HIGH/MED/LOW/NONE)
        amount_idx_to_value = torch.tensor(
            [0.0, 0.75, 0.3, 0.1, 0.0],  # idx: 0=unset, 1=HIGH, 2=MEDIUM, 3=LOW, 4=NONE
            device=self.device,
            dtype=self.float,
        )

        # Build plans tensor from state plan lists
        plans_indices = torch.zeros(
            batch_size, self.n_techs, dtype=torch.long, device=self.device
        )
        for i, state in enumerate(states):
            plans_indices[i] = torch.tensor(
                state["plan"], dtype=torch.long, device=self.device
            )

        # Convert indices to values
        plans_tensor = amount_idx_to_value[plans_indices]

        # Ordered tech names (matching column order in tensor)
        tech_names = [self.idx2token_techs[idx] for idx in range(1, self.n_techs + 1)]

        return plans_tensor, tech_names

    def states2policy(
        self, states: Union[List[Dict[str, int]], List[TensorType["max_length"]]]
    ) -> torch.Tensor:  # TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model.

        Encoding:
        - Partial: one-hot for SECTOR (n_sectors+1), TAG (n_tags+1),
                   TECH (n_techs+1), AMOUNT (n_amounts+1)
        - Plan: one-hot for each tech's amount (n_techs * (n_amounts+1))
        """
        batch_tensors = []

        for state in states:
            partial = state["partial"]
            plan = state["plan"]

            # One-hot encode partial
            sectors = tlong([partial["SECTOR"]], self.device)
            tags = tlong([partial["TAG"]], self.device)
            techs = tlong([partial["TECH"]], self.device)
            amounts = tlong([partial["AMOUNT"]], self.device)

            onehot_sector = F.one_hot(sectors, num_classes=self.n_sectors + 1).to(
                self.float
            )
            onehot_tag = F.one_hot(tags, num_classes=self.n_tags + 1).to(self.float)
            onehot_tech = F.one_hot(techs, num_classes=self.n_techs + 1).to(self.float)
            onehot_amount = F.one_hot(amounts, num_classes=self.n_amounts + 1).to(
                self.float
            )

            # One-hot encode plan
            plan_tensor = tlong(plan, self.device)
            onehot_plan = F.one_hot(plan_tensor, num_classes=self.n_amounts + 1).to(
                self.float
            )
            onehot_plan_flat = onehot_plan.flatten()

            # Concatenate
            state_tensor = torch.cat(
                [
                    onehot_sector.squeeze(0),
                    onehot_tag.squeeze(0),
                    onehot_tech.squeeze(0),
                    onehot_amount.squeeze(0),
                    onehot_plan_flat,
                ]
            )
            batch_tensors.append(state_tensor)

        return torch.stack(batch_tensors)

    def state2readable(self, state: Dict = None) -> str:
        """
        Converts a state into a human-readable string.
        """
        state = self._get_state(state)
        partial = state["partial"]
        plan = state["plan"]

        # Partial
        if partial["SECTOR"] != 0:
            sector_str = self.idx2token_sectors[partial["SECTOR"]]
        else:
            sector_str = "UNASSIGNED"
        if partial["TAG"] != 0:
            tag_str = self.idx2token_tags[partial["TAG"]]
        else:
            tag_str = "UNASSIGNED"
        if partial["TECH"] != 0:
            tech_str = self.idx2token_techs[partial["TECH"]]
        else:
            tech_str = "UNASSIGNED"
        if partial["AMOUNT"] != 0:
            amount_str = self.idx2token_amounts[partial["AMOUNT"]]
        else:
            amount_str = "UNASSIGNED"

        partial_str = f"PARTIAL: {sector_str} | {tag_str} | {tech_str} | {amount_str}"

        # Plan
        plan_strs = []
        for tech_idx in range(1, self.n_techs + 1):
            tech_token = self.idx2token_techs[tech_idx]
            amount_idx = plan[tech_idx - 1]
            if amount_idx != 0:
                amount_token = self.idx2token_amounts[amount_idx]
            else:
                amount_token = "UNASSIGNED"
            plan_strs.append(f"{tech_token}: {amount_token}")

        plan_str = "PLAN: " + "; ".join(plan_strs)

        return f"{partial_str}\n{plan_str}"

    def readable2state(self, readable: str) -> Dict:
        """
        Converts a human-readable representation of a state into the standard format.
        """
        state = copy(self.source)
        lines = readable.strip().split("\n")

        for line in lines:
            if line.startswith("PARTIAL:"):
                parts = line[8:].strip().split(" | ")
                if parts[0] != "UNASSIGNED":
                    state["partial"]["SECTOR"] = self.token2idx_sectors[parts[0]]
                if parts[1] != "UNASSIGNED":
                    state["partial"]["TAG"] = self.token2idx_tags[parts[1]]
                if parts[2] != "UNASSIGNED":
                    state["partial"]["TECH"] = self.token2idx_techs[parts[2]]
                if parts[3] != "UNASSIGNED":
                    state["partial"]["AMOUNT"] = self.token2idx_amounts[parts[3]]

            elif line.startswith("PLAN:"):
                plan_part = line[5:].strip()
                assignments = plan_part.split("; ")
                for assignment in assignments:
                    if ": " in assignment:
                        tech_token, amount_token = assignment.split(": ")
                        if amount_token != "UNASSIGNED":
                            tech_idx = self.token2idx_techs[tech_token]
                            amount_idx = self.token2idx_amounts[amount_token]
                            state["plan"][tech_idx - 1] = amount_idx

        return state

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[Dict[str, int]]:
        """
        Constructs a batch of n states uniformly sampled in the sample space of the
        environment.

        Parameters
        ----------
        n_states : int
            The number of states to sample.

        seed : int
            Random seed.
        """
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        states = []

        for _ in range(n_states):
            plan = [random.randint(1, self.n_amounts) for _ in range(self.n_techs)]
            state = {
                "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
                "plan": plan,
            }
            states.append(state)

        return states

    def well_defined_plan(self, state: Optional[Dict] = None) -> bool:
        """Check if plan is complete and partial is empty."""
        state = self._get_state(state)
        partial = state["partial"]
        plan = state["plan"]
        partial_empty = not any(partial[k] != 0 for k in partial)
        return self._plan_is_complete(plan) and partial_empty

    def get_plan_as_list(self, state: Optional[Dict] = None) -> List[Dict]:
        """
        Get the investment plan as a list of dictionaries.
        This is the final output format.
        """
        state = self._get_state(state)
        plan = state["plan"]

        result = []
        for tech_idx in range(1, self.n_techs + 1):
            tech_token = self.idx2token_techs[tech_idx]
            amount_idx = plan[tech_idx - 1]

            result.append(
                {
                    "TECH": tech_token,
                    "SECTOR": self.network_structure["tech2sector"][tech_token],
                    "TAG": self.network_structure["tech2tag"][tech_token],
                    "AMOUNT": (
                        self.idx2token_amounts[amount_idx] if amount_idx > 0 else None
                    ),
                    "AMOUNT_IDX": amount_idx,
                }
            )

        return result
