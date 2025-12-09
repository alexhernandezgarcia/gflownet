"""
Climate-economics environment:
Discrete investment options
"""

import random
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from sympy.logic.boolalg import Boolean
from sympy.physics.quantum import state
from torch.distributions import Categorical
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tlong

CHOICES = tuple(
    [
        "SECTOR",
        "TAG",
        "TECH",
        "AMOUNT",
    ]
)

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


class InvestmentDiscrete(GFlowNetEnv):

    def __init__(
        self,
        sectors: Iterable = None,
        tags: Iterable = None,
        techs: Iterable = None,
        amounts: Iterable = None,
        techs_available: Iterable = None,
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
            "SECTOR": 0,
            "TAG": 0,
            "TECH": 0,
            "AMOUNT": 0,
        }
        # Available technologies
        if techs_available is None:
            self.techs_available = tuple(range(1, self.n_techs + 1))
        else:
            if isinstance(techs_available[0], str):
                self.techs_available = tuple(
                    [self.token2idx_techs[tech] for tech in techs_available]
                )
            elif isinstance(techs_available[0], int):
                self.techs_available = tuple(techs_available)

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

        self.filled_on_set = torch.zeros(self.n_techs - 1, 3)

        # pre-computed inverse mappings for masking
        self.tech2sector_idx = {
            self.token2idx_techs[t]: self.token2idx_sectors[s]
            for t, s in self.network_structure["tech2sector"].items()
        }

        self.tech2tag_idx = {
            self.token2idx_techs[t]: self.token2idx_tags[g]
            for t, g in self.network_structure["tech2tag"].items()
        }

        self.sector2tech_idx = {
            self.token2idx_sectors[s]: {self.token2idx_techs[t] for t in techs}
            for s, techs in self.network_structure["sector2tech"].items()
        }

        self.tag2tech_idx = {
            self.token2idx_tags[g]: {self.token2idx_techs[t] for t in techs}
            for g, techs in self.network_structure["tag2tech"].items()
        }

        # Base class init
        super().__init__(**kwargs)

        self.mask_buffer = [True] * len(self.get_action_space())

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs list with all possible actions, including eos.

        An action is represented by a two-element tuple: the first element is the index
        of the element of the dictionary (state) to be changed; the second element is
        the value used to update the state.

        For example, action (3, 2) sets for choice 3 (technology) the value at index 2
        (coal power with CCS), starting from index 1.
        """
        all_actions = (
            [
                (self.token2idx_choices["SECTOR"], self.token2idx_sectors[token_S])
                for token_S in self.sectors
            ]
            + [
                (self.token2idx_choices["TAG"], self.token2idx_tags[token_t])
                for token_t in self.tags
            ]
            + [
                (self.token2idx_choices["TECH"], self.token2idx_techs[token_T])
                for token_T in self.techs
            ]
            + [
                (self.token2idx_choices["AMOUNT"], self.token2idx_amounts[token_A])
                for token_A in self.amounts
            ]
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

        # Initialize mask to all invalid
        mask = self.mask_buffer.copy()

        if done:
            return mask

        assigned = self.get_assigned_attributes(state)
        if self.well_defined_investment(state):  # if you have full investment
            mask[self.action2index(self.eos)] = False
            return mask

        if "TECH" in assigned and not "SECTOR" in assigned:
            forced_sector_idx = self.tech2sector_idx[state["TECH"]]
            forced_action = (
                self.token2idx_choices["SECTOR"],
                forced_sector_idx,
            )
            mask[self.action2index(forced_action)] = False
            return mask

        if "TECH" in assigned and not "TAG" in assigned:
            forced_tag_idx = self.tech2tag_idx[state["TECH"]]
            forced_action = (
                self.token2idx_choices["TAG"],
                forced_tag_idx,
            )
            mask[self.action2index(forced_action)] = False
            return mask

        techs_available_tokens = [self.idx2token_techs[t] for t in self.techs_available]

        if "SECTOR" not in assigned:
            if "TAG" in assigned:
                allowed_sectors = self.network_structure["tag2sector"][
                    self.idx2token_tags[state["TAG"]]
                ]
                for a in allowed_sectors:
                    allowed_techs_sector = self.network_structure["sector2tech"][a]
                    allowed_techs_tag = self.network_structure["tag2tech"][
                        self.idx2token_tags[state["TAG"]]
                    ]
                    available_techs = list(
                        set(allowed_techs_sector)
                        & set(allowed_techs_tag)
                        & set(techs_available_tokens)
                    )
                    if available_techs:
                        mask[
                            self.action2index(
                                (
                                    self.token2idx_choices["SECTOR"],
                                    self.token2idx_sectors[a],
                                )
                            )
                        ] = False
            else:
                for b in range(self.n_sectors):
                    mask[
                        self.action2index((self.token2idx_choices["SECTOR"], b + 1))
                    ] = False

        if "TAG" not in assigned:
            if "SECTOR" in assigned:
                allowed_tags = self.network_structure["sector2tag"][
                    self.idx2token_sectors[state["SECTOR"]]
                ]
                for a in allowed_tags:
                    allowed_techs_tag = self.network_structure["tag2tech"][a]
                    allowed_techs_sector = self.network_structure["sector2tech"][
                        self.idx2token_sectors[state["SECTOR"]]
                    ]
                    available_techs = list(
                        set(allowed_techs_sector)
                        & set(allowed_techs_tag)
                        & set(techs_available_tokens)
                    )
                    if available_techs:
                        mask[
                            self.action2index(
                                (self.token2idx_choices["TAG"], self.token2idx_tags[a])
                            )
                        ] = False
            else:
                for b in range(self.n_tags):
                    mask[self.action2index((self.token2idx_choices["TAG"], b + 1))] = (
                        False
                    )

        if "TECH" not in assigned:
            if "SECTOR" in assigned and "TAG" in assigned:
                allowed_techs_sector = self.sector2tech_idx[state["SECTOR"]]
                allowed_techs_tag = self.tag2tech_idx[state["TAG"]]
                allowed_techs_idx = list(
                    set(allowed_techs_sector) & set(allowed_techs_tag)
                )
                for a in allowed_techs_idx:
                    mask[self.action2index((self.token2idx_choices["TECH"], a))] = False
            elif "SECTOR" in assigned and "TAG" not in assigned:
                allowed_techs_idx = self.sector2tech_idx[state["SECTOR"]]
                for a in allowed_techs_idx:
                    mask[self.action2index((self.token2idx_choices["TECH"], a))] = False
            elif "SECTOR" not in assigned and "TAG" in assigned:
                allowed_techs_idx = self.tag2tech_idx[state["TAG"]]
                for a in allowed_techs_idx:
                    mask[self.action2index((self.token2idx_choices["TECH"], a))] = False
            else:
                for b in range(self.n_techs):
                    mask[self.action2index((self.token2idx_choices["TECH"], b + 1))] = (
                        False
                    )

        if "AMOUNT" not in assigned:
            for b in range(self.n_amounts):
                mask[self.action2index((self.token2idx_choices["AMOUNT"], b + 1))] = (
                    False
                )

        if (
            "TECH" not in assigned and len(self.techs_available) != self.n_techs
        ):  # no need to double check if TECH has alredy been assigned or all techs are available

            unavailable_techs_idx = [
                self.token2idx_techs[t]
                for t in self.techs
                if t not in techs_available_tokens
            ]
            unavailable_sectors_idx = []
            for s in self.sectors:
                sector_technologies = self.network_structure["sector2tech"][s]
                available_sector = bool(
                    set(sector_technologies) & set(techs_available_tokens)
                )
                if not available_sector:
                    unavailable_sectors_idx.append(self.token2idx_sectors[s])

            unavailable_tags_idx = []
            for t in self.tags:
                tag_technologies = self.network_structure["tag2tech"][t]
                available_tag = bool(
                    set(tag_technologies) & set(techs_available_tokens)
                )
                if not available_tag:
                    unavailable_tags_idx.append(self.token2idx_tags[t])

            for t in unavailable_techs_idx:
                mask[self.action2index((self.token2idx_choices["TECH"], t))] = True
            for s in unavailable_sectors_idx:
                mask[self.action2index((self.token2idx_choices["SECTOR"], s))] = True
            for t in unavailable_tags_idx:
                mask[self.action2index((self.token2idx_choices["TAG"], t))] = True

        return mask

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

        assigned = self.get_assigned_attributes(state)

        parents = []
        actions = []
        # Parents are the ones which could have assigned each value
        # Exceptions: force avoiding path tech -> tag -> sector, or assigning an amount
        # in the middle of the tech-sector-tag sequence
        for a in assigned:
            if (a == "TAG" and "TECH" in assigned and "SECTOR" not in assigned) or (
                a == "AMOUNT"
                and "TECH" in assigned
                and ("SECTOR" not in assigned or "TAG" not in assigned)
            ):
                continue
            temp_state = copy(state)
            temp_state[a] = 0
            parents.append(copy(temp_state))
            temp_action = (self.token2idx_choices[a], state[a])
            actions.append(copy(temp_action))

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

        # If EOS check investment first
        if action == self.eos:
            if self.well_defined_investment():
                self.n_actions += 1
                self.done = True
                return self.state, action, True
            else:
                return self.state, action, False

        valid = True
        self.n_actions += 1

        # Read the choice, and apply the value
        self.state[self.idx2token_choices[action[0]]] = action[1]

        return self.state, action, valid

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment.

        The maximum trajectory length is 5, corresponding to one action per choice plus
        EOS.
        """
        return 5

    def states2proxy(
        self, states: List[Dict]
    ) -> List[List[Dict]]:  # TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: the batch is
        simply converted into a tensor of indices.

        Parameters
        ----------
        states : list of dictionaries
            A batch of states in environment format, as list of states

        Returns
        -------
        The same list, but as a list of lists of one dictionary each, for compatibility with plan.
        Processing is performed in the proxy model
        """
        processed_states = [[single_state] for single_state in states]

        return processed_states

    def states2policy(
        self, states: Union[List[Dict[str, int]], List[TensorType["max_length"]]]
    ) -> torch.Tensor:  # TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: states
        are one-hot encoded.

        Parameters
        ----------
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        sectors = tlong([s["SECTOR"] for s in states], self.device)
        tags = tlong([s["TAG"] for s in states], self.device)
        techs = tlong([s["TECH"] for s in states], self.device)
        amounts = tlong([s["AMOUNT"] for s in states], self.device)

        # One-hot encode each
        onehot_sector = F.one_hot(sectors, num_classes=self.n_sectors + 1).to(
            self.float
        )
        onehot_tag = F.one_hot(tags, num_classes=self.n_tags + 1).to(self.float)
        onehot_tech = F.one_hot(techs, num_classes=self.n_techs + 1).to(self.float)
        onehot_amount = F.one_hot(amounts, num_classes=self.n_amounts + 1).to(
            self.float
        )

        # Concatenate along the last dimension
        batch_tensor = torch.cat(
            [onehot_sector, onehot_tag, onehot_tech, onehot_amount], dim=-1
        )
        return batch_tensor

    def state2readable(self, state: Dict[str, int] = None) -> str:
        """
        Converts a state into a human-readable string.

        The output string contains the letter corresponding to each index in the state,
        separated by spaces.

        Args
        ----
        states : tensor
            A state in environment format. If None, self.state is used.

        Returns
        -------
        A string of space-separated letters.
        """
        state = self._get_state(state)
        assigned = self.get_assigned_attributes(state)
        # Map each index to its token
        if "SECTOR" in assigned:
            sector_str = self.idx2token_sectors[state["SECTOR"]]
        else:
            sector_str = "UNASSIGNED_SECTOR"
        if "TAG" in assigned:
            tag_str = self.idx2token_tags[state["TAG"]]
        else:
            tag_str = "UNASSIGNED_TAG"
        if "TECH" in assigned:
            tech_str = self.idx2token_techs[state["TECH"]]
        else:
            tech_str = "UNASSIGNED_TECH"
        if "AMOUNT" in assigned:
            amount_str = self.idx2token_amounts[state["AMOUNT"]]
        else:
            amount_str = "UNASSIGNED_AMOUNT"

        # Combine into a single readable string
        return f"{sector_str} | {tag_str} | {tech_str} | {amount_str}"

    def readable2state(self, readable: str) -> Dict[str, int]:
        """
        Converts a state in readable format into the environment format (dict of indices).

        Args
        ----
        readable : str
            A state in readable format, with fields separated by " | ".
            Example: "POWER | UNASSIGNED_TAG | power_HYDRO | HIGH"

        Returns
        -------
        dict
            A state dictionary with indices for SECTOR, TAG, TECH, and AMOUNT.
        """

        parts = [p.strip() for p in readable.split("|")]
        state = {}

        if parts[0] != "UNASSIGNED_SECTOR":
            state["SECTOR"] = self.token2idx_sectors[parts[0]]
        else:
            state["SECTOR"] = 0

        if parts[1] != "UNASSIGNED_TAG":
            state["TAG"] = self.token2idx_tags[parts[1]]
        else:
            state["TAG"] = 0

        if parts[2] != "UNASSIGNED_TECH":
            state["TECH"] = self.token2idx_techs[parts[2]]
        else:
            state["TECH"] = 0

        if parts[3] != "UNASSIGNED_AMOUNT":
            state["AMOUNT"] = self.token2idx_amounts[parts[3]]
        else:
            state["AMOUNT"] = 0

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
            tech_idx = random.choice(self.techs_available)
            tech_token = self.idx2token_techs[tech_idx]

            amount_token = random.choice(self.amounts)
            amount_idx = self.token2idx_amounts[amount_token]

            sector_token = self.network_structure["tech2sector"][tech_token]
            sector_idx = self.token2idx_sectors[sector_token]

            tag_token = self.network_structure["tech2tag"][tech_token]
            tag_idx = self.token2idx_tags[tag_token]

            state = {
                "SECTOR": sector_idx,
                "TAG": tag_idx,
                "TECH": tech_idx,
                "AMOUNT": amount_idx,
            }
            states.append(state)

        return states

    def well_defined_investment(
        self,
        state: Optional[Dict] = None,
    ) -> bool:
        state = self._get_state(state)
        assigned = self.get_assigned_attributes(state)
        return (
            "TECH" in assigned
            and "AMOUNT" in assigned
            and "SECTOR" in assigned
            and "TAG" in assigned
        )

    def get_assigned_attributes(
        self,
        state: Optional[Dict] = None,
    ) -> List[str]:
        state = self._get_state(state)
        return [key for key, value in state.items() if value != 0]

    def set_available_techs(self, techs: Iterable):
        """
        Updates the set of available techs.

        This method modifies ``self.techs_available`` by updating with a tuple version
        of the technology indices passed as input.

        Parameters
        ----------
        techs : iterable
            A set of technology indices.
        """
        self.techs_available = tuple(techs)

    def constrain_on_all(self, filled: TensorType):
        """
        Defines the constraints based on other elements of the set of investments.

        Receives as input an Mx3 tensor, with M being the other elements of the Set, *excluding* the environment itself.
        First Columns contains all assigned sector indices, second columns contains all tag indices, third columns contains all technology indices.
        """
        # Extract assigned techs from column 2 (TECH column)
        tech_column = filled[:, 2]

        # Get non-zero tech indices
        assigned_mask = tech_column != 0
        assigned_techs = set(tech_column[assigned_mask].tolist())
        techs_available = set(range(1, self.n_techs + 1)) - assigned_techs
        self.set_available_techs(techs_available)
        self.filled_on_set = filled
