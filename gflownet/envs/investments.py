"""
Climate-economics environment:
Discrete investment options
"""

from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
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

TAGS = tuple(
    [
        "GREEN",
        "BROWN",
        "CCS"
    ]
)

TECHS = tuple(
    [
        'power_COAL_noccs',
        'power_COAL_ccs',
        'power_NUCLEAR',
        'power_OIL',
        'power_GAS_noccs',
        'power_GAS_ccs',
        'power_HYDRO',
        'power_BIOMASS_noccs',
        'power_BIOMASS_ccs',
        'power_WIND_onshore',
        'power_WIND_offshore',
        'power_SOLAR',
        'thermal_SOLAR',
        'enduse_COAL_ccs',
        'power_STORAGE',
        'production_HYDROGEN',
        'refueling_station_HYDROGEN',
        'pipelines_HYDROGEN',
        'DAC_liquid_sorbents',
        'DAC_solid_sorbents',
        'DAC_calcium_oxide',
        'CARS_trad',
        'CARS_hybrid',
        'CARS_electric',
        'CARS_fuelcell',
        'HEAVYDUTY_trad',
        'HEAVYDUTY_hybrid',
        'HEAVYDUTY_electric',
        'HEAVYDUTY_fuelcell',
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
    "POWER" : ["GREEN", "BROWN", "CCS"],
    "ENERGY" : ["GREEN", "CCS"],
    "VEHICLES" : ["GREEN", "BROWN"],
    "STORAGE" : ["GREEN"],
    "DAC" : ["CCS"],
}
ALLOWED_TAG2SECTOR = {
    "GREEN" : ["POWER", "ENERGY", "VEHICLES", "STORAGE"],
    "BROWN" : ["POWER", "VEHICLES"],
    "CCS" : ["POWER", "ENERGY", "DAC"],
}
ALLOWED_SECTOR2TECH = {
    "POWER": ['power_COAL_noccs',
        'power_COAL_ccs',
        'power_NUCLEAR',
        'power_OIL',
        'power_GAS_noccs',
        'power_GAS_ccs',
        'power_HYDRO',
        'power_BIOMASS_noccs',
        'power_BIOMASS_ccs',
        'power_WIND_onshore',
        'power_WIND_offshore',
        'power_SOLAR'],
    "ENERGY": ['thermal_SOLAR',
        'enduse_COAL_ccs'],
    "VEHICLES": ['CARS_trad',
        'CARS_hybrid',
        'CARS_electric',
        'CARS_fuelcell',
        'HEAVYDUTY_trad',
        'HEAVYDUTY_hybrid',
        'HEAVYDUTY_electric',
        'HEAVYDUTY_fuelcell'],
    "STORAGE": ['power_STORAGE',
        'production_HYDROGEN',
        'refueling_station_HYDROGEN',
        'pipelines_HYDROGEN'],
    "DAC": ['DAC_liquid_sorbents',
        'DAC_solid_sorbents',
        'DAC_calcium_oxide'],
}
ALLOWED_TAG2TECH = {
    "GREEN" : ['power_NUCLEAR',
        'power_HYDRO',
        'power_WIND_onshore',
        'power_WIND_offshore',
        'power_SOLAR',
        'thermal_SOLAR',
        'power_STORAGE',
        'production_HYDROGEN',
        'refueling_station_HYDROGEN',
        'pipelines_HYDROGEN',
        'CARS_hybrid',
        'CARS_electric',
        'CARS_fuelcell',
               'HEAVYDUTY_hybrid',
        'HEAVYDUTY_electric',
        'HEAVYDUTY_fuelcell'],
    "BROWN" : ['power_COAL_noccs',
        'power_OIL',
        'power_GAS_noccs',
        'power_BIOMASS_noccs',
        'CARS_trad',
        'HEAVYDUTY_trad'],
    "CCS" : ['power_COAL_ccs',
        'power_GAS_ccs',
        'power_BIOMASS_ccs',
        'enduse_COAL_ccs',
        'DAC_liquid_sorbents',
        'DAC_solid_sorbents',
        'DAC_calcium_oxide']
}

class Single_Investment_DISCRETE(GFlowNetEnv):

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
        if tags is None:
            self.tags = TAGS
        else:
            self.tags = tags
        if techs is None:
            self.techs = TECHS
        else:
            self.techs = techs
        if amounts is None:
            self.amounts = AMOUNTS
        else:
            self.amounts = amounts
        self.n_sectors = len(self.sectors)
        # Dictionaries
        self.idx2token_choices = {idx + 1: token for idx, token in enumerate(self.choices)}
        self.token2idx_choices = {token: idx for idx, token in self.idx2token_choices.items()}

        self.idx2token_sectors = {idx + 1: token for idx, token in enumerate(self.sectors)}
        self.token2idx_sectors = {token: idx for idx, token in self.idx2token_sectors.items()}

        self.idx2token_tags = {idx + 1: token for idx, token in enumerate(self.tags)}
        self.token2idx_tags = {token: idx for idx, token in self.idx2token_tags.items()}

        self.idx2token_techs = {idx + 1: token for idx, token in enumerate(self.techs)}
        self.token2idx_techs = {token: idx for idx, token in enumerate(self.idx2token_techs.items())}

        self.idx2token_amounts = {idx + 1: token for idx, token in enumerate(self.amounts)}
        self.token2idx_amounts = {token: idx for idx, token in enumerate(self.idx2token_techs.items())}
        # Source state: undefined investment
        #self.source = {'SECTOR': 0,
        #               'TAG': 0,
        #               'TECH': 0,
        #               'AMOUNT': 0,
        #               }
        self.source = [0,0,0,0]
        self.state_names = ['SECTOR', 'TAG', 'TECH', 'AMOUNT']

        self.network_structure = {
            "sector2tag": ALLOWED_SECTOR2TAGS,
            "tag2sector": ALLOWED_TAG2SECTOR,
            "sector2tech": ALLOWED_SECTOR2TECH,
            "tag2tech": ALLOWED_TAG2TECH,
        }
        # DO WE NEED AN END OF SEQUENCE ACTION?
        # self.eos = (self.eos_idx,)

        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs list with all possible actions, including eos.

        An action is represented by a single-element tuple indicating the index of the
        letter to be added to the current sequence (state).

        The action space of this parent class is:
            action_space: [(0,), (1,), (-1,)]
        """
        all_actions = [
            (self.token2idx_choices['SECTOR'], self.token2idx_sectors[token_S]) for token_S in self.sectors] + [
            (self.token2idx_choices['TAG'], self.token2idx_tags[token_t]) for token_t in self.tags] + [
            (self.token2idx_choices['TECH'], self.token2idx_techs[token_T]) for token_T in self.techs] + [
            (self.token2idx_choices['AMOUNT'], self.token2idx_amounts[token_A]) for token_A in self.amounts]
        return all_actions

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
    ) -> List[bool]:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.

        Args
        ----
        state : tensor
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
        mask = [False for _ in range(self.action_space_dim)]
        flags = []

        assigned = [i for i, val in enumerate(state) if val != 0]
        for a in assigned: #what is already set cannot be changed
            flags.extend(i for i, (x, y) in enumerate(self.action_space) if x == a)

        if (state[self.state_names.index('TECH')]!=0): #If the tech is set, only the amount can be chosen
            flags.extend(i for i, (x, y) in enumerate(self.action_space) if x != self.state_names.index('AMOUNT'))

        if (state[self.state_names.index('SECTOR')] != 0): #if the sector is set, choose only a compatible tech
            allowed_techs = self.network_structure['sector2tech'][
                self.state_names[state[self.state_names.index('SECTOR')]]]
            flags.extend(i for i, (x, y) in enumerate(self.action_space) if (x == self.state_names.index('TECH') and not
                                                                          self.idx2token_techs[y] in allowed_techs))
            if (state[self.state_names.index('TAG')] == 0): #if the sector is chosen and the tag not, choose only a compatible tag
                allowed_tags = self.network_structure['sector2tags'][self.state_names[state[self.state_names.index('SECTOR')]]]
                flags.extend(i for i, (x, y) in enumerate(self.action_space) if (x == self.state_names.index('TAG') and not
                                                                             self.idx2token_tags[y] in allowed_tags))

        if (state[self.state_names.index('TAG')] != 0): #if the taf is set, choose only a compatible tech
            allowed_techs = self.network_structure['tag2tech'][self.state_names[state[self.state_names.index('TAG')]]]
            flags.extend(i for i, (x, y) in enumerate(self.action_space) if (x == self.state_names.index('TECH') and not
                                                                        self.idx2token_techs[y] in allowed_techs))
            if (state[self.state_names.index('SECTOR')] == 0): #if the tag is set and the sector not, choose only a compatible sector
                allowed_sectors = self.network_structure['tags2sector'][self.state_names[state[self.state_names.index('TAG')]]]
                flags.extend(i for i, (x, y) in enumerate(self.action_space) if (x == self.state_names.index('SECTOR') and not
                                                                             self.idx2token_sectors[y] in allowed_sectors))

        for f in set(flags):
            mask[f] = True
        return mask

    def get_parents(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        The GFlowNet graph is a tree and there is only one parent per state.

        Args
        ----
        state : tensor
            Input state. If None, self.state is used.

        done : bool
            Whether the trajectory is done. If None, self.done is used.

        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format. This environment has a single parent per
            state.

        actions : list
            List of actions that lead to state for each parent in parents. This
            environment has a single parent per state.
        """
        state = self._get_state(state)

        if self.equal(state, self.source):
            return [], []

        assigned = [i for i, val in enumerate(state) if val != 0]

        parents = []
        actions = []
        # Parents are the ones who could have assigned each value
        # exceptions: if the tech is already assigned, sector and tag had to be previously assigned
        for a in assigned:
            if (self.state_names[a] == 'SECTOR' and state[self.state_names.index('TECH')] != 0) or (self.state_names[a] == 'TAG' and state[self.state_names.index('TECH')] != 0):
                continue
            temp_state = copy(state)
            temp_state[a] = 0
            parents.append(copy(temp_state))
            temp_action = (a, state[a])
            actions.append(copy(temp_action))

        return parents, actions

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> [List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple int values indicating the
            dimensions to increment by 1.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        # If done, exit immediately <- CHECK WITH ALEX
        if self.done:
            return self.state, action, False

        # Generic pre-step checks
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, False

        valid = True
        self.n_actions += 1

        # read the choice, and apply the value
        #self.state[self.idx2token_choices[action[0]]] = action[1] # WORKS ON DICTIONARY
        self.state[action[0]] = action[1]

        #check if action is complete
        if self.state[self.state_names.index('TECH')] != 0 and self.state[self.state_names.index('AMOUNT')] != 0:
            self.done = True
        return self.state, action, valid

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment.

        The maximum trajectory lenght is the maximum sequence length (self.max_length)
        plus one (EOS action).
        """
        return 4

    def states2proxy(
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: the batch is
        simply converted into a tensor of indices.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        A list containing all the states in the batch, represented themselves as lists.
        """
        batch_size = len(states)
        batch_tensor = torch.zeros((batch_size, len(self.techs)), dtype=torch.float32)

        for i, row in enumerate(states):
            pos, val = row[2], row[3]
            batch_tensor[i, pos - 1] = val  # subtract 1 since positions start at 1
        return tlong(states, device=self.device)

    def states2policy(
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: states
        are one-hot encoded.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tlong(states, device=self.device)
        return states

    def state2readable(self, state: List[int] = None) -> str:
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
        state = self._unpad(state)
        return "".join([str(self.idx2token[idx]) + " " for idx in state])[:-1]

    def readable2state(self, readable: str) -> List[int]:
        """
        Converts a state in readable format into the "environment format" (tensor)

        Args
        ----
        readable : str
            A state in readable format - space-separated letters.

        Returns
        -------
        A tensor containing the indices of the letters.
        """
        if readable == "":
            return self.source
        return self._pad([self.token2idx[token] for token in readable.split(" ")])

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[List[int]]:
        """
        Constructs a batch of n states uniformly sampled in the sample space of the
        environment.

        Args
        ----
        n_states : int
            The number of states to sample.

        seed : int
            Random seed.
        """
        n_letters = len(self.letters)
        n_per_length = tlong(
            [n_letters**length for length in range(1, self.max_length + 1)],
            device=self.device,
        )
        lengths = Categorical(logits=n_per_length.repeat(n_states, 1)).sample() + 1
        samples = torch.randint(
            low=1, high=n_letters + 1, size=(n_states, self.max_length)
        )
        for idx, length in enumerate(lengths):
            samples[idx, length:] = 0
        return samples.tolist()
