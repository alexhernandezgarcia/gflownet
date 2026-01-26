"""
Environment to sample a plan of investments, namely an investment amount for each
technology.

The environment is designed as a Stack of N investment environments, with N the total
number of technologies. Each technology must be selected once, therefore the
inter-environment constraints restrict the choice of technologies to those that have
not been set yet.
"""

import copy
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from torchtyping import TensorType

import math
from torch.distributions import Bernoulli
from gflownet.utils.common import tbool

from gflownet.envs.iam.investment import TECHS, InvestmentDiscrete
from gflownet.envs.set import SetFix

try:
    profile
except NameError:
    def profile(func):
        return func

class Plan(SetFix):
    """
    A Stack of InvestmentDiscrete environments.
    """

    def __init__(
        self,
        sectors: Iterable = None,
        tags: Iterable = None,
        techs: Iterable = None,
        amounts: Iterable = None,
        **kwargs,
    ):
        """
        Initializes the Plan environment.

        Parameters
        ----------
        sectors : iterable (optional)
           A set of sectors to which technologies belong, described by a string.
        tags : iterable (optional)
           A set of tags associated to technologies, described by a string.
        techs : iterable (optional)
           A set of technologies, described by a string.
        amounts : iterable (optional)
           A set of amounts of investment, described by a string.
        """
        if techs is None:
            self.n_techs = len(TECHS)
        else:
            self.n_techs = len(techs)
        self.techs = set(range(1, self.n_techs + 1))

        # Define sub-environments of the Stack
        subenvs = [
            InvestmentDiscrete(sectors=sectors, tags=tags, techs=techs, amounts=amounts)
            for _ in range(self.n_techs)
        ]

        self.idx2token_techs = subenvs[0].idx2token_techs
        self.idx2token_amounts = subenvs[0].idx2token_amounts
        self.token2idx_sectors = subenvs[0].token2idx_sectors
        self.token2idx_tags = subenvs[0].token2idx_tags
        self.network_structure = subenvs[0].network_structure

        self.empy_investment = subenvs[0].source

        self.n_sector_choices = subenvs[0].n_sectors + 1
        self.n_tags_choices = subenvs[0].n_tags + 1
        self.n_techs_choices = self.n_techs + 1
        self.n_amounts_choices = subenvs[0].n_amounts + 1

        self._last_constraint_hash = None

        # Initialize base Stack environment
        super().__init__(
            subenvs=tuple(subenvs),
            can_alternate_subenvs=False,
            **kwargs
        )

        # Pre-compute tech to sector/tag index mappings for faster lookup
        self._tech_idx_to_sector_idx = {
            tech_idx: self.token2idx_sectors[
                self.network_structure["tech2sector"][self.idx2token_techs[tech_idx]]
            ]
            for tech_idx in range(1, self.n_techs + 1)
        }

        self._tech_idx_to_tag_idx = {
            tech_idx: self.token2idx_tags[
                self.network_structure["tech2tag"][self.idx2token_techs[tech_idx]]
            ]
            for tech_idx in range(1, self.n_techs + 1)
        }

    def _check_has_constraints(self) -> bool:
        """
        The Plan has constraints across sub-environments: each technology can only
        be selected once across all investments.
        """
        return True

    @profile
    def _apply_constraints_forward(
        self,
        action: Tuple = None,
        state: List = None,
        dones: List[bool] = None,
    ):
        """
        Applies constraints across sub-environments, when applicable, in the forward
        direction.

        The constraint to be applied is restricting the available technologies of the
        next environment. The constraint is applied if the EOS action of one of the
        Investment environments is received or, for the use by set_state(), if the
        action is None, then taking into account the first environment that is not
        done.

        Parameters
        ----------
        action : tuple
            An action from the SetBox environment.
        state : list or tensor (optional)
            A state from the SetBox environment.
        dones : list
            A list indicating the sub-environments that are done.
        """
        if action is not None and action[0] != -1:
            return

        current_state = state if state is not None else self.state

        # Get done flags - only update non-done subenvs
        dones_list = self._get_dones(current_state)
        non_done_indices = [i for i, d in enumerate(dones_list) if not d]

        if not non_done_indices:
            return

        # Compute hash of current tech assignments to avoid redundant updates
        tech_assignments = tuple(self._get_substate(current_state, i)["TECH"] for i in range(self.n_techs))
        constraint_hash = hash(tech_assignments)

        if constraint_hash == self._last_constraint_hash:
            return  # No change in tech assignments, skip constraint update

        self._last_constraint_hash = constraint_hash

        # Compute filled array once
        filled = self._states2array(
            current_state=current_state, fill_in_from_tech=True, with_amounts=False
        )
        filled_tensor = torch.from_numpy(filled).float()

        # Only update non-done subenvs
        n = self.n_techs
        for idx in non_done_indices:
            mask = torch.ones(n, dtype=torch.bool)
            mask[idx] = False
            other_investments = filled_tensor[mask]
            self.subenvs[idx].constrain_on_all(other_investments)

    @profile
    def _apply_constraints_backward(self, action: Tuple = None, state = None):
        """
        Applies constraints in the backward direction by recomputing based on current state.

        Parameters
        ----------
        action : tuple
            An action from the Set environment.
        state : dict (optional)
            A state from the Set environment. If None, self.state is used.
        """
        current_state = state if state is not None else self.state
        self._apply_constraints_forward(action=None, state=current_state)

    @profile
    def _get_techs_set(self, state: Optional[List]) -> Set[int]:
        """
        Returns the set of technologies that have already been set in the state.

        Parameters
        ----------
        state : list (optional)
            A state of the Plan environment. If None, self.state is used.

        Returns
        -------
        set
            The set of technology indices that have been set in the state.
        """
        state = self._get_state(state)
        techs = []
        for stage in range(self.n_techs):
            tech = self._get_substate(state, stage)["TECH"]
            if tech != 0:
                techs.append(tech)
        return set(techs)

    @profile
    def states2proxy(self, states: List[List]) -> List[List[Dict]]:
        """
        Prepares a batch of states in "environment format" for a proxy: the batch is
        simply converted into a tensor of indices.

        Parameters
        ----------
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        The same list of lists, but with only the dictionaries, processing is performed in the proxy model
        """
        processed_states = []
        for state in states:
            substates = self._get_substates(state)
            processed_states.append(
                [self.decode_investment_for_proxy(x) for x in substates]
            )
        return processed_states

    def decode_investment_for_proxy(self, state: Dict) -> Dict:
        to_pass = state.copy()
        to_pass["TECH"] = "SUBS_" + self.idx2token_techs[state["TECH"]]
        to_pass["AMOUNT"] = self.idx2token_amounts[state["AMOUNT"]]
        return to_pass

    @profile
    def _states2array(
        self, current_state=None, fill_in_from_tech=False, with_amounts=False
    ) -> np.ndarray:
        """
        Read all state subenvironments and convert them into tensors.
        Three columns: [SECTOR, TAG, TECH]
        if with_amount: Four Columns [SECTOR, TAG, TECH, AMOUNT]
        """
        current_state = current_state if current_state is not None else self.state

        # Get all substates
        substates = self._get_substates(current_state)

        # Convert to list of values
        if with_amounts:
            state_to_list = [
                [inv["SECTOR"], inv["TAG"], inv["TECH"], inv["AMOUNT"]]
                for inv in substates
            ]
        else:
            state_to_list = [
                [inv["SECTOR"], inv["TAG"], inv["TECH"]]
                for inv in substates
            ]

        filled = np.array(state_to_list, dtype=np.int32)

        if fill_in_from_tech:
            # Vectorized approach: find all rows where tech is set but sector/tag are not
            tech_col = filled[:, 2]
            sector_col = filled[:, 0]
            tag_col = filled[:, 1]

            # Boolean masks for rows that need filling
            has_tech = tech_col != 0
            needs_sector = has_tech & (sector_col == 0)
            needs_tag = has_tech & (tag_col == 0)

            # Fill sectors using vectorized lookup
            if np.any(needs_sector):
                tech_indices = tech_col[needs_sector].astype(int)
                filled[needs_sector, 0] = [
                    self._tech_idx_to_sector_idx[idx] for idx in tech_indices
                ]

            # Fill tags using vectorized lookup
            if np.any(needs_tag):
                tech_indices = tech_col[needs_tag].astype(int)
                filled[needs_tag, 1] = [
                    self._tech_idx_to_tag_idx[idx] for idx in tech_indices
                ]

        return filled

    @profile
    def randomize_and_temper_sampling_distribution(
            self,
            policy_outputs: TensorType["n_states", "policy_output_dim"],
            probability_random_action: Optional[float] = 0.0,
            temperature: Optional[float] = 1.0,
    ) -> TensorType["n_states", "policy_output_dim"]:
        """
        Override to handle sliced policy outputs correctly.
        """

        do_temper = not math.isclose(temperature, 1.0, abs_tol=1e-08)
        do_random = not math.isclose(probability_random_action, 0.0, abs_tol=1e-08)

        if not do_temper and not do_random:
            return policy_outputs

        logits_sampling = policy_outputs.clone().detach()

        if do_temper:
            logits_sampling /= temperature

        if do_random:
            idx_random = tbool(
                Bernoulli(
                    probability_random_action * torch.ones(policy_outputs.shape[0])
                ).sample(),
                device=self.device,
            )
            # Use uniform distribution matching the actual policy_outputs shape
            logits_sampling[idx_random, :] = 1.0

        return logits_sampling
