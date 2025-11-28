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

import torch
from torchtyping import TensorType

from gflownet.envs.iam.investment import TECHS, InvestmentDiscrete
from gflownet.envs.stack import Stack


class Plan(Stack):
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

        self.idx2token_techs = copy.deepcopy(subenvs[0].idx2token_techs)
        self.idx2token_amounts = copy.deepcopy(subenvs[0].idx2token_amounts)
        self.token2idx_sectors = copy.deepcopy(subenvs[0].token2idx_sectors)
        self.token2idx_tags = copy.deepcopy(subenvs[0].token2idx_tags)
        self.network_structure = copy.deepcopy(subenvs[0].network_structure)

        self.empy_investment = copy.deepcopy(subenvs[0].source)

        self.n_sector_choices = copy.deepcopy(subenvs[0].n_sectors + 1)
        self.n_tags_choices = copy.deepcopy(subenvs[0].n_tags + 1)
        self.n_techs_choices = self.n_techs + 1
        self.n_amounts_choices = copy.deepcopy(subenvs[0].n_amounts + 1)

        self.n_partial_state_encoding = (
            self.n_sector_choices
            + self.n_tags_choices
            + self.n_techs_choices
            + self.n_amounts_choices
        )

        self.width_one_hot = self.n_partial_state_encoding * self.n_techs

        self.base_row_one_hot = torch.zeros(self.width_one_hot)
        # initialize for source state
        self.base_row_one_hot[
            [x * self.n_partial_state_encoding for x in range(self.n_techs)]
        ] = 1
        self.base_row_one_hot[
            [
                x * self.n_partial_state_encoding + self.n_sector_choices
                for x in range(self.n_techs)
            ]
        ] = 1
        self.base_row_one_hot[
            [
                x * self.n_partial_state_encoding
                + self.n_sector_choices
                + self.n_tags_choices
                for x in range(self.n_techs)
            ]
        ] = 1
        self.base_row_one_hot[
            [
                x * self.n_partial_state_encoding
                + self.n_sector_choices
                + self.n_tags_choices
                + self.n_techs_choices
                for x in range(self.n_techs)
            ]
        ] = 1

        # Initialize base Stack environment
        super().__init__(subenvs=tuple(subenvs), **kwargs)

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

        current_state = state if state is not None else self.state

        filled = self._states2tensor(
            current_state=current_state, fill_in_from_tech=True, with_amounts=False
        )

        for idx, subenv in self.subenvs.items():
            select = list(range(self.n_techs))
            select.pop(idx)
            other_investments = filled[select, :]
            subenv.constrain_on_all(other_investments)

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
        processed_states = [
            [
                self.decode_investment_for_proxy(x)
                for x in single_plan
                if isinstance(x, dict)
            ]
            for single_plan in states
        ]
        return processed_states

    def decode_investment_for_proxy(self, state: Dict) -> Dict:
        to_pass = copy.deepcopy(state)
        to_pass["TECH"] = "SUBS_" + self.idx2token_techs[state["TECH"]]
        to_pass["AMOUNT"] = self.idx2token_amounts[state["AMOUNT"]]
        return to_pass

    def _states2tensor(
        self, current_state=None, fill_in_from_tech=False, with_amounts=False
    ) -> torch.Tensor:
        """
        Read all state subenvironments and convert them into tensors.
        Three columns: [SECTOR, TAG, TECH]
        if with_amount: Four Columns [SECTOR, TAG, TECH, AMOUNT]
        """
        current_state = current_state if current_state is not None else self.state
        if with_amounts:
            filled = torch.zeros(self.n_techs, 4)
        else:
            filled = torch.zeros(self.n_techs, 3)

        for idx in range(self.n_techs):
            filled[idx, 0] = current_state[idx + 1]["SECTOR"]
            filled[idx, 1] = current_state[idx + 1]["TAG"]
            filled[idx, 2] = current_state[idx + 1]["TECH"]

            if fill_in_from_tech:
                if filled[idx, 2] and not filled[idx, 0]:
                    filled[idx, 0] = self.token2idx_sectors[
                        self.network_structure["tech2sector"][
                            self.idx2token_techs[int(filled[idx, 2])]
                        ]
                    ]
                if filled[idx, 2] and not filled[idx, 1]:
                    filled[idx, 1] = self.token2idx_tags[
                        self.network_structure["tech2tag"][
                            self.idx2token_techs[int(filled[idx, 2])]
                        ]
                    ]

            if with_amounts:
                filled[idx, 3] = current_state[idx + 1]["AMOUNT"]
        return filled

    def states2policy(
        self, states: List[List]
    ) -> torch.Tensor:  # TensorType["batch", "state_policy_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: states
        are one-hot encoded.

        Permutation invariance of states:
        initial one hot encoding of eventually partially defined states, followed by ordered one hot encoding of tech-amount pairs

        Parameters
        ----------
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.
        Returns
        -------
        A tensor containing all the states in the batch.
        """

        temp = torch.zeros(len(states), self.width_one_hot)
        for batch_idx, s_ in enumerate(states):
            state_as_tensor = self._states2tensor(
                current_state=s_, fill_in_from_tech=False, with_amounts=True
            )
            # to make it permutation invariant, apply consecutive sorting
            _, indices_amount = torch.sort(
                state_as_tensor[:, 3], descending=False, stable=True
            )
            state_as_tensor = state_as_tensor[indices_amount]
            _, indices_tech = torch.sort(
                state_as_tensor[:, 2], descending=False, stable=True
            )
            state_as_tensor = state_as_tensor[indices_tech]
            _, indices_tag = torch.sort(
                state_as_tensor[:, 1], descending=False, stable=True
            )
            state_as_tensor = state_as_tensor[indices_tag]
            _, indices_sect = torch.sort(
                state_as_tensor[:, 0], descending=False, stable=True
            )
            state_as_tensor = state_as_tensor[indices_sect]

            batch_row = copy.deepcopy(self.base_row_one_hot)

            for inv in range(self.n_techs):
                if any(state_as_tensor[inv, :]):
                    inv_sect = int(state_as_tensor[inv, 0])
                    inv_tag = int(state_as_tensor[inv, 1])
                    inv_tech = int(state_as_tensor[inv, 2])
                    inv_amo = int(state_as_tensor[inv, 3])

                    # remove base encoding
                    batch_row[
                        [
                            inv * self.n_partial_state_encoding,
                            inv * self.n_partial_state_encoding + self.n_sector_choices,
                            inv * self.n_partial_state_encoding
                            + self.n_sector_choices
                            + self.n_tags_choices,
                            inv * self.n_partial_state_encoding
                            + self.n_sector_choices
                            + self.n_tags_choices
                            + self.n_techs_choices,
                        ]
                    ] = 0
                    # assign value: 0 is not assigned, consistent with 1hot
                    batch_row[
                        [
                            inv_sect + inv * self.n_partial_state_encoding,
                            inv_tag
                            + inv * self.n_partial_state_encoding
                            + self.n_sector_choices,
                            inv_tech
                            + inv * self.n_partial_state_encoding
                            + self.n_sector_choices
                            + self.n_tags_choices,
                            inv_amo
                            + inv * self.n_partial_state_encoding
                            + self.n_sector_choices
                            + self.n_tags_choices
                            + self.n_techs_choices,
                        ]
                    ] = 1

            assert batch_row.sum() == (4 * self.n_techs)
            temp[batch_idx, :] = batch_row
        return temp
