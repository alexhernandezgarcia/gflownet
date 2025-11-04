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
        # Initialize base Stack environment
        super().__init__(subenvs=tuple(subenvs), **kwargs)

    def _apply_constraints(
        self,
        action: Tuple = None,
        state: List = None,
        dones: List[bool] = None,
        is_backward: bool = False,
    ):
        """
        Applies constraints across sub-environments, when applicable.

        This method is used in step() and set_state().

        Parameters
        ----------
        action : tuple
            An action from the Plan environment.
        state : list
            A state from the Plan environment.
        is_backward : bool
            Boolean flag to indicate whether the action is in the backward direction.
        dones : list
            A list indicating the sub-environments that are done.
        """
        if not is_backward:
            self._apply_constraints_forward(action, state, dones)

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
        if action is not None and self._depad_action(action) == self.subenvs[0].eos:
            stage_to_constrain = self._get_stage(state) + 1
        elif action is None and sum(dones) > 0 and sum(dones) < self.n_techs:
            stage_to_constrain = dones.index(False)
        else:
            return

        techs_available = self.techs - self._get_techs_set(state)
        self.subenvs[stage_to_constrain].set_available_techs(techs_available)

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
        for stage in range(self._get_stage(state) + 1):
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
