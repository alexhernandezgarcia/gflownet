"""
Toy environment: A playground environment for small-scale, controllable experiments.

With the default values, the sample, state and action spaces of the environment are
defined as in Figure 2 of the GFlowNet Foundations paper, Bengio et al (JMLR, 2023):

    .. _a link: https://jmlr.org/papers/v24/22-0364.html
"""

import random
from typing import List, Optional, Tuple

import torch.nn.functional as F
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import tfloat, tlong


class Toy(GFlowNetEnv):
    """
    Toy environment: with the default values, the environment has a DAG as in Figure 2
    of the GFlowNet Foundations paper.

    The DAG can be described as follows:

    - The source state, s0, is connected to s1 and s2.
    - s1 is only connected to s3.
    - s2 is connected to s3 and s4.
    - s3 is a terminating state and is also connected to s5.
    - s4 is a terminating state and is also connected to s6.
    - s5 is connected to s7 and s8.
    - s6 is a terminating state and is also connected to s8 and s10.
    - s7 is only connected to s9.
    - s8 is a terminating state and is also connected to s9.
    - s9 is a terminating state and is not connected to other states.
    - s10 is a terminating state and is not connected to other states.

    Therefore, the terminating states are s3, s4, s6, s8, s9 and s10.

    States are represented as a single-element list with the identifying integer.

    Actions are represented as tuples with two integers, where the first element is the
    source state and the second element is the target state, interpreted in the forward
    direction. The end-of-sequence is (-1, -1).

    Attributes
    ----------
    connections : dict
        A dictionary of state connections. Each key is a state index, and the values
        are an iterable (tuple) of state indices to which the state is connected. If
        the state is a terminating state, then -1 must be included in the iterable.
    action_space_only_valid : bool
        Whether the action space should be restricted to only the valid actions (True),
        or instead it should contain or theoretically available actions, that is
        between any two pairs of states (False).
    """

    def __init__(
        self,
        connections: dict = {
            0: (1, 2),
            1: (3,),
            2: (3, 4),
            3: (5, -1),
            4: (6, -1),
            5: (7, 8),
            6: (8, 10, -1),
            7: (9,),
            8: (9, -1),
            9: (-1,),
            10: (-1,),
        },
        action_space_only_valid: bool = True,
        **kwargs,
    ):
        # Convert the iterable with the target states in connections into a tuple for
        # efficiency
        self.connections = {k: tuple(v) for k, v in connections.items()}
        self.action_space_only_valid = action_space_only_valid
        # Source state
        self.source = [0]
        # End-of-sequence action
        self.eos = (-1, -1)
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs list with all possible actions, including eos.

        Actions are represented as tuples with two integers, where the first element is
        the source state and the second element is the target state, interpreted in the
        forward direction. The end-of-sequence is (-1, -1).
        """
        if not self.action_space_only_valid:
            raise NotImplementedError
        actions = []
        for source, targets in self.connections.items():
            actions.extend([(source, target) for target in targets if target != -1])
        actions.append(self.eos)
        return actions

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
    ) -> List[bool]:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.

        Parameters
        ----------
        state : list
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
            return [True] * self.action_space_dim

        # Initialize the mask to all invalid
        mask = [True] * self.action_space_dim
        # Iterate over the action space and make valid the actions corresponding to
        # valid transitions. This assumes that EOS is the last action in the action
        # space.
        for idx, action in enumerate(self.action_space[:-1]):
            if action[0] != state[0]:
                continue
            if action[1] in self.connections[state[0]]:
                mask[idx] = False
        if -1 in self.connections[state[0]]:
            mask[-1] = False
        return mask

    def get_parents(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        Parameters
        ----------
        state : list
            Input state. If None, self.state is used.
        done : bool
            Whether the trajectory is done. If None, self.done is used.
        action : None
            Ignored

        Returns
        -------
        parents : list
            List of valid parents in environment format.
        actions : list
            List of actions that lead to state for each parent in parents.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [state], [self.eos]
        if self.equal(state, self.source):
            return [], []
        # Iterate over the valid transitions to determine the valid parents.
        parents = []
        actions = []
        for source, targets in self.connections.items():
            if state[0] not in targets:
                continue
            parents.append([source])
            actions.append((source, state[0]))
        return parents, actions

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> [List[int], Tuple[int], bool]:
        """
        Performs step given an action.

        Parameters
        ----------
        action : tuple
            Action to be performed. An action is a tuple indicating the source and
            target states, in the forward sense.
        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The state after performing the action.
        action : tuple
            The performed action or attempted to be performed.
        valid : bool
            False, if the action is not allowed for the current state.
        """
        # Generic pre-step checks
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, False
        valid = True
        self.n_actions += 1
        # If action is EOS, set done to True and return state as is
        if action == self.eos:
            self.done = True
            return self.state, action, valid
        # Update state by setting its value to the index of the target state in the
        # action
        self.state[0] = action[1]
        return self.state, action, valid

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment.
        """
        return len(self.connections) + 1

    def states2proxy(self, states: List[List[int]]) -> TensorType["batch", 1]:
        """
        Prepares a batch of states in environment format for a proxy: the batch is
        simply converted into a tensor of state indices.

        Parameters
        ----------
        states : list
            A batch of states in environment format, as a list of states.

        Returns
        -------
        A 2D tensor containing all the states in the batch.
        """
        return tlong(states, device=self.device)

    def states2policy(self, states: List[List[int]]) -> TensorType["batch", "n_states"]:
        """
        Prepares a batch of states in environment format for the policy model: states
        indices are one-hot encoded.

        Parameters
        ----------
        states : list
            A batch of states in environment format, as a list of states.

        Returns
        -------
        A 2D tensor containing all the states in the batch.
        """
        states_policy = F.one_hot(
            tlong(states, device=self.device).squeeze(), len(self.connections)
        )
        return tfloat(states_policy, device=self.device, float_type=self.float)

    def state2readable(self, state: List[int] = None) -> str:
        """
        Converts a state into a human-readable string.

        The output string is simply "s" followed by the index of the state.

        Parameters
        ----------
        state : list
            A state in environment format. If None, self.state is used.

        Returns
        -------
        A string representing the state
        """
        state = self._get_state(state)
        return f"s{state[0]}"

    def readable2state(self, readable: str) -> List[int]:
        """
        Converts a state in readable format into the environment format.

        Parameters
        ----------
        readable : str
            A state in readable format.

        Returns
        -------
        A state in environment format.
        """
        return [int(readable.strip("s"))]

    def get_all_terminating_states(self) -> List[List[int]]:
        """
        Returns a list with all the terminating states in environment format.

        Returns
        -------
        list
            The list of all terminating states.
        """
        return [[idx] for idx, targets in self.connections.items() if -1 in targets]

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[List[int]]:
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
        random.seed(seed)
        return random.choices(self.get_all_terminating_states(), k=n_states)
