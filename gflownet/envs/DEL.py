from typing import List, Tuple
import itertools
import numpy as np
import numpy.typing as npt
import torch
from torchtyping import TensorType
from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.grid import Grid

import itertools

class DEL(Grid):
    def __init__(self, n_dim, min_step_len, max_step_len, eos, length):
        super().__init__(n_dim, length=2)
        self.n_dim = n_dim
        self.min_step_len = min_step_len
        self.max_step_len = max_step_len
        self.eos = eos
        self.state = [0] * n_dim  
        self.done = False
        self.action_space = self.get_actions_space()  
        self.length = length

    def get_actions_space(self):
        """
        Constructs a list with all possible actions, including eos.
        """
        dims = [i for i in range(self.n_dim)]
        actions = [(i,) for i in dims]
        actions.append((-1,))
        return actions

    def step(self, action):
        """
        Update the state based on the given action.
        """
        if action == self.eos:
            # End of sequence action
            self.done = True
            return
        for dim_index in action:
            self.state[dim_index] = 1

    def get_parents(self, state=None, done=None, action=None):
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state, as a list of length length where each element is
            the position at each dimension.

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [(self.eos,)]
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space[:-1]):
                state_aux = state.copy()
                for a_sub in a:
                    if state_aux[a_sub] > 0:
                        state_aux[a_sub] -= 1
                    else:
                        break
                else:
                    parents.append(state_aux)
                    actions.append(a)
            return parents, actions

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        Returns a vector of length the action space + 1: True if forward action is
        invalid given the current state (action dimension is 1), False otherwise (action dimension is 0).
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(len(self.action_space))]
        mask = [False for _ in range(len(self.action_space))]
        for idx, a in enumerate(self.action_space[:-1]):
            for d in a:
                if state[d] == 1:
                    mask[idx] = True
                    break
        return mask
