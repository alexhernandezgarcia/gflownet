from abc import abstractmethod
from typing import List, Tuple
import numpy.typing as npt
from torchtyping import TensorType
from .base import GFlowNetEnv
import numpy as np


class MultiFidelityEnvWrapper(GFlowNetEnv):
    """
    Multi-fidelity environment for GFlowNet.
    Assumes same data transformation required for all oracles.
    Does not require the different oracles as scoring is performed by GFN not env
    """

    def __init__(self, env, n_fid):
        self.env = env
        self.n_fid = n_fid
        self.fid = self.env.eos + 1
        # Member variables of env are also member variables of MFENV
        vars(self).update(vars(self.env))
        self.action_space = self.get_actions_space()
        self.fixed_policy_output = self.get_fixed_policy_output()
        self.policy_input_dim = len(self.state2policy())
        self.random_policy_output = self.fixed_policy_output
        # Assumes that all oracles required the same kind of transformed dtata
        # self.state2oracle = self.env.state2oracle
        # self.oracle = self.env.oracle
        self.reset()

    def get_actions_space(self):
        actions = self.env.get_actions_space()
        for fid_index in range(self.n_fid):
            actions = actions + [(self.fid, fid_index)]
        return actions

    def reset(self, env_id=None, fid: int = -1):
        # Fid in the state should range from 0 to self.n_fid -1 for easy one hot encoding
        # Offline Trajectory: might want to reset the env with a specific fid
        # TODO: does member variable env need to be updated?
        # env is a member variable of MFENV because we need the functions
        self.env = self.env.reset(env_id)
        # If fid is required as a member variable
        # then it should not be named self.fid as self.fid represents fidelity action
        self.state = self.env.state + [fid]
        # Update the variables that were reset and are required in mf
        self.done = self.env.done
        self.id = self.env.id
        return self

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(len(self.action_space))]
        mask = self.env.get_mask_invalid_actions_forward(state[:-1], done)
        if state[-1] == -1:
            mask = mask + [False for _ in range(self.n_fid)]
        else:
            mask = mask + [True for _ in range(self.n_fid)]
        return mask

    def state2policy(self, state: List = None):
        if state is None:
            state = self.state.copy()
        state_policy = self.env.state2policy(state[:-1])
        fid_policy = np.zeros((self.n_fid), dtype=np.float32)
        if state[-1] != -1:
            fid_policy[state[-1]] = 1
        state_fid_policy = np.concatenate((state_policy, fid_policy), axis=0)
        return state_fid_policy

    def statebatch2policy(self, states: List[List]) -> npt.NDArray[np.float32]:
        state_policy, fid_list = zip(*[(state[:-1], state[-1]) for state in states])
        state_policy = self.env.statebatch2policy(state_policy)
        fid_policy = np.zeros((len(states), self.n_fid), dtype=np.float32)
        fid_array = np.array(fid_list)
        index = np.where(fid_array != -1)[0]
        if index.size:
            fid_policy[index, fid_array[index]] = 1
        state_fid_policy = np.concatenate((state_policy, fid_policy), axis=1)
        return state_fid_policy

    def policy2state(self, state_policy: List) -> List:
        policy_no_fid = state_policy[:, : -self.n_fid]
        state_no_fid = self.env.policy2state(policy_no_fid)
        fid = np.argmax(state_policy[:, -self.n_fid :], axis=1)
        state = state_no_fid + fid.to_list()
        return state

    def state2readable(self, state=None):
        readable_state = super().state2readable(state[-1])
        fid = str(state[-1])
        return readable_state + "?" + fid

    def readable2state(self, readable):
        fid = readable.split("?")[-1]
        state = super().readable2state(readable)
        state = state + [fid]
        return state

    def get_parents(self, state=None, done=None, action=None):
        assert self.state[:-1] == self.env.state
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        parents_no_fid, actions = self.env.get_parents(state[:-1])
        parents = [parent + [-1] for parent in parents_no_fid]
        # TODO: if fidelity has been chosen,
        # parents = parents + [parent + [fid] for parent in parents]
        if state[-1] != -1:
            fid = state[-1]
            parent = state[:-1] + [-1]
            actions.append((self.fid, fid))
            parents.append(parent)
        return parents, actions

    def step(self, action):
        assert self.state[:-1] == self.env.state
        if self.done:
            return self.state, action, False
        if action[0] == self.fid:
            # unnecessary to check because already implemented in  mask
            if self.state[-1] != -1:
                return state, action, False
            self.state[-1] = action[1]
            if self.env.done == True:
                self.done = True
            return self.state, action, True
        else:
            fid = self.state[-1]
            state, action, valid = self.env.step(action)
            self.state = state + [fid]
            return self.state, action, valid
