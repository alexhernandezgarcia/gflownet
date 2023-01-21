from abc import abstractmethod
from typing import List, Tuple
import numpy.typing as npt
from torchtyping import TensorType
from .base import GFlowNetEnv
import numpy as np

class MultiFidelityEnvWrapper(GFlowNetEnv):
    """
    Multi-fidelity environment for GFlowNet.
    Define a class that takes any instance of the derived GFlowNetEnv class and adds support for multi-fidelity.
    The following members must be defined:
    self.fidelities: List[int] = [0, 1, 2, 3]
    """
    def __init__(self, env, n_fid, oracle):
        self.env = env
        self.n_fid = n_fid
        self.fid = self.env.eos + 1
        if isinstance(oracle, list):
            # TODO: make state2oracle a list too
            pass
        else:
            self.state2oracle = self.env.state2oracle
        self.action_space = self.get_actions_space()
        self.fixed_policy_output = self.get_fixed_policy_output()
        self.reset()
        # TODO: I think some memeber variable of the base env that are accessed by gflownet.py will be required to be part of this class so as to not break the existing sf-compatible code
        # Fid in the state should range from 0 to self.n_fid -1 for easy one hot encoding

    def get_actions_space(self): 
        actions =  self.env.get_actions_space()
        for fid_index in range(self.n_fid):
            actions = actions +[(self.fid, fid_index)] 

    def get_fixed_policy_output(self):
        return self.env.get_fixed_policy_output()

    def reset(self, env_id=None, fid: int = None):
        # Offline Trajectory: might want to reset the env with a specific fid
        # TODO: should this be self.env or env?
        # TODO: env is a member variable of MFENV
        # TODO: all member variables of env are also members of MFENV. Is this desired?
        self.env = self.env.reset(env_id)
        # TODO: check if fid as a member variable is required.
        self.fid = fid
        self.state = self.state + [self.fid]
        return self

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        mask = self.env.get_mask_invalid_actions_forward(state, done)
        # if fidelity has not been chosen, done is invalid and fid is valid
        if state[-1] == -1:
            mask[-1] = True
            mask = + [False]
        # if fidelity has been chosen then fid is invalid
        else:
            mask=+[True]
        return mask
        # TODO: do we need check done if fid is invalid?

    def state2polciy(self, state: List = None):
        state_proxy = self.env.state2policy(state)
        fid_proxy = np.zeros((self.n_fid), dtype=np.float32)
        fid_proxy[state[-1]] = 1
        return np.concatenate((state_proxy, fid_proxy), axis=0)

    def statebatch2policy(self, states: List[List]) -> npt.NDArray[np.float32]:
        # Remark: Fidelity cannot be -1 for any state here
        # Feed states (without fidelity) to state2policy
        # print last element of each state in states
        # print(states[:, -1])
        state_proxy =  self.env.statebatch2policy(states[:, -1])
        fid_proxy = np.zeros((len(states), self.n_fid), dtype=np.float32)
        fid_proxy[np.arange(len(states)), states[:, -1]] = 1
        return np.concatenate((state_proxy, fid_proxy), axis=1)

    def policy2state(self, state_policy: List) -> List:
        policy_no_fid = state_policy[:, :-self.n_fid]
        state_no_fid = self.env.policy2state(policy_no_fid)
        fid = np.argmax(state_policy[:, -self.n_fid:], axis=1)
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

    def state2oracle(self, state: List = None):
        state_oracle = self.env.state2oracle(state)
        return state_oracle, self.fid

    def get_parents(self, state=None, done=None, action=None):
        parents, actions = super().get_parents(state[:, :-1], done, action)
        # TODO: optimize code
        for idx, s in enumerate(state):
            if s[-1]==-1:
                fid = s[-1]
                parent = s[:-1] + [-1]
                actions[idx].append(self.action_space[fid+self.fid])
                parents[idx].append(parent)
        return parents, actions

    def step(self, action_idx, state=None, done = None):
        if state is None:
            state = self.state
        if done is None:
            done = self.done
        if done:
            return state, action_idx, False
        if action_idx == self.fid:
            state[-1] = self.action_space[action_idx]
            return state, action_idx, True
        else:
            fid = state[-1]
            state, action_idx, done = self.env.step(action_idx, state[:-1], done)
            state = state + [fid]
            return state, action_idx, done