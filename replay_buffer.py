# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from collections import namedtuple
import random

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10

# Definition needed to store memory replay in pickle
Query_Transition = namedtuple(
    "Transition",
    ("model_state", "action_state", "next_model_state", "next_action_state", "reward", "terminal"),
)

Parameter_Transition = namedtuple(
    "Transition",
    ("model_state", "action", "next_model_state", "reward", "terminal"),
)


class QuerySelectionReplayMemory(object):
    """
    Class that encapsulates the experience replay buffer, the push and sampling method
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(
        self, model_state, action_state, next_model_state, next_action_state, reward, terminal
    ):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = None
        self.memory[self.position] = Query_Transition(
            model_state, action_state, next_model_state, next_action_state, reward, terminal
        )
        self.position = (self.position + 1) % self.capacity

        del model_state
        del action_state
        del next_model_state
        del next_action_state
        del terminal
        del reward

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ParameterUpdateReplayMemory(object):
    """
    Class that encapsulates the experience replay buffer, the push and sampling method
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(
        self, model_state, action, next_model_state, reward, terminal
    ):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = None
        #self.memory[self.position] = Parameter_Transition(
        #    model_state.cpu(), action, next_model_state.cpu(), reward, terminal
        #)
        self.memory[self.position] = {
            "model_state":model_state.cpu(),
            "action":action,
            "next_model_state":next_model_state.cpu(),
            "reward":reward,
            "terminal":terminal
        }
        self.position = (self.position + 1) % self.capacity

        del model_state
        del action
        del next_model_state
        del terminal
        del reward

    def sample(self, batch_size):
        return random.sample(list(self.memory), batch_size)

    def __len__(self):
        return len(self.memory)