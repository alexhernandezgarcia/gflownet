# Adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
from collections import namedtuple
import random

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10

# Definition needed to store memory replay in pickle
Transition = namedtuple('Transition',
                        ('state', 'state_subset', 'action', 'next_state', 'next_state_subset', 'reward'))


class ReplayMemory(object):
    '''
    Class that encapsulates the experience replay buffer, the push and sampling method
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, current_state, action, next_state, reward):
        """Saves a transition."""
        if next_state is None:
            next_state = {'pool': [None] * current_state['pool'].size()[0], 'subset': None}
        for cs, a, ns, r in zip(current_state['pool'], action, next_state['pool'], reward):
            if len(self.memory) < self.capacity:
                self.memory.append(None)

            self.memory[self.position] = None
            self.memory[self.position] = Transition(cs.unsqueeze(0), current_state['subset'].unsqueeze(0), a,
                                                    ns.unsqueeze(0) if ns is not None else ns,
                                                    next_state['subset'].unsqueeze(0) if next_state[
                                                                                             'subset'] is not None else
                                                    next_state['subset'], r)
            self.position = (self.position + 1) % self.capacity

            del (cs)
            del (a)
            del (ns)
            del (r)
        del (current_state)
        del (action)
        del (next_state)
        del (reward)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
