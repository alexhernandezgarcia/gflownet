from gflownet.envs.sequences.base import SequenceBase
from gflownet.utils.common import copy, set_device, tlong

"""
Bidirectional, right-only, left-only appending implementations
left: [0, x]
right: [1, x]
eos: [-1, -1]
"""


class BidirectionalSequence(SequenceBase):

    def __init__(self, right_only=False, left_only=False, **kwargs,):
        if right_only and left_only:
            raise ValueError("Impossible to have right only and left only appending at the same time!")
        self.right_only = right_only
        self.left_only = left_only

        super().__init__(**kwargs)

        #self.eos = (-1, self.eos_idx)

    def get_action_space(self):
        self.eos = (-1, -1)
        right_actions = []
        left_actions = []

        for token in self.tokens:
            right_actions.append((1, self.token2idx[token]))
            left_actions.append((0, self.token2idx[token]))

        if self.right_only:
            return right_actions + [self.eos]
        if self.left_only:
            return left_actions + [self.eos]

        return right_actions + left_actions + [self.eos]

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        # 1. if done, all is invalid (True)
        # 2. eos is invalid when < min_length
        # 3. all actions are valid when < max_length
        # when = max_length, only eos is valid
        # Changes with bidirectional: 
        # 1. at source, only right is allowed
        # 2. 
        state = self._get_state(state)
        done = self._get_done(done)
        mask = super().get_mask_invalid_actions_forward(state, done)

        if self.is_source(state):
            if not self.right_only and not self.left_only:
                for i in range(self.action_space_dim):
                    if self.action_space[i][0] != 1:
                        mask[i] = True

        return mask

    def get_parents(self, state=None, done=None, action=None):
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [state], [self.eos]
        if self.equal(state, self.source):
            return [], []
        pos_last_token_right = self._get_seq_length(state) - 1
        pos_last_token_left = 0
        parent_right = copy(state)
        parent_left = copy(state)
        parent_right[pos_last_token_right] = self.pad_idx
        p_action_right = (1, int(state[pos_last_token_right]))

        parent_left[:-1] = state[1:]
        parent_left[-1] = self.pad_idx
        p_action_left = (0, int(state[0]))

        if self.right_only:
            return [parent_right], [p_action_right]
        if self.left_only:
            return [parent_left], [p_action_left]

        if pos_last_token_right == 0:
            return [parent_right], [p_action_right]

        return [parent_right, parent_left], [p_action_right, p_action_left]

    def step(self, action, skip_mask_check=False):
        do_step, self.state, action = self._pre_step(
                action, skip_mask_check=(skip_mask_check or self.skip_mask_check)
                )
        if not do_step:
            return self.state, action, False

        valid = True
        self.n_actions += 1
        if action == self.eos:
            self.done = True
            return self.state, action, valid

        if action[0] == 0:
            self.state[1:] = self.state[:-1].clone()
            self.state[0] = action[1]
            #self.state = [action[1]] + self.state
        else:
            self.state[self._get_seq_length()] = action[1]
        
        return self.state, action, valid


