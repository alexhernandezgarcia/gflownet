from gflownet.utils.common import tfloat, tlong, tint, tbool

class Batch:
    def __init__(self, loss, device, float):
        self.loss = loss
        self.device = device
        self.float = float
        self.envs = dict()
        self.state = []
        self.action = []
        self.done = []
        self.env_id = []
        self.mask_invalid_actions_forward = []
        self.mask_invalid_actions_backward = []
        self.parents = []
        self.parents_actions = []
        self.step = []


    def add_to_batch(self, envs, actions, valids, train=True):
        self.needs_update = True
        for env in envs:
            self.envs.update({env.id: env})

        for env, action, valid in zip(envs, actions, valids):
            if valid is False:
                continue 
            if train:
                self.state.append(env.state)
                self.action.append(action)
                self.env_id.append(env.id)
                self.done.append(env.done)
                mask_f = env.get_mask_invalid_actions_forward()
                self.mask_invalid_actions_forward.append(mask_f)
                if loss == 'flowmatch':
                    parents, parents_a = env.get_parents(action=action)
                    self.parents.append(parents)
                    self.parents_actions.append(parents_a)
                if loss == 'trajectorybalance':
                    mask_b = env.get_mask_invalid_actions_backward(
                        env.state, env.done, [action]
                    )
                    self.mask_invalid_actions_backward.append(mask_b)
                    self.step.append(env.n_actions)
            else:
                if env.done:
                    self.state.append(env.state)
                    self.env_id.append(env.id)

    def process_batch(self):
        self._process_states()
        self.action = tfloat(self.action, device=self.device, float=self.float)
        self.done = tbool(self.done, device=self.device)
        self.env_id = tlong(self.env_id, device=self.device)
        self.mask_invalid_actions_forward = tbool(self.mask_invalid_actions_forward, device=self.device)
        if loss == 'flowmatch':
            self.parents_state_idx = tlong(
                sum([[idx] * len(p) for idx, p in enumerate(self.parents)], []), device=self.device
            )
            self.parents_actions = torch.cat([tfloat(x, device=self.device, float=self.float) for x in self.parents_actions])
        elif loss == 'trajectorybalance':
            self.mask_invalid_actions_backward = tbool(self.mask_invalid_actions_backward, device=self.device)
            self.step = tlong(self.step, device=self.device)
        self._process_parents() # should be called after creating self.parents_state_idx

    def _process_states(self):
        states = []
        for state, env_id in zip(self.state, self.env_id):
            states.append(self.envs[env_id].state2policy(state))
        self.state = tfloat(states, device=self.device, float=self.float)

    def _process_parents(self):
        if self.loss == 'flowmatch':
            parents = []
            for par, env_id in zip(self.parents, self.env_id):
                parents.append(tfloat(self.envs[env_id].statebatch2policy(par), device=self.device, float=self.float))
            self.parents = torch.cat(parents)
        elif loss == 'trajectorybalance':
            # TODO find parent for each state in the trajectory
            pass

    

    
    

    






    
