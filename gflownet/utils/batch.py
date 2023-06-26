from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from gflownet.utils.common import (
    set_device,
    set_float_precision,
    tbool,
    tfloat,
    tint,
    tlong,
)


class Batch:
    """
    Class to handle GFlowNet batches.

    loss: string
        String identifier of the GFlowNet loss.

    device: str or torch.device
        torch.device or string indicating the device to use ("cpu" or "cuda")

    float_type: torch.dtype or int
        One of float torch.dtype or an int indicating the float precision (16, 32 or
        64).

    Important note: one env should correspond to only one trajectory, all env_id should
    be unique.
    """

    def __init__(self, loss: str, device: Any = "cpu", float_type: Any = 32):
        # Device
        if isinstance(device, str):
            device = set_device(device)
        self.device = device
        # Float precision
        if isinstance(float_type, int):
            float_type = set_float_precision(float_type)
        self.float = float_type
        # Loss
        self.loss = loss
        # Initialize empty batch variables
        self.envs = dict()
        self.states = []
        self.actions = []
        self.done = []
        self.env_ids = []
        self.masks_invalid_actions_forward = []
        self.masks_invalid_actions_backward = []
        self.parents = []
        self.parents_actions = []
        self.steps = []
        self.is_processed = False
        self.states_policy = None
        self.parents_policy = None
        self.trajectory_indices = None

    def __len__(self):
        return len(self.states)

    def add_to_batch(
        self, envs, actions, valids, masks_invalid_actions_forward=None, train=True
    ):
        """
        Adds information about current state of env to the batch after performing step in this env.
        If train, updates internal lists with values required for computing self.loss.
        Otherwise, stores only states, env_id, step number (everything needed when sampling a trajectory at inference)
        """
        if self.is_processed:
            raise Exception("Cannot append to the processed batch")

        # Sample masks of invalid actions if required and none are provided
        if masks_invalid_actions_forward is None:
            if train:
                masks_invalid_actions_forward = [
                    env.get_mask_invalid_actions_forward() for env in envs
                ]
            else:
                masks_invalid_actions_forward = [None] * len(envs)

        # Add data samples to the batch
        for sample_data in zip(envs, actions, valids, masks_invalid_actions_forward):
            env, action, valid, mask_forward = sample_data
            self.envs.update({env.id: env})
            if not valid:
                continue
            if train:
                self.states.append(deepcopy(env.state))
                self.actions.append(action)
                self.env_ids.append(env.id)
                self.done.append(env.done)
                self.steps.append(env.n_actions)
                self.masks_invalid_actions_forward.append(mask_forward)
                if self.loss == "flowmatch":
                    parents, parents_a = env.get_parents(action=action)
                    assert (
                        action in parents_a
                    ), f"""
                    Sampled action is not in the list of valid actions from parents.
                    \nState:\n{env.state}\nAction:\n{action}
                    """
                    self.parents.append(parents)
                    self.parents_actions.append(parents_a)
                if self.loss == "trajectorybalance":
                    self.masks_invalid_actions_backward.append(
                        env.get_mask_invalid_actions_backward(
                            env.state, env.done, [action]
                        )
                    )
            else:
                if env.done:
                    self.states.append(env.state)
                    self.env_ids.append(env.id)
                    self.steps.append(env.n_actions)

    def process_batch(self):
        """
        Process internal lists to more convenient formats:
        - converts and stacks list to a single torch tensor
        - computes trajectory indicies (indecies of the states in self.states corresponding to each trajectory)
        - if needed, computes states and parents in policy formats (stored in self.states_policy, self.parents_policy)
        """
        self._process_states()
        self.env_ids = tlong(self.env_ids, device=self.device)
        self.steps = tlong(self.steps, device=self.device)
        self._process_trajectory_indices()
        # process other variables, if we are in the train mode and recorded them
        if len(self.actions) > 0:
            self.actions = tfloat(
                self.actions, device=self.device, float_type=self.float
            )
            self.done = tbool(self.done, device=self.device)
            self.masks_invalid_actions_forward = tbool(
                self.masks_invalid_actions_forward, device=self.device
            )
            if self.loss == "flowmatch":
                self.parents_state_idx = tlong(
                    sum([[idx] * len(p) for idx, p in enumerate(self.parents)], []),
                    device=self.device,
                )
                self.parents_actions = torch.cat(
                    [
                        tfloat(x, device=self.device, float_type=self.float)
                        for x in self.parents_actions
                    ]
                )
            elif self.loss == "trajectorybalance":
                self.masks_invalid_actions_backward = tbool(
                    self.masks_invalid_actions_backward, device=self.device
                )
            self._process_parents()
        self.is_processed = True

    def _process_states(self):
        """
        Converts self.statees from list to torch tensor, computes states in the policy format
        (stored in self.states_policy)
        """
        self.states_policy = self.states2policy()
        self.states = tfloat(self.states, device=self.device, float_type=self.float)

    def states2policy(self, states=None, env_ids=None):
        """
        Converts states from a list of states in gflownet fromat to a tensor of states in policy format
        states: list of gflownet states,
        env_ids: list of env ids indicating which env corresponds to each state in states list

        Returns: torch tensor of stattes in policy format
        """
        if states is None:
            states = self.states
            env_ids = self.env_ids
        elif env_ids is None:
            # if states are provided, env_ids should be provided too
            raise Exception(
                "env_ids must be provided to the batch for converting provided states to the policy format"
            )
        states_policy = []
        for state, env_id in zip(states, env_ids):
            states_policy.append(self.envs[env_id].state2policy(state))
        return tfloat(states_policy, device=self.device, float_type=self.float)

    def _process_parents(self):
        """
        Prepares self.parents (gflownet format) and self.parents_policy (policy format) as torch tensors.
        Different behaviour depending on self.loss:
            - for flowmatch, parents contain all the possible parents for each state,
            so this tensor is bigger than self.state (all the parents are alinged the zero dimension, )
            - for trajectorybalance, parents contain only one parent for each state which was its parent in the trajectory
        """
        if self.loss == "flowmatch":
            parents_policy = []
            for par, env_id in zip(self.parents, self.env_ids):
                parents_policy.append(
                    tfloat(
                        self.envs[env_id.item()].statebatch2policy(par),
                        device=self.device,
                        float_type=self.float,
                    )
                )
            self.parents_policy = torch.cat(parents_policy)
            # import ipdb; ipdb.set_trace()
            self.parents = torch.cat(
                [
                    tfloat(par, device=self.device, float_type=self.float)
                    for par in self.parents
                ]
            )
        elif self.loss == "trajectorybalance":
            assert self.trajectory_indices is not None
            parents_policy = torch.zeros_like(self.states_policy)
            parents = torch.zeros_like(self.states)
            for env_id, traj in self.trajectory_indices.items():
                # parent is source
                parents_policy[traj[0]] = tfloat(
                    self.envs[env_id].state2policy(self.envs[env_id].source),
                    device=self.device,
                    float_type=self.float,
                )
                parents[traj[0]] = tfloat(
                    self.envs[env_id].source,
                    device=self.device,
                    float_type=self.float,
                )
                # parent is not source
                parents_policy[traj[1:]] = self.states_policy[traj[:-1]]
                parents[traj[1:]] = self.states[traj[:-1]]
            self.parents_policy = parents_policy
            self.parents = parents

    def merge(self, another_batch):
        """
        Merges two unprocessed batches
        """
        if self.is_processed or another_batch.is_processed:
            raise Exception("Cannot merge processed batches")
        if self.loss != another_batch.loss:
            raise Exception("Cannot merge batches with different losses")
        self.envs.update(another_batch.envs)
        self.states += another_batch.states
        self.actions += another_batch.actions
        self.done += another_batch.done
        self.env_ids += another_batch.env_ids
        self.masks_invalid_actions_forward += (
            another_batch.masks_invalid_actions_forward
        )
        self.masks_invalid_actions_backward += (
            another_batch.masks_invalid_actions_backward
        )
        self.parents += another_batch.parents
        self.parents_actions += another_batch.parents_actions
        self.steps += another_batch.steps

    def _process_trajectory_indices(self):
        """
        Creates a dict of trajectory indices (key: env_id, value: indecies of the
        states in self.states going in the order from s_1 to s_f. The dict is created
        and stored in the self.trajectory_indices
        """
        trajs = defaultdict(list)
        for idx, (env_id, step) in enumerate(zip(self.env_ids, self.steps)):
            trajs[env_id.item()].append((idx, step))
        trajs = {
            env_id: list(map(lambda x: x[0], sorted(traj, key=lambda x: x[1])))
            for env_id, traj in trajs.items()
        }
        self.trajectory_indices = trajs

    def unpack_terminal_states(self):
        """
        For storing terminal states and trajectory actions in the buffer
        Unpacks the terminating states and trajectories of a batch and converts them
        to Python lists/tuples.
        """
        # TODO: make sure that unpacked states and trajs are sorted by traj_id (like
        # rewards will be)
        if not self.is_processed:
            self.process_batch()
        traj_actions = []
        terminal_states = []
        for traj_idx in self.trajectory_indices.values():
            traj_actions.append(self.actions[traj_idx].tolist())
            terminal_states.append(tuple(self.states[traj_idx[-1]].tolist()))
        traj_actions = [tuple([tuple(a) for a in t]) for t in traj_actions]
        return terminal_states, traj_actions

    def compute_rewards(self):
        """
        Computes rewards for self.states using env.reward_tobatch
        """
        rewards = torch.zeros(len(self.states), device=self.device, dtype=self.float)
        for env_id, env in self.envs.items():
            idx = self.env_ids == env_id
            rewards[idx] = env.reward_torchbatch(self.states[idx], self.done[idx])
        return rewards
