from collections import defaultdict
from copy import deepcopy
from typing import Tuple, Union

import numpy as np
import torch

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import (
    concat_items,
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

    def __init__(
        self,
        loss: str,
        device: Union[str, torch.device] = "cpu",
        float_type: Union[int, torch.dtype] = 32,
    ):
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_type)
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
        self.all_possible_parents = []
        self.all_possible_parents_actions = []
        self.steps = []
        self.is_processed = False
        self.states_policy = None
        self.parents_policy = None
        self.trajectory_indices = None

    def __len__(self):
        return len(self.states)

    def add_to_batch(
        self,
        envs: List[GFlowNetEnv],
        actions: List[Tuple],
        valids: List[bool],
        train: bool = True,
    ):
        """
        Adds information from a list of environments and actions to the batch after
        performing steps in the envs. If train is True, it adds all the variables
        required for computing the loss specified by self.loss. Otherwise, it stores
        only states, env_id, step number (everything needed when sampling a trajectory
        at inference)

        Args
        ----
        envs : list
            A list of environments (GFlowNetEnv).

        actions : list
            A list of actions attempted or performed on the envs.

        valids : list
            A list of boolean values indicated whether the actions were valid.
        """
        if self.is_processed:
            raise Exception("Cannot append to the processed batch")
        for env, action, valid in zip(envs, actions, valids):
            self.envs.update({env.id: env})
            if not valid:
                continue
            if train:
                self.states.append(deepcopy(env.state))
                self.actions.append(action)
                self.env_ids.append(env.id)
                self.done.append(env.done)
                self.steps.append(env.n_actions)
                self.masks_invalid_actions_forward.append(
                    env.get_mask_invalid_actions_forward()
                )
                if self.loss == "flowmatch":
                    parents, parents_a = env.get_parents(action=action)
                    assert (
                        action in parents_a
                    ), f"""
                    Sampled action is not in the list of valid actions from parents.
                    \nState:\n{env.state}\nAction:\n{action}
                    """
                    self.all_possible_parents.append(parents)
                    self.all_possible_parents_actions.append(parents_a)
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
        - computes trajectory indicies (indecies of the states in self.states
          corresponding to each trajectory)
        - if needed, computes states and parents in policy formats (stored in
          self.states_policy, self.parents_policy)
        """
        self.env_ids = tlong(self.env_ids, device=self.device)
        self._process_states()
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
                self.all_possible_parents_state_idx = tlong(
                    sum(
                        [
                            [idx] * len(p)
                            for idx, p in enumerate(self.all_possible_parents)
                        ],
                        [],
                    ),
                    device=self.device,
                )
                self.all_possible_parents_actions = torch.cat(
                    [
                        tfloat(x, device=self.device, float_type=self.float)
                        for x in self.all_possible_parents_actions
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
        Converts self.statees from list to torch tensor, computes states in the policy
        format (stored in self.states_policy)
        """
        self.states = tfloat(self.states, device=self.device, float_type=self.float)
        self.states_policy = self.states2policy()

    def states2policy(self, states=None, env_ids=None):
        """
        Converts states from a list of states in gflownet format to a tensor of states
        in policy format
        states: list of gflownet states,
        env_ids: list of env ids indicating which env corresponds to each state in
        states list

        Returns: torch tensor of stattes in policy format
        """
        if states is None:
            states = self.states
            env_ids = self.env_ids
        elif env_ids is None:
            # if states are provided, env_ids should be provided too
            raise Exception(
                """
                env_ids must be provided to the batch for converting provided states to
                the policy format
                """
            )
        env = self._get_first_env()
        if env.conditional:
            states_policy = torch.zeros(
                (states.shape[0], env.policy_input_dim),
                device=self.device,
                dtype=self.float,
            )
            for env_id in torch.unique(env_ids):
                states_policy[env_ids == env_id] = self.envs[
                    env_id.item()
                ].statetorch2policy(states[env_ids == env_id])
        else:
            states_policy = env.statetorch2policy(states)
        return states_policy

    def states2proxy(self, states=None, env_ids=None):
        """
        Converts states from a list of states in gflownet format to a tensor of states
        in proxy format
        states: list of gflownet states,
        env_ids: list of env ids indicating which env corresponds to each state in
        states list

        Returns: list of stattes in proxy format
        """
        if states is None:
            states = self.states
            env_ids = self.env_ids
        elif env_ids is None:
            # if states are provided, env_ids should be provided too
            raise Exception(
                """
                env_ids must be provided to the batch for converting provided states to
                the policy format
                """
            )
        env = self._get_first_env()
        if env.conditional:
            states_proxy = []
            index = torch.arange(states.shape[0], device=self.device)
            perm_index = []
            for env_id in torch.unique(env_ids):
                states_proxy.append(
                    self.envs[env_id.item()].statetorch2proxy(states[env_ids == env_id])
                )
                perm_index.append(index[env_ids == env_id])
            perm_index = torch.cat(perm_index)
            # reverse permutation to make it index the states_proxy array
            index[perm_index] = index.clone()
            states_proxy = concat_items(states_proxy, index)
        else:
            states_proxy = env.statetorch2proxy(states)
        return states_proxy

    def _process_parents(self):
        """
        Prepares self.parents (gflownet format) and self.parents_policy (policy format)
        as torch tensors.
        Different behaviour depending on self.loss:
            - for flowmatch, parents contain all the possible parents for each state,
              so this tensor is bigger than self.state (all the parents are aligned
              along the zero dimension)
            - for trajectorybalance, parents contain only one parent for each state
              which was its parent in the trajectory
        """
        if self.loss == "flowmatch":
            all_possible_parents_policy = []
            for par, env_id in zip(self.all_possible_parents, self.env_ids):
                all_possible_parents_policy.append(
                    tfloat(
                        self.envs[env_id.item()].statebatch2policy(par),
                        device=self.device,
                        float_type=self.float,
                    )
                )
            self.all_possible_parents_policy = torch.cat(all_possible_parents_policy)
            self.all_possible_parents = torch.cat(
                [
                    tfloat(par, device=self.device, float_type=self.float)
                    for par in self.all_possible_parents
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
        self.all_possible_parents += another_batch.all_possible_parents
        self.all_possible_parents_actions += another_batch.all_possible_parents_actions
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
        Computes rewards for self.states using proxy from one of the self.envs
        """
        states_proxy_done = self.states2proxy(
            states=self.states[self.done], env_ids=self.env_ids[self.done]
        )
        env = self._get_first_env()
        rewards = torch.zeros(self.done.shape[0], dtype=self.float, device=self.device)
        if self.states[self.done, :].shape[0] > 0:
            rewards[self.done] = env.proxy2reward(env.proxy(states_proxy_done))
        return rewards

    def _get_first_env(self):
        return self.envs[next(iter(self.envs))]
