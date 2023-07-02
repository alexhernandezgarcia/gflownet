from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torchtyping import TensorType

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
        self.parents_all = []
        self.parents_actions_all = []
        self.n_actions = []
        self.is_processed = False
        self.states_policy = None
        self.parents_policy = None
        # Dictionary of env_id (traj_id): batch indices of the trajectory
        self.trajectories = {}
        # Trajectory index and state index of each element in the batch
        self.traj_state_indices = []
        self.parents_available = False

    def __len__(self):
        return len(self.states)

    def batch_idx_to_traj_state_idx(batch_idx: int):
        traj_id, state_id = self.traj_state_indices[batch_idx]
        return traj_id, state_id

    def traj_idx_to_batch_indices(traj_idx: int):
        batch_indices = self.trajectories[traj_idx]
        return batch_indices

    def add_to_batch(
        self,
        envs: List[GFlowNetEnv],
        actions: List[Tuple],
        valids: List[bool],
        masks_invalid_actions_forward: Optional[List[List[bool]]] = None,
        train: Optional[bool] = True,
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

        masks_invalid_actions_forward : list
            A list of masks indicating, among all the actions that could have been
            selected for all environments, which ones were invalid. Optional, will be
            computed if not provided.

        train : bool
            A boolean value indicating whether the data to add to the batch will be used
            for training. Optional, default is True.
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
                self.n_actions.append(env.n_actions)
                self.masks_invalid_actions_forward.append(mask_forward)

                if self.loss == "flowmatch":
                    parents, parents_a = env.get_parents(action=action)
                    assert (
                        action in parents_a
                    ), f"""
                    Sampled action is not in the list of valid actions from parents.
                    \nState:\n{env.state}\nAction:\n{action}
                    """
                    self.parents_all.append(parents)
                    self.parents_actions_all.append(parents_a)
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
                    self.n_actions.append(env.n_actions)
            # Add batch index to trajectory
            if env.id not in trajectories:
                self.trajectories.update({env.id: [len(self)]})
            else:
                self.trajectories[env.id].append(len(self))
            # Add trajectory index and state index
            self.traj_state_indices.append((env.id, env.n_actions))

    def process_batch(self):
        """
        Converts internal lists into more convenient formats:
        - converts and stacks lists into a single torch tensor
        - computes trajectory indices (indices of the states in self.states
          corresponding to each trajectory)
        - if needed, converts states and parents into policy formats (stored in
          self.states_policy, self.parents_policy)
        """
        self.env_ids = tlong(self.env_ids, device=self.device)
        self.states, self.states_policy = self._process_states()
        self.n_actions = tlong(self.n_actions, device=self.device)
        self.trajectory_indices = self._process_trajectory_indices()
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
                self.parents_all_state_idx = tlong(
                    sum(
                        [[idx] * len(p) for idx, p in enumerate(self.parents_all)],
                        [],
                    ),
                    device=self.device,
                )
                self.parents_actions_all = tfloat(
                    [a for actions in self.parents_actions_all for a in actions],
                    device=self.device,
                    float_type=self.float,
                )
            elif self.loss == "trajectorybalance":
                self.masks_invalid_actions_backward = tbool(
                    self.masks_invalid_actions_backward, device=self.device
                )
            (
                self.parents,
                self.parents_policy,
                self.parents_all,
                self.parents_all_policy,
            ) = self._process_parents()
        self.is_processed = True

    def get_states(
        self, policy: Optional[bool] = False, force_recompute: Optional[bool] = False
    ) -> Union[List, TensorType["n_states", "..."]]:
        """
        Returns all the states in the batch.

        The states are returned in "policy format" if policy is True, otherwise they
        are returned in "GFlowNet" format (default).

        Args
        ----
        policy : bool
            If True, the policy format of the states is returned. Otherwise, the
            GFlowNet format is returned.

        force_recompute : bool
            If True, the policy states are recomputed even if they are available.
            Ignored if policy is False.

        Returns
        -------
        self.states or self.states_policy : list or torch.tensor
            The set of all states in the batch.
        """
        if policy is False:
            return self.states
        if states_policy is None or force_recompute is True:
            self.states_policy = self.states2policy(self.states, self.env_ids)
        return self.states_policy

    def _process_states(self):
        """
        Convert self.states from a list to a torch tensor and compute states in the policy format.

        Returns
        -------
        states: torch.tensor
            Tensor containing the states converted to a torch tensor.
        states_policy: torch.tensor
            Tensor containing the states converted to the policy format.

        """
        # TODO: check if available
        # TODO: perhaps better, separate states and states2policy
        states = tfloat(self.states, device=self.device, float_type=self.float)
        states_policy = self.states2policy(states, self.env_ids)
        return states, states_policy

    def states2policy(
        self,
        states: Optional[Union[List, TensorType["n_states", "..."]]] = None,
        env_ids: Optional[List[int]] = None,
    ):
        """
        Converts states from a list of states in GFlowNet format to a tensor of states
        in policy format.

        Args
        ----
        states: list or torch.tensor
            States in GFlowNet format.

        env_ids: list
            Ids indicating which env corresponds to each state in states.

        Returns
        -------
        states: torch.tensor
            States in policy format.
        """
        if states is None:
            states = self.states
            env_ids = self.env_ids
        elif env_ids is None:
            # if states are provided, env_ids should be provided too
            raise Exception(
                """
                env_ids must be provided to the batch for converting provided states to
                the policy format.
                """
            )
        env = self._get_first_env()
        if env.conditional:
            states_policy = torch.zeros(
                (len(self), env.policy_input_dim),
                device=self.device,
                dtype=self.float,
            )
            for env_id in torch.unique(env_ids):
                states_policy[env_ids == env_id] = self.envs[
                    env_id.item()
                ].statetorch2policy(states[env_ids == env_id])
            return states_policy
        return env.statebatch2policy(states)

    def states2proxy(
        self,
        states: Optional[Union[List, TensorType["n_states", "..."]]] = None,
        env_ids: Optional[List[int]] = None,
    ):
        """
        Converts states from a list of states in GFlowNet format to a tensor of states
        in proxy format.

        Args
        ----
        states: list or torch.tensor
            States in GFlowNet format.

        env_ids: list
            Ids indicating which env corresponds to each state in states.

        Returns
        -------
        states: torch.tensor
            States in policy format.
        """
        if states is None:
            states = self.states
            env_ids = self.env_ids
        elif env_ids is None:
            # if states are provided, env_ids should be provided too
            raise Exception(
                """
                env_ids must be provided to the batch for converting provided states to
                the proxy format.
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
            # Reverse permutation to make it index the states_proxy array
            index[perm_index] = index.clone()
            states_proxy = concat_items(states_proxy, index)
            return states_proxy
        return env.statetorch2proxy(states)

    def get_parents(
        self, policy: Optional[bool] = False, force_recompute: Optional[bool] = False
    ) -> TensorType["n_states", "..."]:
        """
        Returns the parent (single parent for each state) of all states in the batch.
        The parents are computed, obtaining all necessary components, if they are not
        readily available. Missing components and newly computed components are added
        to the batch (self.component is set).

        The parents are returned in "policy format" if policy is True, otherwise they
        are returned in "GFlowNet" format (default).

        Args
        ----
        policy : bool
            If True, the policy format of parents is returned. Otherwise, the GFlowNet
            format is returned.

        force_recompute : bool
            If True, the parents are recomputed even if they are available.

        Returns
        -------
        self.parents or self.parents_policy : torch.tensor
            The parent of all states in the batch.
        """
        if self.parents_available is False or force_recompute is True:
            self._compute_parents()
        if policy:
            return (self.parents_policy,)
        else:
            return self.parents

    def _compute_parents(self):
        """
        Obtains the parent (single parent for each state) of all states in the batch.
        The parents are computed, obtaining all necessary components, if they are not
        readily available. Missing components and newly computed components are added
        to the batch (self.component is set). The following components are obtained:

        - self.parents: the parent of each state in the batch. It will be the same type
          as self.states (list of lists or tensor)
            Length: n_states
            Shape: [n_states, state_dims]
        - self.parents_policy: the parent of each state in the batch in policy format.
            Shape: [n_states, state_policy_dims]

        self.parents_policy is stored as a torch tensor and self.parents_available is
        set to True.
        """
        self.states_policy = self.get_states(policy=True)
        self.parents_policy = torch.zeros_like(self.states_policy)
        self.parents = []
        indices = []
        # Iterate over the trajectories to obtain the parents from the states
        for traj_id, batch_indices in self.trajectories.items():
            # parent is source
            self.parents.extend(self.envs[traj_id].source)
            self.parents_policy[batch_indices[0]] = tfloat(
                self.envs[traj_id].state2policy(self.envs[traj_id].source),
                device=self.device,
                float_type=self.float,
            )
            # parent is not source
            self.parents.extend(self.states[batch_indices[:-1]])
            self.parents_policy[batch_indices[1:]] = self.states_policy[
                batch_indices[:-1]
            ]
            indices.extend(batch_indices)
        # Sort parents list in the same order as states
        self.parents = self.parents[indices]

    def get_parents_all(
        self, policy: bool = False, force_recompute: bool = False
    ) -> Tuple[
        TensorType["n_parents", "..."],
        TensorType["n_parents", "..."],
        TensorType["n_parents"],
    ]:
        """
        Returns the whole set of parents, their corresponding actions and indices of
        all states in the batch. If the parents are not available
        (self.parents_all_available is False) or if force_recompute is True, then
        self._compute_parents_all() is called to compute the required components.

        The parents are returned in "policy format" if policy is True, otherwise they
        are returned in "GFlowNet" format (default).

        Args
        ----
        policy : bool
            If True, the policy format of parents is returned. Otherwise, the GFlowNet
            format is returned.

        force_recompute : bool
            If True, the parents are recomputed even if they are available.

        Returns
        -------
        self.parents_all or self.parents_all_policy : torch.tensor
            The whole set of parents of all states in the batch.

        self.parents_actions_all : torch.tensor
            The actions corresponding to each parent in self.parents_all or
            self.parents_all_policy, linking them to the corresponding state in the
            trajectory.

        self.parents_all_indices : torch.tensor
            The state index corresponding to each parent in self.parents_all or
            self.parents_all_policy, linking them to the corresponding state in the
            batch.
        """
        if self.parents_all_available is False or force_recompute is True:
            self._compute_parents_all()
        if policy:
            return (
                self.parents_all_policy,
                self.parents_actions_all,
                self.parents_all_indices,
            )
        else:
            return self.parents_all, self.parents_actions_all, self.parents_all_indices

    def _compute_parents_all(self):
        """
        Obtains the whole set of parents all states in the batch. The parents are
        computed via env.get_parents(). The following components are obtained:

        - self.parents_all: all the parents of all states in the batch.
            Shape: [n_parents, state_dims]
        - self.parents_actions_all: the actions corresponding to the transition from
          each parent in self.parents_all to its corresponding state in the batch.
            Shape: [n_parents, action_dim]
        - self.parents_all_indices: the indices corresponding to the state in the batch
          of which each parent in self.parents_all is a parent.
            Shape: [n_parents]
        - self.parents_all_policy: self.parents_all in policy format.
            Shape: [n_parents, state_policy_dims]

        All the above components are stored as torch tensors and
        self.parents_all_available is set to True.
        """
        # Iterate over the trajectories to obtain all parents
        self.parents_all = []
        self.parents_actions_all = []
        self.parents_all_indices = []
        self.parents_all_policy = []
        for idx, (traj_idx, _) in self.traj_state_indices:
            state = self.states[idx]
            done = self.done[idx]
            action = self.actions[idx]
            parents, parents_a = self.envs[traj_id].get_parents(
                state=state,
                done=done,
                action=action,
            )
            assert (
                action in parents_a
            ), f"""
            Sampled action is not in the list of valid actions from parents.
            \nState:\n{state}\nAction:\n{action}
            """
            self.parents_all.append(parents)
            self.parents_actions_all.append(parents_a)
            self.parents_all_indices.append([[idx] * len(parents)])
            self.parents_all_policy.append(
                tfloat(
                    self.envs[traj_id].statebatch2policy(parents),
                    device=self.device,
                    float_type=self.float,
                )
            )
        # Flatten parents lists and convert to tensors
        self.parents_all = tfloat(
            [p for parents in self.parents_all for p in parents],
            device=self.device,
            float_type=self.float,
        )
        self.parents_actions_all = tfloat(
            [a for actions in self.parents_actions_all for a in actions],
            device=self.device,
            float_type=self.float,
        )
        self.parents_all_indices = tlong(
            [idx for indices in self.parents_all_indices for idx in indices],
            device=self.device,
        )
        self.parents_all_policy = torch.cat(parents_all_policy)
        self.parents_all_available = True

    def _process_parents(self):
        """
        Process parents for the given loss type.

        Returns
        -------
        parents: torch.tensor
            Tensor of parent states.
        parents_policy: torch.tensor
            Tensor of parent states converted to policy format.
        parents_all: torch.tensor
            Tensor of all parent states.
        parents_all_policy: torch.tensor
            Tensor of all parent states converted to policy format.
        """
        parents = []
        parents_policy = []
        parents_all = []
        parents_all_policy = []
        if self.loss == "flowmatch":
            for par, env_id in zip(self.parents_all, self.env_ids):
                parents_all_policy.append(
                    tfloat(
                        self.envs[env_id.item()].statebatch2policy(par),
                        device=self.device,
                        float_type=self.float,
                    )
                )
            parents_all_policy = torch.cat(parents_all_policy)
            parents_all = tfloat(
                [p for parents in self.parents_all for p in parents],
                device=self.device,
                float_type=self.float,
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
        return parents, parents_policy, parents_all, parents_all_policy

    # TODO: handle mix of backward and forward trajectories
    # TODO: opportunity to improve efficiency by caching. Note that
    # env.get_masks_invalid_actions_backward() may be expensive because it calls
    # env.get_parents().
    def get_masks_backward(
        self,
        force_recompute: bool = False,
    ) -> TensorType["n_states", "action_space_dim"]:
        """
        Computes (and returns) the backward mask of invalid actions of all states in the
        batch, by calling env.get_mask_invalid_actions_backward().

        Args
        ----
        force_recompute : bool
            If True, the masks are recomputed even if they are available.

        Returns
        -------
        self.masks_invalid_actions_backward : torch.tensor
            The backward mask of all states in the batch.
        """
        if self.masks_backward_available is True and force_recompute is False:
            return self.masks_invalid_actions_backward
        # Iterate over the trajectories to compute all backward masks
        self.masks_invalid_actions_backward = []
        for idx, (traj_idx, _) in self.traj_state_indices:
            state = self.states[idx]
            done = self.done[idx]
            action = self.actions[idx]
            self.masks_invalid_actions_backward.append(
                self.envs[traj_idx].get_mask_invalid_actions_backward(
                    state, done, [action]
                )
            )
        # Make tensor
        self.masks_invalid_actions_backward = tbool(
            self.masks_invalid_actions_backward, device=self.device
        )
        self.masks_backward_available = True
        return self.masks_invalid_actions_backward

    def merge(self, another_batch):
        """
        Merges two unprocessed batches.
        """
        if self.is_processed or another_batch.is_processed:
            raise Exception("Cannot merge processed batches.")
        if self.loss != another_batch.loss:
            raise Exception("Cannot merge batches with different losses.")
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
        self.parents_all += another_batch.parents_all
        self.parents_actions_all += another_batch.parents_actions_all
        self.n_actions += another_batch.n_actions

    def _process_trajectory_indices(self):
        """
        Obtain the indices in the batch that correspond to each environment.  Creates a
        dictionary of trajectory indices (key: env_id, value: indices of the states in
        self.states ordered from s_1 to s_f).

        Returns
        -------
        trajs: dict
            Dictionary containing trajectory indices for each environment.
        """
        # Check if self.trajectory_indices is already computed and valid
        if self.trajectory_indices is not None:
            n_indices = sum([len(v) for v in self.trajectory_indices.values])
            if n_indices == len(self):
                return self.trajectory_indices
        # Obtain trajectory indices since they are not computed yet
        trajs = defaultdict(list)
        for idx, (env_id, step) in enumerate(zip(self.env_ids, self.n_actions)):
            trajs[env_id.item()].append((idx, step))
        trajs = {
            env_id: list(map(lambda x: x[0], sorted(traj, key=lambda x: x[1])))
            for env_id, traj in trajs.items()
        }
        return trajs

    # TODO: rethink and re-implement. Outputs should not be tuples. It needs to be
    # refactored together with the buffer.
    # TODO: docstring
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
        terminal_states = []
        traj_actions = []
        for traj_idx in self.trajectory_indices.values():
            traj_actions.append(self.actions[traj_idx].tolist())
            terminal_states.append(tuple(self.states[traj_idx[-1]].tolist()))
        traj_actions = [tuple([tuple(a) for a in t]) for t in traj_actions]
        return terminal_states, traj_actions

    def compute_rewards(self):
        """
        Computes rewards for self.states using proxy from one of the self.envs

        Returns
        -------
        rewards: torch.tensor
            Tensor of rewards.
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
