from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.proxy.base import Proxy
from gflownet.utils.common import (
    concat_items,
    copy,
    extend,
    set_device,
    set_float_precision,
    tbool,
    tfloat,
    tlong,
)


class Batch:
    """
    Class to handle GFlowNet batches.

    Important note: one env should correspond to only one trajectory, all env_id should
    be unique.

    Note: self.state_indices start from index 1 to indicate that index 0 would correspond
    to the source state, but the latter is not stored in the batch for each trajectory.
    This implies that one has to be careful when indexing the list of batch_indices in
    self.trajectories by using self.state_indices. For example, the batch index of
    state state_idx of trajectory traj_idx is self.trajectories[traj_idx][state_idx-1]
    (not self.trajectories[traj_idx][state_idx]).
    """

    def __init__(
        self,
        env: Optional[GFlowNetEnv] = None,
        proxy: Optional[Proxy] = None,
        device: Union[str, torch.device] = "cpu",
        float_type: Union[int, torch.dtype] = 32,
    ):
        """
        Arguments
        ---------
        env : GFlowNetEnv
            An instance of the environment that will be used to form the batch.
        proxy : Proxy
            An instance of a GFlowNet proxy that will be used to compute proxy values
            and rewards.
        device : str or torch.device
            torch.device or string indicating the device to use ("cpu" or "cuda")
        float_type : torch.dtype or int
            One of float torch.dtype or an int indicating the float precision (16, 32
            or 64).
        """
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_type)
        # Generic environment, properties and dictionary of state and forward mask of
        # source (as tensor)
        if env is not None:
            self.set_env(env)
        else:
            self.env = None
            self.source = None
            self.conditional = None
            self.continuous = None
        # Proxy
        self.proxy = proxy
        # Initialize batch size 0
        self.size = 0
        # Initialize empty batch variables
        # TODO: make single ordered dictionary of dictionaries
        self.envs = OrderedDict()
        self.trajectories = OrderedDict()
        self.is_backward = OrderedDict()
        self.traj_indices = []
        # TODO: state_indices is currently unused, it is redundant and inconsistent
        # between forward and backward trajectories. We may want to remove it.
        self.state_indices = []
        self.states = []
        self.actions = []
        self.done = []
        self.masks_invalid_actions_forward = []
        self.masks_invalid_actions_backward = []
        self.parents = []
        self.parents_all = []
        self.parents_actions_all = []
        self.n_actions = []
        self.states_policy = None
        self.parents_policy = None
        # Flags for available items
        self._parents_available = False
        self._parents_policy_available = False
        self._parents_all_available = False
        self._masks_forward_available = False
        self._masks_backward_available = False
        self._rewards_available = False
        self._rewards_parents_available = False
        self._rewards_source_available = False
        self._logrewards_available = False
        self._logrewards_parents_available = False
        self._logrewards_source_available = False
        self._proxy_values_available = False

    def __len__(self):
        return self.size

    def batch_idx_to_traj_state_idx(self, batch_idx: int):
        traj_idx = self.traj_indices[batch_idx]
        state_idx = self.state_indices[batch_idx]
        return traj_idx, state_id

    def traj_idx_to_batch_indices(self, traj_idx: int):
        batch_indices = self.trajectories[traj_idx]
        return batch_indices

    def traj_state_idx_to_batch_idx(self, traj_idx: int, state_idx: int):
        batch_idx = self.trajectories[traj_idx][state_idx]
        return batch_idx

    def traj_idx_action_idx_to_batch_idx(
        self, traj_idx: int, action_idx: int, backward: bool
    ):
        if traj_idx not in self.trajectories:
            return None
        if backward:
            if action_idx >= len(self.trajectories[traj_idx]):
                return None
            return self.trajectories[traj_idx][::-1][action_idx]
        if action_idx > len(self.trajectories[traj_idx]):
            return None
        return self.trajectories[traj_idx][action_idx - 1]

    def idx2state_idx(self, idx: int):
        return self.trajectories[self.traj_indices[idx]].index(idx)

    def rewards_available(self, log: bool = False) -> bool:
        """
        Returns True if the (log)rewards are available.

        Parameters
        ----------
        log : bool
            If True, check self._logrewards_available. Otherwise (default), check
            self._rewards_available.

        Returns
        -------
        bool
            True if the (log)rewards are available, False otherwise.
        """
        if log:
            return self._logrewards_available
        else:
            return self._rewards_available

    def rewards_parents_available(self, log: bool = False) -> bool:
        """
        Returns True if the (log)rewards of the parents are available.

        Parameters
        ----------
        log : bool
            If True, check self._logrewards_parents_available. Otherwise (default),
            check self._rewards_parents_available.

        Returns
        -------
        bool
            True if the (log)rewards of the parents are available, False otherwise.
        """
        if log:
            return self._logrewards_parents_available
        else:
            return self._rewards_parents_available

    def rewards_source_available(self, log: bool = False) -> bool:
        """
        Returns True if the (log)rewards of the source are available.

        Parameters
        ----------
        log : bool
            If True, check self._logrewards_source_available. Otherwise (default),
            check self._rewards_source_available.

        Returns
        -------
        bool
            True if the (log)rewards of the source are available, False otherwise.
        """
        if log:
            return self._logrewards_source_available
        else:
            return self._rewards_source_available

    def set_env(self, env: GFlowNetEnv):
        """
        Sets the generic environment passed as an argument and initializes the
        environment-dependent properties.
        """
        self.env = env.copy().reset()
        self.source = {
            "state": self.env.source,
            "mask_forward": tbool(
                self.env.get_mask_invalid_actions_forward(), device=self.device
            ),
        }
        self.conditional = self.env.conditional
        self.continuous = self.env.continuous

    def set_proxy(self, proxy: Proxy):
        """
        Sets the proxy, used to compute rewards from a batch of states.
        """
        self.proxy = proxy

    def add_to_batch(
        self,
        envs: List[GFlowNetEnv],
        actions: List[Tuple],
        valids: List[bool],
        backward: Optional[bool] = False,
        train: Optional[bool] = True,
    ):
        """
        Adds information from a list of environments and actions to the batch after
        performing steps in the envs. If train is False, only the variables of
        terminating states are stored.

        Args
        ----
        envs : list
            A list of environments (GFlowNetEnv).

        actions : list
            A list of actions attempted or performed on the envs.

        valids : list
            A list of boolean values indicated whether the actions were valid.

        backward : bool
            A boolean value indicating whether the action was sampled backward (False
            by default). If True, the behavior is slightly different so as to match
            what is stored in forward sampling:
                - If it is the first state in the trajectory (action from a done
                  state/env), then done is stored as True, instead of taking env.done
                  which will be False after having performed the step.
                - If it is not the first state in the trajectory, the stored state will
                  be the previous one in the trajectory, to match the state-action
                  stored in forward sampling and the convention that the source state
                  is not stored, but the terminating state is repeated with action eos.

        train : bool
            A boolean value indicating whether the data to add to the batch will be used
            for training. Optional, default is True.
        """
        # TODO: do we need this?
        if self.continuous is None:
            self.continuous = envs[0].continuous

        # Add data samples to the batch
        for env, action, valid in zip(envs, actions, valids):
            if train is False and env.done is False:
                continue
            if not valid:
                continue
            # Add env to dictionary
            if env.id not in self.envs:
                self.envs.update({env.id: env})
            # Add batch index to trajectory
            if env.id not in self.trajectories:
                self.trajectories.update({env.id: [len(self)]})
            else:
                if backward:
                    self.trajectories[env.id].insert(0, len(self))
                else:
                    self.trajectories[env.id].append(len(self))
            # Set whether trajectory is backward
            if env.id not in self.is_backward:
                self.is_backward.update({env.id: backward})
            # Add trajectory index and state index
            self.traj_indices.append(env.id)
            self.state_indices.append(env.n_actions)
            # Add states, parents, actions, done and masks
            self.actions.append(action)
            if backward:
                self.parents.append(copy(env.state))
                if len(self.trajectories[env.id]) == 1:
                    self.states.append(copy(env.state))
                    self.done.append(True)
                else:
                    self.states.append(copy(self.parents[self.trajectories[env.id][1]]))
                    self.done.append(env.done)
            else:
                self.states.append(copy(env.state))
                self.done.append(env.done)
                if len(self.trajectories[env.id]) == 1:
                    self.parents.append(copy(self.source["state"]))
                else:
                    self.parents.append(
                        copy(self.states[self.trajectories[env.id][-2]])
                    )
            # Set masks to None
            self.masks_invalid_actions_forward.append(None)
            self.masks_invalid_actions_backward.append(None)
            # Increment size of batch
            self.size += 1
        # Other variables are not available after new items were added to the batch
        self._masks_forward_available = False
        self._masks_backward_available = False
        self._parents_policy_available = False
        self._parents_all_available = False
        self._rewards_available = False
        self._logrewards_available = False

    def get_n_trajectories(self) -> int:
        """
        Returns the number of trajectories in the batch.

        Returns
        -------
        The number of trajectories in the batch (int).
        """
        return len(self.trajectories)

    def get_unique_trajectory_indices(self) -> List:
        """
        Returns the unique trajectory indices as the keys of self.trajectories, which
        is an OrderedDict, as a list.
        """
        return list(self.trajectories.keys())

    def get_trajectory_indices(
        self, consecutive: bool = False, return_mapping_dict: bool = False
    ) -> TensorType["n_states", int]:
        """
        Returns the trajectory index of all elements in the batch as a long int torch
        tensor.

        Args
        ----
        consecutive : bool
            If True, the trajectory indices are mapped to consecutive indices starting
            from 0, in the order of the OrderedDict self.trajectory.keys(). If False
            (default), the trajectory indices are returned as they are.

        return_mapping_dict : bool
            If True, the dictionary mapping actual_index: consecutive_index is returned
            as a second argument. Ignored if consecutive is False.

        Returns
        -------
        traj_indices : torch.tensor
            self.traj_indices as a long int torch tensor.

        traj_index_to_consecutive_dict : dict
            A dictionary mapping the actual trajectory indices in the Batch to the
            consecutive indices. Ommited if return_mapping_dict is False (default).
        """
        if consecutive:
            traj_index_to_consecutive_dict = {
                traj_idx: consecutive
                for consecutive, traj_idx in enumerate(self.trajectories)
            }
            traj_indices = list(
                map(lambda x: traj_index_to_consecutive_dict[x], self.traj_indices)
            )
        else:
            traj_indices = self.traj_indices
        if return_mapping_dict and consecutive:
            return (
                tlong(traj_indices, device=self.device),
                traj_index_to_consecutive_dict,
            )
        else:
            return tlong(traj_indices, device=self.device)

    def get_state_indices(self) -> TensorType["n_states", int]:
        """
        Returns the state index of all elements in the batch as a long int torch
        tensor.

        Returns
        -------
        state_indices : torch.tensor
            self.state_indices as a long int torch tensor.
        """
        return tlong(self.state_indices, device=self.device)

    def get_states(
        self,
        policy: Optional[bool] = False,
        proxy: Optional[bool] = False,
        force_recompute: Optional[bool] = False,
    ) -> Union[TensorType["n_states", "..."], npt.NDArray[np.float32], List]:
        """
        Returns all the states in the batch.

        The states are returned in "policy format" if policy is True, in "proxy format"
        if proxy is True and otherwise they are returned in "GFlowNet" format by
        default. An error is raised if both policy and proxy are True.

        Args
        ----
        policy : bool
            If True, the policy format of the states is returned and self.states_policy
            is updated if not available yet or if force_recompute is True.

        proxy : bool
            If True, the proxy format of the states is returned. States in proxy format
            are not stored.

        force_recompute : bool
            If True, the policy states are recomputed even if they are available.
            Ignored if policy is False.

        Returns
        -------
        self.states or self.states_policy or self.states2proxy(self.states) : list or
        torch.tensor or ndarray
            The set of all states in the batch.
        """
        if policy is True and proxy is True:
            raise ValueError(
                "Ambiguous request! Only one of policy or proxy can be True."
            )
        if policy is True:
            if self.states_policy is None or force_recompute is True:
                self.states_policy = self.states2policy()
            return self.states_policy
        if proxy is True:
            return self.states2proxy()
        return self.states

    def states2policy(
        self,
        states: Optional[Union[List[List], List[TensorType["n_states", "..."]]]] = None,
        traj_indices: Optional[Union[List, TensorType["n_states"]]] = None,
    ) -> TensorType["n_states", "state_policy_dims"]:
        """
        Converts states from a list of states in GFlowNet format to a tensor of states
        in policy format.

        Args
        ----
        states: list
            List of states in GFlowNet format.

        traj_indices: list or torch.tensor
            Ids indicating which env corresponds to each state in states. It is only
            used if the environments are conditional to call state2policy from the
            right environment. Ignored if self.conditional is False.

        Returns
        -------
        states: torch.tensor
            States in policy format.
        """
        # If traj_indices is not None and self.conditional is True, then both states
        # and traj_indices must be the same type and have the same length.
        if traj_indices is not None and self.conditional is True:
            assert type(states) == type(traj_indices)
            assert len(states) == len(traj_indices)
        if states is None:
            states = self.states
            traj_indices = self.traj_indices
        # TODO: will env.policy_input_dim be the same for all envs if conditional?
        if self.conditional:
            states_policy = torch.zeros(
                (len(states), self.env.policy_input_dim),
                device=self.device,
                dtype=self.float,
            )
            traj_indices_torch = tlong(traj_indices, device=self.device)
            for traj_idx in self.trajectories:
                if traj_idx not in traj_indices:
                    continue
                states_policy[traj_indices_torch == traj_idx] = self.envs[
                    traj_idx
                ].statebatchpolicy(
                    self.get_states_of_trajectory(traj_idx, states, traj_indices)
                )
            return states_policy
        return self.env.states2policy(states)

    def states2proxy(
        self,
        states: Optional[Union[List[List], List[TensorType["n_states", "..."]]]] = None,
        traj_indices: Optional[Union[List, TensorType["n_states"]]] = None,
    ) -> Union[
        TensorType["n_states", "state_proxy_dims"], npt.NDArray[np.float32], List
    ]:
        """
        Converts states from a list of states in GFlowNet format to a tensor of states
        in proxy format. Note that the implementatiuon of this method differs from
        Batch.states2policy() because the latter always returns torch.tensors. The
        output of the present method can also be numpy arrays or Python lists,
        depending on the proxy.

        Args
        ----
        states: list
            List of states in GFlowNet format.

        traj_indices: list or torch.tensor
            Ids indicating which env corresponds to each state in states. It is only
            used if the environments are conditional to call state2proxy from the right
            environment. Ignored if self.conditional is False.

        Returns
        -------
        states: torch.tensor or ndarray or list
            States in proxy format.
        """
        # If traj_indices is not None and self.conditional is True, then both states
        # and traj_indices must be the same type and have the same length.
        if traj_indices is not None and self.conditional is True:
            assert type(states) == type(traj_indices)
            assert len(states) == len(traj_indices)
        if states is None:
            states = self.states
            traj_indices = self.traj_indices
        if self.conditional:
            states_proxy = []
            index = torch.arange(len(states), device=self.device)
            perm_index = []
            # TODO: rethink this
            for traj_idx in self.trajectories:
                if traj_idx not in traj_indices:
                    continue
                states_proxy.append(
                    self.envs[traj_idx].states2proxy(
                        self.get_states_of_trajectory(traj_idx, states, traj_indices)
                    )
                )
                perm_index.append(index[env_ids == env_id])
            perm_index = torch.cat(perm_index)
            # Reverse permutation to make it index the states_proxy array
            index[perm_index] = index.clone()
            states_proxy = concat_items(states_proxy, index)
            return states_proxy
        return self.env.states2proxy(states)

    def get_actions(self) -> TensorType["n_states, action_dim"]:
        """
        Returns the actions in the batch as a float tensor.
        """
        return tfloat(self.actions, float_type=self.float, device=self.device)

    def get_done(self) -> TensorType["n_states"]:
        """
        Returns the list of done flags as a boolean tensor.
        """
        return tbool(self.done, device=self.device)

    # TODO: check availability one by one as in get_masks
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
        if self._parents_available is False or force_recompute is True:
            self._compute_parents()
        if policy:
            if self._parents_policy_available is False or force_recompute is True:
                self._compute_parents_policy()
            return self.parents_policy
        else:
            return self.parents

    def get_parents_indices(self):
        """
        Returns the indices of the parents of the states in the batch.

        Each i-th item in the returned list contains the index in self.states that
        contains the parent of self.states[i], if it is present there. If a parent
        is not present in self.states (because it is the source), the index is -1.

        Returns
        -------
        self.parents_indices
            The indices in self.states of the parents of self.states.
        """
        if self._parents_available is False:
            self._compute_parents()
        return self.parents_indices

    def _compute_parents(self):
        """
        Obtains the parent (single parent for each state) of all states in the batch
        and its index.

        The parents are computed, obtaining all necessary components, if they are not
        readily available. Missing components and newly computed components are added
        to the batch (self.component is set). The following variables are stored:

        - self.parents: the parent of each state in the batch. It will be the same type
          as self.states (list of lists or tensor)
            Length: n_states
            Shape: [n_states, state_dims]
        - self.parents_indices: the position of each parent in self.states tensor. If a
          parent is not present in self.states (i.e. it is source), the corresponding
          index is -1.

        self._parents_available is set to True.
        """
        self.parents = []
        self.parents_indices = []

        indices_dict = {}
        indices_next = 0

        # Iterate over the trajectories to obtain the parents from the states
        for traj_idx, batch_indices in self.trajectories.items():
            # parent is source
            self.parents.append(self.envs[traj_idx].source)
            # there's no source state in the batch
            self.parents_indices.append(-1)
            # parent is not source
            # TODO: check if tensor and sort without iter
            self.parents.extend([self.states[idx] for idx in batch_indices[:-1]])
            self.parents_indices.extend(batch_indices[:-1])

            # Store the indices required to reorder the parents lists in the same
            # order as the states
            for b_idx in batch_indices:
                indices_dict[b_idx] = indices_next
                indices_next += 1

        # Sort parents list in the same order as states
        # TODO: check if tensor and sort without iter
        self.parents = [self.parents[indices_dict[idx]] for idx in range(len(self))]
        self.parents_indices = tlong(
            [self.parents_indices[indices_dict[idx]] for idx in range(len(self))],
            device=self.device,
        )
        self._parents_available = True

    # TODO: consider converting directly from self.parents
    def _compute_parents_policy(self):
        """
        Obtains the parent (single parent for each state) of all states in the batch,
        in policy format.  The parents are computed, obtaining all necessary
        components, if they are not readily available. Missing components and newly
        computed components are added to the batch (self.component is set). The
        following variable is stored:

        - self.parents_policy: the parent of each state in the batch in policy format.
            Shape: [n_states, state_policy_dims]

        self.parents_policy is stored as a torch tensor and
        self._parents_policy_available is set to True.
        """
        self.states_policy = self.get_states(policy=True)
        self.parents_policy = torch.zeros_like(self.states_policy)
        # Iterate over the trajectories to obtain the parents from the states
        for traj_idx, batch_indices in self.trajectories.items():
            # parent is source
            self.parents_policy[batch_indices[0]] = tfloat(
                self.envs[traj_idx].state2policy(self.envs[traj_idx].source),
                device=self.device,
                float_type=self.float,
            )
            # parent is not source
            self.parents_policy[batch_indices[1:]] = self.states_policy[
                batch_indices[:-1]
            ]
        self._parents_policy_available = True

    def get_parents_all(
        self, policy: bool = False, force_recompute: bool = False
    ) -> Tuple[
        Union[List, TensorType["n_parents", "..."]],
        TensorType["n_parents", "..."],
        TensorType["n_parents"],
    ]:
        """
        Returns the whole set of parents, their corresponding actions and indices of
        all states in the batch. If the parents are not available
        (self._parents_all_available is False) or if force_recompute is True, then
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
        self.parents_all or self.parents_all_policy : list or torch.tensor
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
        if self.continuous:
            raise Exception("get_parents() is ill-defined for continuous environments!")
        if self._parents_all_available is False or force_recompute is True:
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

        - self.parents_all: all the parents of all states in the batch. It will be the
          same type as self.states (list of lists or tensor)
            Length: n_parents
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
        self._parents_all_available is set to True.
        """
        # Iterate over the trajectories to obtain all parents
        self.parents_all = []
        self.parents_actions_all = []
        self.parents_all_indices = []
        self.parents_all_policy = []
        for idx, traj_idx in enumerate(self.traj_indices):
            state = self.states[idx]
            done = self.done[idx]
            action = self.actions[idx]
            parents, parents_a = self.envs[traj_idx].get_parents(
                state=state,
                done=done,
                action=action,
            )
            assert (
                self.env.action2representative(action) in parents_a
            ), f"""
            Sampled action is not in the list of valid actions from parents.
            \nState:\n{state}\nAction:\n{action}
            """
            self.parents_all.extend(parents)
            self.parents_actions_all.extend(parents_a)
            self.parents_all_indices.extend([idx] * len(parents))
            self.parents_all_policy.append(self.envs[traj_idx].states2policy(parents))
        # Convert to tensors
        self.parents_actions_all = tfloat(
            self.parents_actions_all,
            device=self.device,
            float_type=self.float,
        )
        self.parents_all_indices = tlong(
            self.parents_all_indices,
            device=self.device,
        )
        self.parents_all_policy = torch.cat(self.parents_all_policy)
        self._parents_all_available = True

    # TODO: opportunity to improve efficiency by caching.
    def get_masks_forward(
        self,
        of_parents: bool = False,
        force_recompute: bool = False,
    ) -> TensorType["n_states", "action_space_dim"]:
        """
        Returns the forward mask of invalid actions of all states in the batch or of
        their parent in the trajectory if of_parents is True. The masks are computed
        via self._compute_masks_forward if they are not available or if force_recompute
        is True.

        Args
        ----
        of_parents : bool
            If True, the returned masks will correspond to the parents of the states,
            instead of to the states (default).

        force_recompute : bool
            If True, the masks are recomputed even if they are available.

        Returns
        -------
        self.masks_invalid_actions_forward : torch.tensor
            The forward mask of all states in the batch.
        """
        if self._masks_forward_available is False or force_recompute is True:
            self._compute_masks_forward()
        # Make tensor
        masks_invalid_actions_forward = tbool(
            self.masks_invalid_actions_forward, device=self.device
        )
        if of_parents:
            trajectories_parents = {
                traj_idx: [-1] + batch_indices[:-1]
                for traj_idx, batch_indices in self.trajectories.items()
            }
            parents_indices = tlong(
                [
                    trajectories_parents[traj_idx][
                        self.trajectories[traj_idx].index(idx)
                    ]
                    for idx, traj_idx in enumerate(self.traj_indices)
                ],
                device=self.device,
            )
            masks_invalid_actions_forward_parents = torch.zeros_like(
                masks_invalid_actions_forward
            )
            masks_invalid_actions_forward_parents[parents_indices == -1] = self.source[
                "mask_forward"
            ]
            masks_invalid_actions_forward_parents[parents_indices != -1] = (
                masks_invalid_actions_forward[parents_indices[parents_indices != -1]]
            )
            return masks_invalid_actions_forward_parents
        return masks_invalid_actions_forward

    def _compute_masks_forward(self):
        """
        Computes the forward mask of invalid actions of all states in the batch, by
        calling env.get_mask_invalid_actions_forward(). self._masks_forward_available
        is set to True.
        """
        # Iterate over the trajectories to compute all forward masks
        for idx, mask in enumerate(self.masks_invalid_actions_forward):
            if mask is not None:
                continue
            state = self.states[idx]
            done = self.done[idx]
            traj_idx = self.traj_indices[idx]
            self.masks_invalid_actions_forward[idx] = self.envs[
                traj_idx
            ].get_mask_invalid_actions_forward(state, done)
        self._masks_forward_available = True

    # TODO: opportunity to improve efficiency by caching. Note that
    # env.get_masks_invalid_actions_backward() may be expensive because it calls
    # env.get_parents().
    def get_masks_backward(
        self,
        force_recompute: bool = False,
    ) -> TensorType["n_states", "action_space_dim"]:
        """
        Returns the backward mask of invalid actions of all states in the batch. The
        masks are computed via self._compute_masks_backward if they are not available
        or if force_recompute is True.

        Args
        ----
        force_recompute : bool
            If True, the masks are recomputed even if they are available.

        Returns
        -------
        self.masks_invalid_actions_backward : torch.tensor
            The backward mask of all states in the batch.
        """
        if self._masks_backward_available is False or force_recompute is True:
            self._compute_masks_backward()
        return tbool(self.masks_invalid_actions_backward, device=self.device)

    def _compute_masks_backward(self):
        """
        Computes the backward mask of invalid actions of all states in the batch, by
        calling env.get_mask_invalid_actions_backward(). self._masks_backward_available
        is set to True.
        """
        # Iterate over the trajectories to compute all backward masks
        for idx, mask in enumerate(self.masks_invalid_actions_backward):
            if mask is not None:
                continue
            state = self.states[idx]
            done = self.done[idx]
            traj_idx = self.traj_indices[idx]
            self.masks_invalid_actions_backward[idx] = self.envs[
                traj_idx
            ].get_mask_invalid_actions_backward(state, done)
        self._masks_backward_available = True

    # TODO: better handling of availability of rewards, logrewards, proxy_values.
    def get_rewards(
        self,
        log: bool = False,
        force_recompute: Optional[bool] = False,
        do_non_terminating: Optional[bool] = False,
    ) -> TensorType["n_states"]:
        """
        Returns the rewards of all states in the batch (including not done).

        Parameters
        ----------
        log : bool
            If True, return the logarithm of the rewards.
        force_recompute : bool
            If True, the rewards are recomputed even if they are available.
        do_non_terminating : bool
            If True, return the actual rewards of the non-terminating states. If
            False, non-terminating states will be assigned reward 0.
        """
        if self.rewards_available(log) is False or force_recompute is True:
            self._compute_rewards(log, do_non_terminating)
        if log:
            return self.logrewards
        else:
            return self.rewards

    def get_proxy_values(
        self,
        force_recompute: Optional[bool] = False,
        do_non_terminating: Optional[bool] = False,
    ) -> TensorType["n_states"]:
        """
        Returns the proxy values of all states in the batch (including not done).

        Parameters
        ----------
        force_recompute : bool
            If True, the proxy values are recomputed even if they are available.
        do_non_terminating : bool
            If True, return the actual proxy values of the non-terminating states. If
            False, non-terminating states will be assigned value inf.
        """
        if self._proxy_values_available is False or force_recompute is True:
            self._compute_rewards(do_non_terminating=do_non_terminating)
        return self.proxy_values

    def _compute_rewards(
        self, log: bool = False, do_non_terminating: Optional[bool] = False
    ):
        """
        Computes rewards for all self.states by first converting the states into proxy
        format. The result is stored in self.rewards as a torch.tensor

        Parameters
        ----------
        log : bool
            If True, compute the logarithm of the rewards.
        do_non_terminating : bool
            If True, compute the rewards of the non-terminating states instead of
            assigning reward 0 and proxy value inf.
        """

        if do_non_terminating:
            rewards, proxy_values = self.proxy.rewards(
                self.states2proxy(), log, return_proxy=True
            )
        else:
            rewards = self.proxy.get_min_reward(log) * torch.ones(
                len(self), dtype=self.float, device=self.device
            )
            proxy_values = torch.full_like(rewards, torch.inf)
            done = self.get_done()
            if len(done) > 0:
                states_proxy_done = self.get_terminating_states(proxy=True)
                rewards[done], proxy_values[done] = self.proxy.rewards(
                    states_proxy_done, log, return_proxy=True
                )

        self.proxy_values = proxy_values
        self._proxy_values_available = True
        if log:
            self.logrewards = rewards
            self._logrewards_available = True
        else:
            self.rewards = rewards
            self._rewards_available = True

    def get_rewards_parents(self, log: bool = False) -> TensorType["n_states"]:
        """
        Returns the rewards of all parents in the batch.

        Parameters
        ----------
        log : bool
            If True, return the logarithm of the rewards.

        Returns
        -------
        self.rewards_parents or self.logrewards_parents
            A tensor containing the rewards of the parents of self.states.
        """
        if not self.rewards_parents_available(log):
            self._compute_rewards_parents(log)
        if log:
            return self.logrewards_parents
        else:
            return self.rewards_parents

    def _compute_rewards_parents(self, log: bool = False):
        """
        Computes the rewards of self.parents by reusing the rewards of the states
        (self.rewards).

        Stores the result in self.rewards_parents or self.logrewards_parents.

        Parameters
        ----------
        log : bool
            If True, compute the logarithm of the rewards.
        """
        # TODO: this may return zero rewards for all parents if before
        # rewards for states were computed with do_non_terminating=False
        state_rewards = self.get_rewards(log=log, do_non_terminating=True)
        rewards_parents = torch.zeros_like(state_rewards)
        parent_indices = self.get_parents_indices()
        parent_is_source = parent_indices == -1
        rewards_parents[~parent_is_source] = state_rewards[
            parent_indices[~parent_is_source]
        ]
        rewards_source = self.get_rewards_source(log)
        rewards_parents[parent_is_source] = rewards_source[parent_is_source]
        if log:
            self.logrewards_parents = rewards_parents
            self._logrewards_parents_available = True
        else:
            self.rewards_parents = rewards_parents
            self._rewards_parents_available = True

    def get_rewards_source(self, log: bool = False) -> TensorType["n_states"]:
        """
        Returns rewards of the corresponding source states for each state in the batch.

        Parameters
        ----------
        log : bool
            If True, return the logarithm of the rewards.

        Returns
        -------
        self.rewards_source or self.logrewards_source
            A tensor containing the rewards the source states.
        """
        if not self.rewards_source_available(log):
            self._compute_rewards_source(log)
        if log:
            return self.logrewards_source
        else:
            return self.rewards_source

    def _compute_rewards_source(self, log: bool = False):
        """
        Computes a tensor of length len(self.states) with the rewards of the
        corresponding source states.

        Stores the result in self.rewards_source or self.logrewards_source.

        Parameters
        ----------
        log : bool
            If True, compute the logarithm of the rewards.
        """
        # This will not work if source is randomised
        if not self.conditional:
            source_proxy = self.env.state2proxy(self.env.source)
            reward_source = self.proxy.rewards(source_proxy, log)
            rewards_source = reward_source.expand(len(self))
        else:
            raise NotImplementedError
        if log:
            self.logrewards_source = rewards_source
            self._logrewards_source_available = True
        else:
            self.rewards_source = rewards_source
            self._rewards_source_available = True

    def get_terminating_states(
        self,
        sort_by: str = "insertion",
        policy: Optional[bool] = False,
        proxy: Optional[bool] = False,
    ) -> Union[TensorType["n_trajectories", "..."], npt.NDArray[np.float32], List]:
        """
        Returns the terminating states in the batch, that is all states with done =
        True. The states will be returned in either GFlowNet format (default), policy
        (policy = True) or proxy (proxy = True) format. If both policy and proxy are
        True, it raises an error due to the ambiguity. The returned states may be
        sorted by order of insertion (sort_by = "insert[ion]", default) or by
        trajectory index (sort_by = "traj[ectory]".

        Args
        ----
        sort_by : str
            Indicates how to sort the output:
                - insert[ion]: sort by order of insertion (states of trajectories that
                  reached the terminating state first come first)
                - traj[ectory]: sort by trajectory index (the order in the ordered
                  dict self.trajectories)

        policy : bool
            If True, the policy format of the states is returned.

        proxy : bool
            If True, the proxy format of the states is returned.
        """
        if sort_by == "insert" or sort_by == "insertion":
            indices = np.arange(len(self))
        elif sort_by == "traj" or sort_by == "trajectory":
            indices = np.argsort(self.traj_indices)
        else:
            raise ValueError("sort_by must be either insert[ion] or traj[ectory]")
        if policy is True and proxy is True:
            raise ValueError(
                "Ambiguous request! Only one of policy or proxy can be True."
            )
        traj_indices = None
        if torch.is_tensor(self.states):
            indices = tlong(indices, device=self.device)
            done = self.get_done()[indices]
            states_term = self.states[indices][done, :]
            if self.conditional and (policy is True or proxy is True):
                traj_indices = tlong(self.traj_indices, device=self.device)[indices][
                    done
                ]
                assert len(traj_indices) == len(torch.unique(traj_indices))
        elif isinstance(self.states, list):
            states_term = [self.states[idx] for idx in indices if self.done[idx]]
            if self.conditional and (policy is True or proxy is True):
                done = np.array(self.done, dtype=bool)[indices]
                traj_indices = np.array(self.traj_indices)[indices][done]
                assert len(traj_indices) == len(np.unique(traj_indices))
        else:
            raise NotImplementedError("self.states can only be list or torch.tensor")
        if policy is True:
            return self.states2policy(states_term, traj_indices)
        elif proxy is True:
            return self.states2proxy(states_term, traj_indices)
        else:
            return states_term

    def get_terminating_rewards(
        self,
        sort_by: str = "insertion",
        log: bool = False,
        force_recompute: Optional[bool] = False,
    ) -> TensorType["n_trajectories"]:
        """
        Returns the reward of the terminating states in the batch, that is all states
        with done = True. The returned rewards may be sorted by order of insertion
        (sort_by = "insert[ion]", default) or by trajectory index (sort_by =
        "traj[ectory]".

        Parameters
        ----------
        sort_by : str
            Indicates how to sort the output:
                - insert[ion]: sort by order of insertion (rewards of trajectories that
                  reached the terminating state first come first)
                - traj[ectory]: sort by trajectory index (the order in the ordered
                  dict self.trajectories)
        log : bool
            If True, return the logarithm of the rewards.
        force_recompute : bool
            If True, the rewards are recomputed even if they are available.
        """
        if sort_by == "insert" or sort_by == "insertion":
            indices = np.arange(len(self))
        elif sort_by == "traj" or sort_by == "trajectory":
            indices = np.argsort(self.traj_indices)
        else:
            raise ValueError("sort_by must be either insert[ion] or traj[ectory]")
        if self.rewards_available(log) is False or force_recompute is True:
            self._compute_rewards(log, do_non_terminating=False)
        done = self.get_done()[indices]
        if log:
            return self.logrewards[indices][done]
        else:
            return self.rewards[indices][done]

    def get_terminating_proxy_values(
        self,
        sort_by: str = "insertion",
        force_recompute: Optional[bool] = False,
    ) -> TensorType["n_trajectories"]:
        """
        Returns the proxy values of the terminating states in the batch, that is all
        states with done = True. The returned proxy values may be sorted by order of
        insertion (sort_by = "insert[ion]", default) or by trajectory index (sort_by =
        "traj[ectory]".

        Parameters
        ----------
        sort_by : str
            Indicates how to sort the output:
                - insert[ion]: sort by order of insertion (proxy values of trajectories
                  that reached the terminating state first come first)
                - traj[ectory]: sort by trajectory index (the order in the ordered
                  dict self.trajectories)
        force_recompute : bool
            If True, the proxy_values are recomputed even if they are available.
        """
        if sort_by == "insert" or sort_by == "insertion":
            indices = np.arange(len(self))
        elif sort_by == "traj" or sort_by == "trajectory":
            indices = np.argsort(self.traj_indices)
        else:
            raise ValueError("sort_by must be either insert[ion] or traj[ectory]")
        if self._proxy_values_available is False or force_recompute is True:
            self._compute_rewards(log, do_non_terminating=False)
        done = self.get_done()[indices]
        return self.proxy_values[indices][done]

    def get_actions_trajectories(self) -> List[List[Tuple]]:
        """
        Returns the actions corresponding to all trajectories in the batch, sorted by
        trajectory index (the order in the ordered dict self.trajectories).
        """
        actions_trajectories = []
        for batch_indices in self.trajectories.values():
            actions_trajectories.append([self.actions[idx] for idx in batch_indices])
        return actions_trajectories

    def get_states_of_trajectory(
        self,
        traj_idx: int,
        states: Optional[
            Union[TensorType["n_states", "..."], npt.NDArray[np.float32], List]
        ] = None,
        traj_indices: Optional[Union[List, TensorType["n_states"]]] = None,
    ) -> Union[
        TensorType["n_states", "state_proxy_dims"], npt.NDArray[np.float32], List
    ]:
        """
        Returns the states of the trajectory indicated by traj_idx. If states and
        traj_indices are not None, then these will be the only states and trajectory
        indices considered.

        See: states2policy()
        See: states2proxy()

        Args
        ----
        traj_idx : int
            Index of the trajectory from which to return the states.

        states : tensor, array or list
            States from the trajectory to consider.

        traj_indices : tensor, array or list
            Trajectory indices of the trajectory to consider.

        Returns
        -------
        Tensor, array or list of states of the requested trajectory.
        """
        # TODO: re-implement using the batch indices in self.trajectories[traj_idx]
        # If either states or traj_indices are not None, both must be the same type and
        # have the same length.
        # TODO: or add sort_by
        if states is not None or traj_indices is not None:
            assert type(states) == type(traj_indices)
            assert len(states) == len(traj_indices)
        else:
            states = self.states
            traj_indices = self.traj_indices
        if torch.is_tensor(states):
            return states[tlong(traj_indices, device=self.device) == traj_idx]
        elif isinstance(states, list):
            return [
                state for state, idx in zip(states, traj_indices) if idx == traj_idx
            ]
        elif isinstance(states, np.ndarray):
            return states[np.array(traj_indices) == traj_idx]
        else:
            raise ValueError("states can only be list, torch.tensor or ndarray")

    def merge(self, batches: List):
        """
        Merges the current Batch (self) with the Batch or list of Batches passed as
        argument.

        Returns
        -------
        self
        """
        if not isinstance(batches, list):
            batches = [batches]
        for batch in batches:
            if len(batch) == 0:
                continue
            # Shift trajectory indices of batch to merge
            if len(self) == 0:
                traj_idx_shift = 0
            else:
                traj_idx_shift = np.max(list(self.trajectories.keys())) + 1
            batch._shift_indices(traj_shift=traj_idx_shift, batch_shift=len(self))
            # Merge main data
            self.size += batch.size
            self.envs.update(batch.envs)
            self.trajectories.update(batch.trajectories)
            self.traj_indices.extend(batch.traj_indices)
            self.state_indices.extend(batch.state_indices)
            self.states.extend(batch.states)
            self.actions.extend(batch.actions)
            self.done.extend(batch.done)
            self.masks_invalid_actions_forward = extend(
                self.masks_invalid_actions_forward,
                batch.masks_invalid_actions_forward,
            )
            self.masks_invalid_actions_backward = extend(
                self.masks_invalid_actions_backward,
                batch.masks_invalid_actions_backward,
            )
            # Merge "optional" data
            if self.states_policy is not None and batch.states_policy is not None:
                self.states_policy = extend(self.states_policy, batch.states_policy)
            else:
                self.states_policy = None
            if self._parents_available and batch._parents_available:
                self.parents = extend(self.parents, batch.parents)
            else:
                self.parents = None
            if self._parents_policy_available and batch._parents_policy_available:
                self.parents_policy = extend(self.parents_policy, batch.parents_policy)
            else:
                self.parents_policy = None
            if self._parents_all_available and batch._parents_all_available:
                self.parents_all = extend(self.parents_all, batch.parents_all)
            else:
                self.parents_all = None
            if self._rewards_available and batch._rewards_available:
                self.rewards = extend(self.rewards, batch.rewards)
            else:
                self.rewards = None
            if self._logrewards_available and batch._logrewards_available:
                self.logrewards = extend(self.logrewards, batch.logrewards)
            else:
                self.logrewards = None
        assert self.is_valid()
        return self

    def is_valid(self) -> bool:
        """
        Performs basic checks on the current state of the batch.

        Returns
        -------
        True if all the checks are valid, False otherwise.
        """
        if len(self.states) != len(self):
            return False
        if len(self.actions) != len(self):
            return False
        if len(self.done) != len(self):
            return False
        if len(self.traj_indices) != len(self):
            return False
        if len(self.state_indices) != len(self):
            return False
        if set(np.unique(self.traj_indices)) != set(self.envs.keys()):
            return False
        if set(self.trajectories.keys()) != set(self.envs.keys()):
            return False
        batch_indices = [
            idx for indices in self.trajectories.values() for idx in indices
        ]
        if len(batch_indices) != len(self):
            return False
        if len(np.unique(batch_indices)) != len(batch_indices):
            return False
        return True

    def traj_indices_are_consecutive(self) -> bool:
        """
        Returns True if the trajectory indices start from 0 and are consecutive; False
        otherwise.
        """
        trajectories_consecutive = list(self.trajectories) == list(
            np.arange(self.get_n_trajectories())
        )
        envs_consecutive = list(self.envs) == list(np.arange(self.get_n_trajectories()))
        return trajectories_consecutive and envs_consecutive

    def make_indices_consecutive(self):
        """
        Updates the trajectory indices as well as the env ids such that they start from
        0 and are consecutive. Note that only the trajectory indices are changed, but
        importantly the order of the main data in the batch is preserved.

        Examples:

        - Original indices: 0, 10, 20
        - New indices: 0, 1, 2

        - Original indices: 1, 5, 3
        - New indices: 0, 1, 2

        Note: this method is unsued as of September 1st 2023, but is left here for
        potential future use.
        """
        if self.traj_indices_are_consecutive():
            return
        self.traj_indices = self.get_trajectory_indices(consecutive=True).tolist()
        self.trajectories = OrderedDict(
            zip(range(self.get_n_trajectories()), self.trajectories.values())
        )
        self.envs = OrderedDict(
            {idx: env.set_id(idx) for idx, env in enumerate(self.envs.values())}
        )
        assert self.traj_indices_are_consecutive()
        assert self.is_valid()

    def _shift_indices(self, traj_shift: int, batch_shift: int):
        """
        Shifts all the trajectory indices and environment ids by traj_shift and the batch
        indices by batch_shift.

        Returns
        -------
        self
        """
        if not self.is_valid():
            raise Exception("Batch is not valid before attempting indices shift")
        self.traj_indices = [idx + traj_shift for idx in self.traj_indices]
        self.trajectories = {
            traj_idx + traj_shift: list(map(lambda x: x + batch_shift, batch_indices))
            for traj_idx, batch_indices in self.trajectories.items()
        }
        self.envs = {
            k + traj_shift: env.set_id(k + traj_shift) for k, env in self.envs.items()
        }
        if not self.is_valid():
            raise Exception("Batch is not valid after performing indices shift")
        return self

    # TODO: rewrite once cache is implemnted
    def get_item(
        self,
        item: str,
        env: GFlowNetEnv = None,
        traj_idx: int = None,
        action_idx: int = None,
        backward: bool = False,
    ):
        """
        Returns the item specified by item of either:
            - environment env, OR
            - trajectory traj_idx AND action number action_idx (in the order of
              sampling)

        If all arguments are given, then they must be consistent, otherwise an
        exception (assert) is raised due to ambiguity.

        If a mask is requested but is missing, it is computed and stored.

        Args
        ----
        item : str
            String identifier of the item to retrieve from the batch. Options
                - state
                - parent
                - action
                - done
                - mask_f[orward]
                - mask_b[ackward]

        traj_idx : int
            Trajectory index

        action_idx : int
            Action index. Regardless of forward of backward, n-th item sampled when
            forming the batch.

        backward : bool
            Whether the trajectory is sampling backward. False (forward) by default.

        Returns
        -------
        The requested item if it is available or None if it is not. It raises an error
        if the request can be identified as incorrect.
        """
        # Preliminary checks
        if env is not None:
            if traj_idx is not None:
                assert (
                    env.id == traj_idx
                ), "env.id {env.id} different to traj_idx {traj_idx}."
            else:
                traj_idx = env.id
            if action_idx is not None:
                assert (
                    env.n_actions == action_idx
                ), "env.n_actions {env.n_actions} different to action_idx {action_idx}."
            else:
                action_idx = env.n_actions
        else:
            assert (
                traj_idx is not None and action_idx is not None
            ), "Either env or traj_idx AND action_idx must be provided"
        # Handle action_idx = 0 (source state)
        if action_idx == 0:
            if backward is False:
                if item == "state":
                    return self.source["state"]
                elif item == "mask_f" or item == "mask_forward":
                    return self.source["mask_forward"]
                else:
                    raise ValueError(
                        "Only state or mask_forward are available for a fresh env "
                        "(action_idx = 0)"
                    )
        #             else:
        #                 # TODO: handle backward masks with cache
        #                 raise NotImplementedError(
        #                     "get_item at action_idx = 0 for backward trajectories is currently "
        #                     "not supported"
        #                 )
        batch_idx = self.traj_idx_action_idx_to_batch_idx(
            traj_idx, action_idx, backward
        )
        if batch_idx is None:
            # TODO: handle this
            if env is None:
                raise ValueError(
                    "{item} not available for action {action_idx} of trajectory "
                    "{traj_idx} and no env was provided."
                )
            else:
                if item == "state":
                    return env.state
                elif item == "done":
                    return env.done
                elif item == "mask_f" or item == "mask_forward":
                    return env.get_mask_invalid_actions_forward()
                elif item == "mask_b" or item == "mask_backward":
                    return env.get_mask_invalid_actions_backward()
                else:
                    raise ValueError(
                        "Not available in the batch. item must be one of: state, done, "
                        "mask_f[orward] or mask_b[ackward]."
                    )
        if item == "state":
            return self.states[batch_idx]
        elif item == "parent":
            return self.parents[batch_idx]
        elif item == "action":
            return self.actions[batch_idx]
        elif item == "done":
            return self.done[batch_idx]
        elif item == "mask_f" or item == "mask_forward":
            if self.masks_invalid_actions_forward[batch_idx] is None:
                state = self.states[batch_idx]
                done = self.done[batch_idx]
                self.masks_invalid_actions_forward[batch_idx] = self.envs[
                    traj_idx
                ].get_mask_invalid_actions_forward(state, done)
            return self.masks_invalid_actions_forward[batch_idx]
        elif item == "mask_b" or item == "mask_backward":
            if self.masks_invalid_actions_backward[batch_idx] is None:
                state = self.states[batch_idx]
                done = self.done[batch_idx]
                self.masks_invalid_actions_backward[batch_idx] = self.envs[
                    traj_idx
                ].get_mask_invalid_actions_backward(state, done)
            return self.masks_invalid_actions_backward[batch_idx]
        else:
            raise ValueError(
                "item must be one of: state, parent, action, done, mask_f[orward] or "
                "mask_b[ackward]"
            )
