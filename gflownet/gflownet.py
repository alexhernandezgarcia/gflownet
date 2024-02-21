"""
GFlowNet
TODO:
    - Seeds
"""

import copy
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.special import logsumexp
from torch.distributions import Bernoulli
from tqdm import tqdm

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.batch import Batch
from gflownet.utils.buffer import Buffer
from gflownet.utils.common import (
    batch_with_rest,
    bootstrap_samples,
    set_device,
    set_float_precision,
    tbool,
    tfloat,
    tlong,
    torch2np,
)


class GFlowNetAgent:
    def __init__(
        self,
        env,
        seed,
        device,
        float_precision,
        optimizer,
        buffer,
        forward_policy,
        backward_policy,
        mask_invalid_actions,
        temperature_logits,
        random_action_prob,
        pct_offline,
        logger,
        num_empirical_loss,
        oracle,
        state_flow=None,
        active_learning=False,
        sample_only=False,
        replay_sampling="permutation",
        train_sampling="permutation",
        **kwargs,
    ):
        # Seed
        self.rng = np.random.default_rng(seed)
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Environment
        self.env = env
        # Continuous environments
        self.continuous = hasattr(self.env, "continuous") and self.env.continuous
        if self.continuous and optimizer.loss in ["flowmatch", "flowmatching"]:
            raise Exception(
                """
            Flow matching loss is not available for continuous environments.
            You may use trajectory balance (gflownet=trajectorybalance) instead.
            """
            )
        # Loss
        if optimizer.loss in ["flowmatch", "flowmatching"]:
            self.loss = "flowmatch"
            self.logZ = None
        elif optimizer.loss in ["trajectorybalance", "tb"]:
            self.loss = "trajectorybalance"
            self.logZ = nn.Parameter(torch.ones(optimizer.z_dim) * 150.0 / 64)
        elif optimizer.loss in ["detailedbalance", "db"]:
            self.loss = "detailedbalance"
            self.logZ = None
        elif optimizer.loss in ["forwardlooking", "fl"]:
            self.loss = "forwardlooking"
            self.logZ = None
        else:
            print("Unkown loss. Using flowmatch as default")
            self.loss = "flowmatch"
            self.logZ = None
        # loss_eps is used only for the flowmatch loss
        self.loss_eps = torch.tensor(float(1e-5)).to(self.device)
        # Logging
        self.num_empirical_loss = num_empirical_loss
        self.logger = logger
        self.oracle_n = oracle.n
        # Buffers
        self.replay_sampling = replay_sampling
        self.train_sampling = train_sampling
        self.buffer = Buffer(
            **buffer, env=self.env, make_train_test=not sample_only, logger=logger
        )
        # Train set statistics and reward normalization constant
        if self.buffer.train is not None:
            energies_stats_tr = [
                self.buffer.min_tr,
                self.buffer.max_tr,
                self.buffer.mean_tr,
                self.buffer.std_tr,
                self.buffer.max_norm_tr,
            ]
            self.env.set_energies_stats(energies_stats_tr)
            print("\nTrain data")
            print(f"\tMean score: {energies_stats_tr[2]}")
            print(f"\tStd score: {energies_stats_tr[3]}")
            print(f"\tMin score: {energies_stats_tr[0]}")
            print(f"\tMax score: {energies_stats_tr[1]}")
        else:
            energies_stats_tr = None
        if self.env.reward_norm_std_mult > 0 and energies_stats_tr is not None:
            self.env.reward_norm = self.env.reward_norm_std_mult * energies_stats_tr[3]
            self.env.set_reward_norm(self.env.reward_norm)
        # Test set statistics
        if self.buffer.test is not None:
            print("\nTest data")
            print(f"\tMean score: {self.buffer.test['energies'].mean()}")
            print(f"\tStd score: {self.buffer.test['energies'].std()}")
            print(f"\tMin score: {self.buffer.test['energies'].min()}")
            print(f"\tMax score: {self.buffer.test['energies'].max()}")

        # Models
        self.forward_policy = forward_policy
        if self.forward_policy.checkpoint is not None:
            self.logger.set_forward_policy_ckpt_path(self.forward_policy.checkpoint)
            # TODO: re-write the logic and conditions to reload a model
            if False:
                self.forward_policy.load_state_dict(
                    torch.load(self.policy_forward_path)
                )
                print("Reloaded GFN forward policy model Checkpoint")
        else:
            self.logger.set_forward_policy_ckpt_path(None)

        self.backward_policy = backward_policy
        self.logger.set_backward_policy_ckpt_path(None)
        if self.backward_policy.checkpoint is not None:
            self.logger.set_backward_policy_ckpt_path(self.backward_policy.checkpoint)
            # TODO: re-write the logic and conditions to reload a model
            if False:
                self.backward_policy.load_state_dict(
                    torch.load(self.policy_backward_path)
                )
                print("Reloaded GFN backward policy model Checkpoint")
        else:
            self.logger.set_backward_policy_ckpt_path(None)

        self.state_flow = state_flow
        if self.state_flow is not None and self.state_flow.checkpoint is not None:
            self.logger.set_state_flow_ckpt_path(self.state_flow.checkpoint)
            # TODO: add the logic and conditions to reload a model
        else:
            self.logger.set_state_flow_ckpt_path(None)

        # Optimizer
        if self.forward_policy.is_model:
            self.target = copy.deepcopy(self.forward_policy.model)
            self.opt, self.lr_scheduler = make_opt(
                self.parameters(), self.logZ, optimizer
            )
        else:
            self.opt, self.lr_scheduler, self.target = None, None, None
        self.n_train_steps = optimizer.n_train_steps
        self.batch_size = optimizer.batch_size
        self.batch_size_total = sum(self.batch_size.values())
        self.ttsr = max(int(optimizer.train_to_sample_ratio), 1)
        self.sttr = max(int(1 / optimizer.train_to_sample_ratio), 1)
        self.clip_grad_norm = optimizer.clip_grad_norm
        self.tau = optimizer.bootstrap_tau
        self.ema_alpha = optimizer.ema_alpha
        self.early_stopping = optimizer.early_stopping
        self.use_context = active_learning
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # Training
        self.mask_invalid_actions = mask_invalid_actions
        self.temperature_logits = temperature_logits
        self.random_action_prob = random_action_prob
        self.pct_offline = pct_offline
        # Metrics
        self.l1 = -1.0
        self.kl = -1.0
        self.jsd = -1.0
        self.corr_prob_traj_rewards = 0.0
        self.var_logrewards_logp = -1.0
        self.nll_tt = 0.0
        self.mean_logprobs_std = -1.0
        self.mean_probs_std = -1.0
        self.logprobs_std_nll_ratio = -1.0

    def parameters(self):
        parameters = list(self.forward_policy.model.parameters())
        if self.backward_policy.is_model:
            if self.loss == "flowmatch":
                raise ValueError("Backward Policy cannot be a model in flowmatch.")
            parameters += list(self.backward_policy.model.parameters())
        if self.state_flow is not None:
            if self.loss not in ["detailedbalance", "forwardlooking"]:
                raise ValueError(f"State flow cannot be trained with {self.loss} loss.")
            parameters += list(self.state_flow.model.parameters())
        return parameters

    def sample_actions(
        self,
        envs: List[GFlowNetEnv],
        batch: Optional[Batch] = None,
        sampling_method: Optional[str] = "policy",
        backward: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        random_action_prob: Optional[float] = None,
        no_random: Optional[bool] = True,
        times: Optional[dict] = None,
    ) -> List[Tuple]:
        """
        Samples one action on each environment of the list envs, according to the
        sampling method specified by sampling_method.

        With probability 1 - random_action_prob, actions will be sampled from the
        self.forward_policy or self.backward_policy, depending on backward. The rest
        are sampled according to the random policy of the environment
        (model.random_distribution).

        If a batch is provided (and self.mask_invalid_actions) is True, the masks are
        retrieved from the batch. Otherwise they are computed from the environments.

        Args
        ----
        envs : list of GFlowNetEnv or derived
            A list of instances of the environment

        batch_forward : Batch
            A batch from which to obtain required variables (e.g. masks) to avoid
            recomputing them.

        sampling_method : string
            - policy: uses current forward to obtain the sampling probabilities.
            - random: samples purely from a random policy, that is
                - random_action_prob = 1.0
              regardless of the value passed as arguments.

        backward : bool
            True if sampling is backward. False (forward) by default.

        temperature : float
            Temperature to adjust the logits by logits /= temperature. If None,
            self.temperature_logits is used.

        random_action_prob : float
            Probability of sampling random actions. If None (default),
            self.random_action_prob is used, unless its value is forced to either 0.0
            or 1.0 by other arguments (sampling_method or no_random).
        no_random : bool
            If True, the samples will strictly be on-policy, that is
                - temperature = 1.0
                - random_action_prob = 0.0
            regardless of the values passed as arguments.

        times : dict
            Dictionary to store times. Currently not implemented.

        Returns
        -------
        actions : list of tuples
            The sampled actions, one for each environment in envs.
        """
        # Preliminaries
        if sampling_method == "random":
            assert (
                no_random is False
            ), "sampling_method random and no_random True is ambiguous"
            random_action_prob = 1.0
            temperature = 1.0
        elif no_random is True:
            temperature = 1.0
            random_action_prob = 0.0
        else:
            if temperature is None:
                temperature = self.temperature_logits
            if random_action_prob is None:
                random_action_prob = self.random_action_prob
        if backward:
            # TODO: backward sampling with FM?
            model = self.backward_policy
        else:
            model = self.forward_policy

        # TODO: implement backward sampling from forward policy as in old
        # backward_sample.
        if not isinstance(envs, list):
            envs = [envs]
        # Build states and masks
        states = [env.state for env in envs]

        # Retrieve masks from the batch (batch.get_item("mask_*") computes the mask if
        # it is not available and stores it in the batch)
        # TODO: make get_mask_ method with the ugly code below
        if self.mask_invalid_actions is True:
            if batch is not None:
                if backward:
                    mask_invalid_actions = tbool(
                        [
                            batch.get_item("mask_backward", env, backward=True)
                            for env in envs
                        ],
                        device=self.device,
                    )
                else:
                    mask_invalid_actions = tbool(
                        [batch.get_item("mask_forward", env) for env in envs],
                        device=self.device,
                    )
            # Compute masks since a batch was not provided
            else:
                if backward:
                    mask_invalid_actions = tbool(
                        [env.get_mask_invalid_actions_backward() for env in envs],
                        device=self.device,
                    )
                else:
                    mask_invalid_actions = tbool(
                        [env.get_mask_invalid_actions_forward() for env in envs],
                        device=self.device,
                    )
        else:
            mask_invalid_actions = None

        # Build policy outputs
        policy_outputs = model.random_distribution(states)
        idx_norandom = (
            Bernoulli(
                (1 - random_action_prob) * torch.ones(len(states), device=self.device)
            )
            .sample()
            .to(bool)
        )
        # Get policy outputs from model
        if sampling_method == "policy":
            # Check for at least one non-random action
            if idx_norandom.sum() > 0:
                states_policy = tfloat(
                    self.env.states2policy(
                        [s for s, do in zip(states, idx_norandom) if do]
                    ),
                    device=self.device,
                    float_type=self.float,
                )
                policy_outputs[idx_norandom, :] = model(states_policy)
        else:
            raise NotImplementedError

        # Sample actions from policy outputs
        # TODO: consider adding logprobs to batch
        actions, logprobs = self.env.sample_actions_batch(
            policy_outputs,
            mask_invalid_actions,
            states,
            backward,
            sampling_method,
            temperature,
        )
        return actions

    def step(
        self,
        envs: List[GFlowNetEnv],
        actions: List[Tuple],
        backward: bool = False,
    ):
        """
        Executes the actions on the environments envs, one by one. This method simply
        calls env.step(action) or env.step_backwards(action) for each (env, action)
        pair, depending on thhe value of backward.

        Args
        ----
        envs : list of GFlowNetEnv or derived
            A list of instances of the environment

        actions : list
            A list of actions to be executed on each env of envs.

        backward : bool
            True if sampling is backward. False (forward) by default.
        """
        assert len(envs) == len(actions)
        if not isinstance(envs, list):
            envs = [envs]
        if backward:
            _, actions, valids = zip(
                *[env.step_backwards(action) for env, action in zip(envs, actions)]
            )
        else:
            _, actions, valids = zip(
                *[env.step(action) for env, action in zip(envs, actions)]
            )
        return envs, actions, valids

    @torch.no_grad()
    # TODO: extract code from while loop to avoid replication
    def sample_batch(
        self,
        n_forward: int = 0,
        n_train: int = 0,
        n_replay: int = 0,
        train=True,
        progress=False,
    ):
        """
        TODO: extend docstring.
        Builds a batch of data by sampling online and/or offline trajectories.
        """
        # PRELIMINARIES: Prepare Batch and environments
        times = {
            "all": 0.0,
            "forward_actions": 0.0,
            "train_actions": 0.0,
            "replay_actions": 0.0,
            "actions_envs": 0.0,
        }
        t0_all = time.time()
        batch = Batch(env=self.env, device=self.device, float_type=self.float)

        # ON-POLICY FORWARD trajectories
        t0_forward = time.time()
        envs = [self.env.copy().reset(idx) for idx in range(n_forward)]
        batch_forward = Batch(env=self.env, device=self.device, float_type=self.float)
        while envs:
            # Sample actions
            t0_a_envs = time.time()
            actions = self.sample_actions(
                envs,
                batch_forward,
                no_random=not train,
                times=times,
            )
            times["actions_envs"] += time.time() - t0_a_envs
            # Update environments with sampled actions
            envs, actions, valids = self.step(envs, actions)
            # Add to batch
            batch_forward.add_to_batch(envs, actions, valids, train=train)
            # Filter out finished trajectories
            envs = [env for env in envs if not env.done]
        times["forward_actions"] = time.time() - t0_forward

        # TRAIN BACKWARD trajectories
        t0_train = time.time()
        envs = [self.env.copy().reset(idx) for idx in range(n_train)]
        batch_train = Batch(env=self.env, device=self.device, float_type=self.float)
        if n_train > 0 and self.buffer.train_pkl is not None:
            with open(self.buffer.train_pkl, "rb") as f:
                dict_train = pickle.load(f)
                x_train = self.buffer.select(
                    dict_train, n_train, self.train_sampling, self.rng
                )
            for env, x in zip(envs, x_train):
                env.set_state(x, done=True)
        while envs:
            # Sample backward actions
            t0_a_envs = time.time()
            actions = self.sample_actions(
                envs,
                batch_train,
                backward=True,
                no_random=not train,
                times=times,
            )
            times["actions_envs"] += time.time() - t0_a_envs
            # Update environments with sampled actions
            envs, actions, valids = self.step(envs, actions, backward=True)
            # Add to batch
            batch_train.add_to_batch(envs, actions, valids, backward=True, train=train)
            assert all(valids)
            # Filter out finished trajectories
            envs = [env for env in envs if not env.equal(env.state, env.source)]
        times["train_actions"] = time.time() - t0_train

        # REPLAY BACKWARD trajectories
        t0_replay = time.time()
        batch_replay = Batch(env=self.env, device=self.device, float_type=self.float)
        if n_replay > 0 and self.buffer.replay_pkl is not None:
            with open(self.buffer.replay_pkl, "rb") as f:
                dict_replay = pickle.load(f)
                n_replay = min(n_replay, len(dict_replay["x"]))
                envs = [self.env.copy().reset(idx) for idx in range(n_replay)]
                x_replay = self.buffer.select(
                    dict_replay, n_replay, self.replay_sampling, self.rng
                )
            for env, x in zip(envs, x_replay):
                env.set_state(x, done=True)
        while envs:
            # Sample backward actions
            t0_a_envs = time.time()
            actions = self.sample_actions(
                envs,
                batch_replay,
                backward=True,
                no_random=not train,
                times=times,
            )
            times["actions_envs"] += time.time() - t0_a_envs
            # Update environments with sampled actions
            envs, actions, valids = self.step(envs, actions, backward=True)
            # Add to batch
            batch_replay.add_to_batch(envs, actions, valids, backward=True, train=train)
            # Filter out finished trajectories
            envs = [env for env in envs if not env.equal(env.state, env.source)]
        times["replay_actions"] = time.time() - t0_replay

        # Merge forward and backward batches
        batch = batch.merge([batch_forward, batch_train, batch_replay])

        times["all"] = time.time() - t0_all

        return batch, times

    def compute_logprobs_trajectories(self, batch: Batch, backward: bool = False):
        """
        Computes the forward or backward log probabilities of the trajectories in a
        batch.

        Args
        ----
        batch : Batch
            A batch of data, containing all the states in the trajectories.

        backward : bool
            False: log probabilities of forward trajectories.
            True: log probabilities of backward trajectories.

        Returns
        -------
        logprobs : torch.tensor
            The log probabilities of the trajectories.
        """
        assert batch.is_valid()
        # Make indices of batch consecutive since they are used for indexing here
        # Get necessary tensors from batch
        states_policy = batch.get_states(policy=True)
        states = batch.get_states(policy=False)
        actions = batch.get_actions()
        parents_policy = batch.get_parents(policy=True)
        parents = batch.get_parents(policy=False)
        traj_indices = batch.get_trajectory_indices(consecutive=True)
        if backward:
            # Backward trajectories
            masks_b = batch.get_masks_backward()
            policy_output_b = self.backward_policy(states_policy)
            logprobs_states = self.env.get_logprobs(
                policy_output_b, actions, masks_b, states, backward
            )
        else:
            # Forward trajectories
            masks_f = batch.get_masks_forward(of_parents=True)
            policy_output_f = self.forward_policy(parents_policy)
            logprobs_states = self.env.get_logprobs(
                policy_output_f, actions, masks_f, parents, backward
            )
        # Sum log probabilities of all transitions in each trajectory
        logprobs = torch.zeros(
            batch.get_n_trajectories(),
            dtype=self.float,
            device=self.device,
        ).index_add_(0, traj_indices, logprobs_states)
        return logprobs

    def flowmatch_loss(self, it, batch):
        """
        Computes the loss of a batch

        Args
        ----
        it : int
            Iteration

        batch : ndarray
            A batch of data: every row is a state (list), corresponding to all states
            visited in each state in the batch.

        Returns
        -------
        loss : float
            Loss, as per Equation 12 of https://arxiv.org/abs/2106.04399v1

        term_loss : float
            Loss of the terminal nodes only

        flow_loss : float
            Loss of the intermediate nodes only
        """
        assert batch.is_valid()
        # Get necessary tensors from batch
        states = batch.get_states(policy=True)
        parents, parents_actions, parents_state_idx = batch.get_parents_all(policy=True)
        done = batch.get_done()
        masks_sf = batch.get_masks_forward()
        parents_a_idx = self.env.actions2indices(parents_actions)
        rewards = batch.get_rewards()
        assert torch.all(rewards[done] > 0)
        # In-flows
        inflow_logits = torch.full(
            (states.shape[0], self.env.policy_output_dim),
            -torch.inf,
            dtype=self.float,
            device=self.device,
        )
        inflow_logits[parents_state_idx, parents_a_idx] = self.forward_policy(parents)[
            torch.arange(parents.shape[0]), parents_a_idx
        ]
        inflow = torch.logsumexp(inflow_logits, dim=1)
        # Out-flows
        outflow_logits = self.forward_policy(states)
        outflow_logits[masks_sf] = -torch.inf
        outflow = torch.logsumexp(outflow_logits, dim=1)
        # Loss at terminating nodes
        loss_term = (inflow[done] - torch.log(rewards[done])).pow(2).mean()
        contrib_term = done.eq(1).to(self.float).mean()
        # Loss at intermediate nodes
        loss_interm = (inflow[~done] - outflow[~done]).pow(2).mean()
        contrib_interm = done.eq(0).to(self.float).mean()
        # Combined loss
        loss = contrib_term * loss_term + contrib_interm * loss_interm
        return loss, loss_term, loss_interm

    def trajectorybalance_loss(self, it, batch):
        """
        Computes the trajectory balance loss of a batch.

        Args
        ----
        it : int
            Iteration

        batch : Batch
            A batch of data, containing all the states in the trajectories.

        Returns
        -------
        loss : float

        term_loss : float
            Loss of the terminal nodes only

        flow_loss : float
            Loss of the intermediate nodes only
        """
        # Get logprobs of forward and backward transitions
        logprobs_f = self.compute_logprobs_trajectories(batch, backward=False)
        logprobs_b = self.compute_logprobs_trajectories(batch, backward=True)
        # Get rewards from batch
        rewards = batch.get_terminating_rewards(sort_by="trajectory")

        # Trajectory balance loss
        loss = (
            (self.logZ.sum() + logprobs_f - logprobs_b - torch.log(rewards))
            .pow(2)
            .mean()
        )
        return loss, loss, loss

    def detailedbalance_loss(self, it, batch):
        """
        Computes the Detailed Balance GFlowNet loss of a batch
        Reference : https://arxiv.org/pdf/2201.13259.pdf (eq 11)

        Args
        ----
        it : int
            Iteration

        batch : Batch
            A batch of data, containing all the states in the trajectories.


        Returns
        -------
        loss : float

        term_loss : float
            Loss of the terminal nodes only

        nonterm_loss : float
            Loss of the intermediate nodes only
        """

        assert batch.is_valid()
        # Get necessary tensors from batch
        states = batch.get_states(policy=False)
        states_policy = batch.get_states(policy=True)
        actions = batch.get_actions()
        parents = batch.get_parents(policy=False)
        parents_policy = batch.get_parents(policy=True)
        done = batch.get_done()
        rewards = batch.get_terminating_rewards(sort_by="insertion")

        # Get logprobs
        masks_f = batch.get_masks_forward(of_parents=True)
        policy_output_f = self.forward_policy(parents_policy)
        logprobs_f = self.env.get_logprobs(
            policy_output_f, actions, masks_f, parents, is_backward=False
        )
        masks_b = batch.get_masks_backward()
        policy_output_b = self.backward_policy(states_policy)
        logprobs_b = self.env.get_logprobs(
            policy_output_b, actions, masks_b, states, is_backward=True
        )

        # Get logflows
        logflows_states = self.state_flow(states_policy)
        logflows_states[done.eq(1)] = torch.log(rewards)
        # TODO: Optimise by reusing logflows_states and batch.get_parent_indices
        logflows_parents = self.state_flow(parents_policy)

        # Detailed balance loss
        loss_all = (logflows_parents + logprobs_f - logflows_states - logprobs_b).pow(2)
        loss = loss_all.mean()
        loss_terminating = loss_all[done].mean()
        loss_intermediate = loss_all[~done].mean()
        return loss, loss_terminating, loss_intermediate

    def forwardlooking_loss(self, it, batch):
        """
        Computes the Forward-Looking GFlowNet loss of a batch
        Reference : https://arxiv.org/pdf/2302.01687.pdf

        Args
        ----
        it : int
            Iteration

        batch : Batch
            A batch of data, containing all the states in the trajectories.


        Returns
        -------
        loss : float

        term_loss : float
            Loss of the terminal nodes only

        nonterm_loss : float
            Loss of the intermediate nodes only
        """

        assert batch.is_valid()
        # Get necessary tensors from batch
        states = batch.get_states(policy=False)
        states_policy = batch.get_states(policy=True)
        actions = batch.get_actions()
        parents = batch.get_parents(policy=False)
        parents_policy = batch.get_parents(policy=True)
        rewards_states = batch.get_rewards(do_non_terminating=True)
        rewards_parents = batch.get_rewards_parents()
        done = batch.get_done()

        # Get logprobs
        masks_f = batch.get_masks_forward(of_parents=True)
        policy_output_f = self.forward_policy(parents_policy)
        logprobs_f = self.env.get_logprobs(
            policy_output_f, actions, masks_f, parents, is_backward=False
        )
        masks_b = batch.get_masks_backward()
        policy_output_b = self.backward_policy(states_policy)
        logprobs_b = self.env.get_logprobs(
            policy_output_b, actions, masks_b, states, is_backward=True
        )

        # Get FL logflows
        logflflows_states = self.state_flow(states_policy)
        # Log FL flow of terminal states is 0 (eq. 9 of paper)
        logflflows_states[done.eq(1)] = 0.0
        # TODO: Optimise by reusing logflows_states and batch.get_parent_indices
        logflflows_parents = self.state_flow(parents_policy)

        # Get energies transitions
        energies_transitions = torch.log(rewards_parents) - torch.log(rewards_states)

        # Forward-looking loss
        loss_all = (
            logflflows_parents
            - logflflows_states
            + logprobs_f
            - logprobs_b
            + energies_transitions
        ).pow(2)
        loss = loss_all.mean()
        loss_terminating = loss_all[done].mean()
        loss_intermediate = loss_all[~done].mean()
        return loss, loss_terminating, loss_intermediate

    @torch.no_grad()
    def estimate_logprobs_data(
        self,
        data: Union[List, str],
        n_trajectories: int = 1,
        max_iters_per_traj: int = 10,
        max_data_size: int = 1e5,
        batch_size: int = 100,
        bs_num_samples=10000,
    ):
        r"""
        Estimates the probability of sampling with current GFlowNet policy
        (self.forward_policy) the objects in a data set given by the argument data. The
        (log) probabilities are estimated by sampling a number of backward trajectories
        (n_trajectories) through importance sampling and calculating the forward
        probabilities of the trajectories.

        $$
        \log p_T(x) = \int_{x \in \tau} P_F(\tau)d\tau \\
        = \log \mathbb{E}_{P_B(\tau|x)} \frac{P_F(x)}{P_B(\tau|x)}\\
        \approx \log \frac{1}{N} \sum_{i=1}^{N} \frac{P_F(x_i)}{P_B(\tau|x_i)}\\
        = \log \sum_{i=1}^{N} \frac{P_F(x_i)}{P_B(\tau|x_i)} - \log N
        $$

        Note: torch.logsumexp is used to compute the log of the sum, in order to have
        numerical stability, since we have the log PF and log PB, instead of directly
        PF and PB.

        Note: the correct indexing of data points and trajectories is ensured by the
        fact that the indices of the environments are set in a consistent way with the
        indexing when storing the log probabilities.

        Args
        ----
        data : list or string
            A data set of terminating states. The data set may be passed directly as a
            list of states, or it may be a string defining the path to a pickled data
            set where the terminating states are stored in key "x".

        n_trajectories : int
            The number of trajectories per object to sample for estimating the log
            probabilities.

        max_iters_per_traj : int
            The maximum number of attempts to sample a distinct trajectory, to avoid
            getting trapped in an infinite loop.

        max_data_size : int
            Maximum number of data points in the data set to avoid an accidental
            situation of having to sample too many backward trajectories. If necessary,
            the user should change this argument manually.

        bs_num_samples: int
            Number of bootstrap resampling times for std estimation of logprobs_estimates.
            Doesn't require recomputing of log probabilities, so can be arbitrary large

        Returns
        -------
        logprobs_estimates: torch.tensor
            The logarithm of the average ratio PF/PB over n trajectories sampled for
            each data point.

        logprobs_std: torch.tensor
            Bootstrap std of the logprobs_estimates

        probs_std: torch.tensor
            Bootstrap std of the torch.exp(logprobs_estimates)
        """
        print("Compute logprobs...", flush=True)
        times = {}
        # Determine terminating states
        if isinstance(data, list):
            states_term = data
        elif isinstance(data, str) and Path(data).suffix == ".pkl":
            with open(data, "rb") as f:
                data_dict = pickle.load(f)
                states_term = data_dict["x"]
        else:
            raise NotImplementedError(
                "data must be either a list of states or a path to a .pkl file."
            )
        n_states = len(states_term)
        assert (
            n_states < max_data_size
        ), "The size of the test data is larger than max_data_size ({max_data_size})."

        # Compute log probabilities in batches
        logprobs_f = torch.full(
            (n_states, n_trajectories),
            -torch.inf,
            dtype=self.float,
            device=self.device,
        )
        logprobs_b = torch.full(
            (n_states, n_trajectories),
            -torch.inf,
            dtype=self.float,
            device=self.device,
        )
        mult_indices = max(n_states, n_trajectories)
        init_batch = 0
        end_batch = min(batch_size, n_states)
        print(
            "Sampling backward actions from test data to estimate logprobs...",
            flush=True,
        )
        pbar = tqdm(total=n_states)
        while init_batch < n_states:
            batch = Batch(env=self.env, device=self.device, float_type=self.float)
            # Create an environment for each data point and trajectory and set the state
            envs = []
            for state_idx in range(init_batch, end_batch):
                for traj_idx in range(n_trajectories):
                    idx = int(mult_indices * state_idx + traj_idx)
                    env = self.env.copy().reset(idx)
                    env.set_state(states_term[state_idx], done=True)
                    envs.append(env)
            # Sample trajectories
            max_iters = n_trajectories * max_iters_per_traj
            while envs:
                # Sample backward actions
                actions = self.sample_actions(
                    envs,
                    batch,
                    backward=True,
                    no_random=True,
                    times=times,
                )
                # Update environments with sampled actions
                envs, actions, valids = self.step(envs, actions, backward=True)
                assert all(valids)
                # Add to batch
                batch.add_to_batch(envs, actions, valids, backward=True, train=True)
                # Filter out finished trajectories
                envs = [env for env in envs if not env.equal(env.state, env.source)]
            # Prepare data structures to compute log probabilities
            traj_indices_batch = tlong(
                batch.get_unique_trajectory_indices(), device=self.device
            )
            data_indices = traj_indices_batch // mult_indices
            traj_indices = traj_indices_batch % mult_indices
            # Compute log probabilities of the trajectories
            logprobs_f[data_indices, traj_indices] = self.compute_logprobs_trajectories(
                batch, backward=False
            )
            logprobs_b[data_indices, traj_indices] = self.compute_logprobs_trajectories(
                batch, backward=True
            )
            # Increment batch indices
            init_batch += batch_size
            end_batch = min(end_batch + batch_size, n_states)
            pbar.update(end_batch - init_batch)

        # Compute log of the average probabilities of the ratio PF / PB
        logprobs_estimates = torch.logsumexp(
            logprobs_f - logprobs_b, dim=1
        ) - torch.log(torch.tensor(n_trajectories, device=self.device))
        logprobs_f_b_bs = bootstrap_samples(
            logprobs_f - logprobs_b, num_samples=bs_num_samples
        )
        logprobs_estimates_bs = torch.logsumexp(logprobs_f_b_bs, dim=1) - torch.log(
            torch.tensor(n_trajectories, device=self.device)
        )
        logprobs_std = torch.std(logprobs_estimates_bs, dim=-1)
        probs_std = torch.std(torch.exp(logprobs_estimates_bs), dim=-1)
        print("Done computing logprobs", flush=True)
        return logprobs_estimates, logprobs_std, probs_std

    def train(self):
        # Metrics
        all_losses = []
        all_visited = []
        loss_term_ema = None
        loss_flow_ema = None
        # Train loop
        pbar = tqdm(range(1, self.n_train_steps + 1), disable=not self.logger.progress)
        for it in pbar:
            # Test
            fig_names = [
                "True reward and GFlowNet samples",
                "GFlowNet KDE Policy",
                "Reward KDE",
            ]
            if self.logger.do_test(it):
                (
                    self.l1,
                    self.kl,
                    self.jsd,
                    self.corr_prob_traj_rewards,
                    self.var_logrewards_logp,
                    self.nll_tt,
                    self.mean_logprobs_std,
                    self.mean_probs_std,
                    self.logprobs_std_nll_ratio,
                    figs,
                    env_metrics,
                ) = self.test()
                self.logger.log_test_metrics(
                    self.l1,
                    self.kl,
                    self.jsd,
                    self.corr_prob_traj_rewards,
                    self.var_logrewards_logp,
                    self.nll_tt,
                    self.mean_logprobs_std,
                    self.mean_probs_std,
                    self.logprobs_std_nll_ratio,
                    it,
                    self.use_context,
                )
                self.logger.log_metrics(env_metrics, it, use_context=self.use_context)
                self.logger.log_plots(
                    figs, it, fig_names=fig_names, use_context=self.use_context
                )
            if self.logger.do_top_k(it):
                metrics, figs, fig_names, summary = self.test_top_k(it)
                self.logger.log_plots(
                    figs, it, use_context=self.use_context, fig_names=fig_names
                )
                self.logger.log_metrics(metrics, use_context=self.use_context, step=it)
                self.logger.log_summary(summary)
            t0_iter = time.time()
            batch = Batch(env=self.env, device=self.device, float_type=self.float)
            for j in range(self.sttr):
                sub_batch, times = self.sample_batch(
                    n_forward=self.batch_size.forward,
                    n_train=self.batch_size.backward_dataset,
                    n_replay=self.batch_size.backward_replay,
                )
                batch.merge(sub_batch)
            for j in range(self.ttsr):
                if self.loss == "flowmatch":
                    losses = self.flowmatch_loss(
                        it * self.ttsr + j, batch
                    )  # returns (opt loss, *metrics)
                elif self.loss == "trajectorybalance":
                    losses = self.trajectorybalance_loss(
                        it * self.ttsr + j, batch
                    )  # returns (opt loss, *metrics)
                elif self.loss == "detailedbalance":
                    losses = self.detailedbalance_loss(it * self.ttsr + j, batch)
                elif self.loss == "forwardlooking":
                    losses = self.forwardlooking_loss(it * self.ttsr + j, batch)
                else:
                    print("Unknown loss!")
                # TODO: deal with this in a better way
                if not all([torch.isfinite(loss) for loss in losses]):
                    if self.logger.debug:
                        print("Loss is not finite - skipping iteration")
                    if len(all_losses) > 0:
                        all_losses.append([loss for loss in all_losses[-1]])
                else:
                    losses[0].backward()
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.parameters(), self.clip_grad_norm
                        )
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()
                    all_losses.append([i.item() for i in losses])
            # Buffer
            t0_buffer = time.time()
            states_term = batch.get_terminating_states(sort_by="trajectory")
            rewards = batch.get_terminating_rewards(sort_by="trajectory")
            actions_trajectories = batch.get_actions_trajectories()
            proxy_vals = self.env.reward2proxy(rewards).tolist()
            rewards = rewards.tolist()
            self.buffer.add(states_term, actions_trajectories, rewards, proxy_vals, it)
            self.buffer.add(
                states_term,
                actions_trajectories,
                rewards,
                proxy_vals,
                it,
                buffer="replay",
            )
            t1_buffer = time.time()
            times.update({"buffer": t1_buffer - t0_buffer})
            # Log
            if self.logger.lightweight:
                all_losses = all_losses[-100:]
                all_visited = states_term
            else:
                all_visited.extend(states_term)
            # Progress bar
            self.logger.progressbar_update(
                pbar, all_losses, rewards, self.jsd, it, self.use_context
            )
            # Train logs
            t0_log = time.time()
            self.logger.log_train(
                losses=losses,
                rewards=rewards,
                proxy_vals=proxy_vals,
                states_term=states_term,
                batch_size=len(batch),
                logz=self.logZ,
                learning_rates=self.lr_scheduler.get_last_lr(),
                step=it,
                use_context=self.use_context,
            )
            t1_log = time.time()
            times.update({"log": t1_log - t0_log})
            # Save intermediate models
            t0_model = time.time()
            self.logger.save_models(
                self.forward_policy, self.backward_policy, self.state_flow, step=it
            )
            t1_model = time.time()
            times.update({"save_interim_model": t1_model - t0_model})

            # Moving average of the loss for early stopping
            if loss_term_ema and loss_flow_ema:
                loss_term_ema = (
                    self.ema_alpha * losses[1].item()
                    + (1.0 - self.ema_alpha) * loss_term_ema
                )
                loss_flow_ema = (
                    self.ema_alpha * losses[2].item()
                    + (1.0 - self.ema_alpha) * loss_flow_ema
                )
                if (
                    loss_term_ema < self.early_stopping
                    and loss_flow_ema < self.early_stopping
                ):
                    break
            else:
                loss_term_ema = losses[1].item()
                loss_flow_ema = losses[2].item()

            # Log times
            t1_iter = time.time()
            times.update({"iter": t1_iter - t0_iter})
            self.logger.log_time(times, use_context=self.use_context)

        # Save final model
        self.logger.save_models(
            self.forward_policy, self.backward_policy, self.state_flow, final=True
        )
        # Close logger
        if self.use_context is False:
            self.logger.end()

    def test(self, **plot_kwargs):
        """
        Computes metrics by sampling trajectories from the forward policy.
        """
        if self.buffer.test_pkl is None:
            return (
                self.l1,
                self.kl,
                self.jsd,
                self.corr_prob_traj_rewards,
                self.var_logrewards_logp,
                self.nll_tt,
                self.mean_logprobs_std,
                self.mean_probs_std,
                self.logprobs_std_nll_ratio,
                (None,),
                {},
            )
        with open(self.buffer.test_pkl, "rb") as f:
            dict_tt = pickle.load(f)
            x_tt = dict_tt["x"]

        # Compute correlation between the rewards of the test data and the log
        # likelihood of the data according the the GFlowNet policy; and NLL.
        # TODO: organise code for better efficiency and readability
        logprobs_x_tt, logprobs_std, probs_std = self.estimate_logprobs_data(
            x_tt,
            n_trajectories=self.logger.test.n_trajs_logprobs,
            max_data_size=self.logger.test.max_data_logprobs,
            batch_size=self.logger.test.logprobs_batch_size,
            bs_num_samples=self.logger.test.logprobs_bootstrap_size,
        )
        mean_logprobs_std = logprobs_std.mean().item()
        mean_probs_std = probs_std.mean().item()
        rewards_x_tt = self.env.reward_batch(x_tt)
        corr_prob_traj_rewards = np.corrcoef(
            np.exp(logprobs_x_tt.cpu().numpy()), rewards_x_tt
        )[0, 1]
        var_logrewards_logp = torch.var(
            torch.log(tfloat(rewards_x_tt, float_type=self.float, device=self.device))
            - logprobs_x_tt
        ).item()
        nll_tt = -logprobs_x_tt.mean().item()
        logprobs_std_nll_ratio = torch.mean(-logprobs_std / logprobs_x_tt).item()

        x_sampled = []
        if self.buffer.test_type is not None and self.buffer.test_type == "all":
            batch, _ = self.sample_batch(n_forward=self.logger.test.n, train=False)
            assert batch.is_valid()
            x_sampled = batch.get_terminating_states()

            if "density_true" in dict_tt:
                density_true = dict_tt["density_true"]
            else:
                rewards = self.env.reward_batch(x_tt)
                z_true = rewards.sum()
                density_true = rewards / z_true
                with open(self.buffer.test_pkl, "wb") as f:
                    dict_tt["density_true"] = density_true
                    pickle.dump(dict_tt, f)
            hist = defaultdict(int)
            for x in x_sampled:
                hist[tuple(x)] += 1
            z_pred = sum([hist[tuple(x)] for x in x_tt]) + 1e-9
            density_pred = np.array([hist[tuple(x)] / z_pred for x in x_tt])
            log_density_true = np.log(density_true + 1e-8)
            log_density_pred = np.log(density_pred + 1e-8)
        elif self.continuous and hasattr(self.env, "fit_kde"):
            batch, _ = self.sample_batch(n_forward=self.logger.test.n, train=False)
            assert batch.is_valid()
            x_sampled = batch.get_terminating_states()
            # TODO make it work with conditional env
            x_sampled = torch2np(self.env.states2proxy(x_sampled))
            x_tt = torch2np(self.env.states2proxy(x_tt))
            kde_pred = self.env.fit_kde(
                x_sampled,
                kernel=self.logger.test.kde.kernel,
                bandwidth=self.logger.test.kde.bandwidth,
            )
            if "log_density_true" in dict_tt and "kde_true" in dict_tt:
                log_density_true = dict_tt["log_density_true"]
                kde_true = dict_tt["kde_true"]
            else:
                # Sample from reward via rejection sampling
                x_from_reward = self.env.sample_from_reward(
                    n_samples=self.logger.test.n
                )
                x_from_reward = torch2np(self.env.states2proxy(x_from_reward))
                # Fit KDE with samples from reward
                kde_true = self.env.fit_kde(
                    x_from_reward,
                    kernel=self.logger.test.kde.kernel,
                    bandwidth=self.logger.test.kde.bandwidth,
                )
                # Estimate true log density using test samples
                # TODO: this may be specific-ish for the torus or not
                scores_true = kde_true.score_samples(x_tt)
                log_density_true = scores_true - logsumexp(scores_true, axis=0)
                # Add log_density_true and kde_true to pickled test dict
                with open(self.buffer.test_pkl, "wb") as f:
                    dict_tt["log_density_true"] = log_density_true
                    dict_tt["kde_true"] = kde_true
                    pickle.dump(dict_tt, f)
            # Estimate pred log density using test samples
            # TODO: this may be specific-ish for the torus or not
            scores_pred = kde_pred.score_samples(x_tt)
            log_density_pred = scores_pred - logsumexp(scores_pred, axis=0)
            density_true = np.exp(log_density_true)
            density_pred = np.exp(log_density_pred)
        else:
            # TODO: refactor
            env_metrics = self.env.test(x_sampled)
            return (
                self.l1,
                self.kl,
                self.jsd,
                corr_prob_traj_rewards,
                var_logrewards_logp,
                nll_tt,
                mean_logprobs_std,
                mean_probs_std,
                logprobs_std_nll_ratio,
                (None,),
                env_metrics,
            )
        # L1 error
        l1 = np.abs(density_pred - density_true).mean()
        # KL divergence
        kl = (density_true * (log_density_true - log_density_pred)).mean()
        # Jensen-Shannon divergence
        log_mean_dens = np.logaddexp(log_density_true, log_density_pred) + np.log(0.5)
        jsd = 0.5 * np.sum(density_true * (log_density_true - log_mean_dens))
        jsd += 0.5 * np.sum(density_pred * (log_density_pred - log_mean_dens))

        # Plots

        if hasattr(self.env, "plot_reward_samples"):
            fig_reward_samples = self.env.plot_reward_samples(x_sampled, **plot_kwargs)
        else:
            fig_reward_samples = None
        if hasattr(self.env, "plot_kde"):
            fig_kde_pred = self.env.plot_kde(kde_pred, **plot_kwargs)
            fig_kde_true = self.env.plot_kde(kde_true, **plot_kwargs)
        else:
            fig_kde_pred = None
            fig_kde_true = None
        return (
            l1,
            kl,
            jsd,
            corr_prob_traj_rewards,
            var_logrewards_logp,
            nll_tt,
            mean_logprobs_std,
            mean_probs_std,
            logprobs_std_nll_ratio,
            [fig_reward_samples, fig_kde_pred, fig_kde_true],
            {},
        )

    @torch.no_grad()
    def test_top_k(self, it, progress=False, gfn_states=None, random_states=None):
        """
        Sample from the current GFN and compute metrics and plots for the top k states
        according to both the energy and the reward.

        Parameters
        ----------
        it : int
            Current iteration.
        progress : bool, optional
            Print sampling progress. Defaults to False.
        gfn_states : list, optional
            Already sampled gfn states. Defaults to None.
        random_states : list, optional
            Already sampled random states. Defaults to None.

        Returns
        -------
        tuple[dict, list[plt.Figure], list[str], dict]
            Computed dict of metrics, and figures, their names and optionally (only
            once) summary metrics.
        """
        # only do random top k plots & metrics once
        do_random = it // self.logger.test.top_k_period == 1
        duration = None
        summary = {}
        prob = copy.deepcopy(self.random_action_prob)
        print()
        if not gfn_states:
            # sample states from the current gfn
            batch = Batch(env=self.env, device=self.device, float_type=self.float)
            self.random_action_prob = 0
            t = time.time()
            print("Sampling from GFN...", end="\r")
            for b in batch_with_rest(
                0, self.logger.test.n_top_k, self.batch_size_total
            ):
                sub_batch, _ = self.sample_batch(n_forward=len(b), train=False)
                batch.merge(sub_batch)
            duration = time.time() - t
            gfn_states = batch.get_terminating_states()

        # compute metrics and get plots
        print("[test_top_k] Making GFN plots...", end="\r")
        metrics, figs, fig_names = self.env.top_k_metrics_and_plots(
            gfn_states, self.logger.test.top_k, name="gflownet", step=it
        )
        if duration:
            metrics["gflownet top k sampling duration"] = duration

        if do_random:
            # sample random states from uniform actions
            if not random_states:
                batch = Batch(env=self.env, device=self.device, float_type=self.float)
                self.random_action_prob = 1.0
                print("[test_top_k] Sampling at random...", end="\r")
                for b in batch_with_rest(
                    0, self.logger.test.n_top_k, self.batch_size_total
                ):
                    sub_batch, _ = self.sample_batch(n_forward=len(b), train=False)
                    batch.merge(sub_batch)
            # compute metrics and get plots
            random_states = batch.get_terminating_states()
            print("[test_top_k] Making Random plots...", end="\r")
            (
                random_metrics,
                random_figs,
                random_fig_names,
            ) = self.env.top_k_metrics_and_plots(
                random_states, self.logger.test.top_k, name="random", step=None
            )
            # add to current metrics and plots
            summary.update(random_metrics)
            figs += random_figs
            fig_names += random_fig_names
            # compute training data metrics and get plots
            print("[test_top_k] Making train plots...", end="\r")
            (
                train_metrics,
                train_figs,
                train_fig_names,
            ) = self.env.top_k_metrics_and_plots(
                None, self.logger.test.top_k, name="train", step=None
            )
            # add to current metrics and plots
            summary.update(train_metrics)
            figs += train_figs
            fig_names += train_fig_names

        self.random_action_prob = prob

        print(" " * 100, end="\r")
        print("test_top_k metrics:")
        max_k = max([len(k) for k in (list(metrics.keys()) + list(summary.keys()))]) + 1
        print(
            "  •  "
            + "\n  •  ".join(
                f"{k:{max_k}}: {v:.4f}"
                for k, v in (list(metrics.items()) + list(summary.items()))
            )
        )
        print()
        return metrics, figs, fig_names, summary

    def get_log_corr(self, times):
        data_logq = []
        times.update(
            {
                "test_trajs": 0.0,
                "test_logq": 0.0,
            }
        )
        # TODO: this could be done just once and store it
        for statestr, score in tqdm(
            zip(self.buffer.test.samples, self.buffer.test["energies"]), disable=True
        ):
            t0_test_traj = time.time()
            traj_list, actions = self.env.get_trajectories(
                [],
                [],
                [self.env.readable2state(statestr)],
                [self.env.eos],
            )
            t1_test_traj = time.time()
            times["test_trajs"] += t1_test_traj - t0_test_traj
            t0_test_logq = time.time()
            data_logq.append(logq(traj_list, actions, self.forward_policy, self.env))
            t1_test_logq = time.time()
            times["test_logq"] += t1_test_logq - t0_test_logq
        corr = np.corrcoef(data_logq, self.buffer.test["energies"])
        return corr, data_logq, times

    # TODO: reorganize and remove
    def log_iter(
        self,
        pbar,
        rewards,
        proxy_vals,
        states_term,
        data,
        it,
        times,
        losses,
        all_losses,
        all_visited,
    ):
        # train metrics
        self.logger.log_sampler_train(
            rewards, proxy_vals, states_term, data, it, self.use_context
        )

        # logZ
        self.logger.log_metric("logZ", self.logZ.sum(), it, use_context=False)

        # test metrics
        # TODO: integrate corr into test()
        if not self.logger.lightweight and self.buffer.test is not None:
            corr, data_logq, times = self.get_log_corr(times)
            self.logger.log_sampler_test(corr, data_logq, it, self.use_context)

        # oracle metrics
        oracle_batch, oracle_times = self.sample_batch(
            n_forward=self.oracle_n, train=False
        )

        if not self.logger.lightweight:
            self.logger.log_metric(
                "unique_states",
                np.unique(all_visited).shape[0],
                step=it,
                use_context=self.use_context,
            )


def make_opt(params, logZ, config):
    """
    Set up the optimizer
    """
    params = params
    if not len(params):
        return None
    if config.method == "adam":
        opt = torch.optim.Adam(
            params,
            config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
        )
        if logZ is not None:
            opt.add_param_group(
                {
                    "params": logZ,
                    "lr": config.lr * config.lr_z_mult,
                }
            )
    elif config.method == "msgd":
        opt = torch.optim.SGD(params, config.lr, momentum=config.momentum)
    # Learning rate scheduling
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=config.lr_decay_period,
        gamma=config.lr_decay_gamma,
    )
    return opt, lr_scheduler


def logq(traj_list, actions_list, model, env):
    # TODO: this method is probably suboptimal, since it may repeat forward calls for
    # the same nodes.
    log_q = torch.tensor(1.0)
    for traj, actions in zip(traj_list, actions_list):
        traj = traj[::-1]
        actions = actions[::-1]
        masks = tbool(
            [env.get_mask_invalid_actions_forward(state, 0) for state in traj],
            device=self.device,
        )
        with torch.no_grad():
            logits_traj = model(
                tfloat(
                    env.states2policy(traj),
                    device=self.device,
                    float_type=self.float,
                )
            )
        logits_traj[masks] = -torch.inf
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        logprobs_traj = logsoftmax(logits_traj)
        log_q_traj = torch.tensor(0.0)
        for s, a, logprobs in zip(*[traj, actions, logprobs_traj]):
            log_q_traj = log_q_traj + logprobs[a]
        # Accumulate log prob of trajectory
        if torch.le(log_q, 0.0):
            log_q = torch.logaddexp(log_q, log_q_traj)
        else:
            log_q = log_q_traj
    return log_q.item()
