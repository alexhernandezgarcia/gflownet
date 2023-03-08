"""
GFlowNet
TODO:
    - Seeds
"""
import sys
import copy
import time
from collections import defaultdict
from pathlib import Path
from omegaconf import OmegaConf
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import pickle
from torch.distributions import Categorical, Bernoulli
from tqdm import tqdm
from scipy.special import logsumexp

from gflownet.envs.base import Buffer
from gflownet.utils.common import set_device, set_float_precision, torch2np


class GFlowNetAgent:
    def __init__(
        self,
        env,
        seed,
        device,
        float_precision,
        optimizer,
        buffer,
        policy,
        mask_invalid_actions,
        temperature_logits,
        random_action_prob,
        pct_offline,
        logger,
        num_empirical_loss,
        oracle,
        proxy=None,
        active_learning=False,
        data_path=None,
        sample_only=False,
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
        self.mask_source = self._tbool([self.env.get_mask_invalid_actions_forward()])
        # Continuous environments
        self.continuous = hasattr(self.env, "continuous") and self.env.continuous
        if self.continuous and optimizer.loss in ["flowmatch", "flowmatching"]:
            print(
                """
            Flow matching loss is not available for continuous environments.
            Trajectory balance will be used instead
            """
            )
            optimizer.loss = "tb"
        # Loss
        if optimizer.loss in ["flowmatch", "flowmatching"]:
            self.loss = "flowmatch"
            self.logZ = None
        elif optimizer.loss in ["trajectorybalance", "tb"]:
            self.loss = "trajectorybalance"
            self.logZ = nn.Parameter(torch.ones(optimizer.z_dim) * 150.0 / 64)
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
        # Policy models
        self.forward_policy = Policy(policy.forward, self.env, self.device, self.float)
        if "checkpoint" in policy.forward and policy.forward.checkpoint:
            self.logger.set_forward_policy_ckpt_path(policy.forward.checkpoint)
            # TODO: re-write the logic and conditions to reload a model
            if False:
                self.forward_policy.load_state_dict(
                    torch.load(self.policy_forward_path)
                )
                print("Reloaded GFN forward policy model Checkpoint")
        else:
            self.logger.set_forward_policy_ckpt_path(None)
        self.backward_policy = Policy(
            policy.backward,
            self.env,
            self.device,
            self.float,
            base=self.forward_policy,
        )
        if (
            policy.backward
            and "checkpoint" in policy.backward
            and policy.backward.checkpoint
        ):
            self.logger.set_backward_policy_ckpt_path(policy.backward.checkpoint)
            # TODO: re-write the logic and conditions to reload a model
            if False:
                self.backward_policy.load_state_dict(
                    torch.load(self.policy_backward_path)
                )
                print("Reloaded GFN backward policy model Checkpoint")
        else:
            self.logger.set_backward_policy_ckpt_path(None)
        self.ckpt_period = policy.ckpt_period
        if self.ckpt_period in [None, -1]:
            self.ckpt_period = np.inf
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

    def _tfloat(self, x):
        return torch.tensor(x, dtype=self.float, device=self.device)

    def _tlong(self, x):
        return torch.tensor(x, dtype=torch.long, device=self.device)

    def _tint(self, x):
        return torch.tensor(x, dtype=torch.int, device=self.device)

    def _tbool(self, x):
        return torch.tensor(x, dtype=torch.bool, device=self.device)

    def parameters(self):
        if self.backward_policy.is_model == False:
            return list(self.forward_policy.model.parameters())
        elif self.loss == "trajectorybalance":
            return list(self.forward_policy.model.parameters()) + list(
                self.backward_policy.model.parameters()
            )
        else:
            raise ValueError("Backward Policy cannot be a nn in flowmatch.")

    def sample_actions(
        self,
        envs,
        times,
        sampling_method="policy",
        model=None,
        is_forward: bool = True,
        temperature=1.0,
        random_action_prob=0.0,
    ):
        """
        Samples one action on each environment of a list.

        Args
        ----
        envs : list of GFlowNetEnv or derived
            A list of instances of the environment

        times : dict
            Dictionary to store times

        sampling_method : string
            - model: uses current forward to obtain the sampling probabilities.
            - uniform: samples uniformly from the action space.

        model : torch model
            Model to use as policy if sampling_method="policy"

        is_forward : bool
            True if sampling is forward. False if backward.

        temperature : float
            Temperature to adjust the logits by logits /= temperature
        """
        # TODO: implement backward sampling from forward policy as in old
        # backward_sample.
        if sampling_method == "random":
            random_action_prob = 1.0
        if not isinstance(envs, list):
            envs = [envs]
        # Build states and masks
        states = [env.state for env in envs]
        if is_forward:
            mask_invalid_actions = self._tbool(
                [env.get_mask_invalid_actions_forward() for env in envs]
            )
        else:
            mask_invalid_actions = self._tbool(
                [env.get_mask_invalid_actions_backward() for env in envs]
            )
        # Build policy outputs
        policy_outputs = model.random_distribution(states)
        idx_norandom = (
            Bernoulli(
                (1 - random_action_prob) * torch.ones(len(states), device=self.device)
            )
            .sample()
            .to(bool)
        )
        # Check for at least one non-random action
        if sampling_method == "policy" and idx_norandom.sum() > 0:
            policy_outputs[idx_norandom, :] = model(
                self._tfloat(
                    self.env.statebatch2policy(
                        [s for s, do in zip(states, idx_norandom) if do]
                    )
                )
            )
        else:
            raise NotImplementedError
        # Sample actions from policy outputs
        actions, logprobs = self.env.sample_actions(
            policy_outputs,
            sampling_method,
            mask_invalid_actions,
            temperature,
        )
        return actions

    def step(
        self,
        envs: List,
        actions: List[Tuple],
        is_forward: bool = True,
    ):
        """
        Executes the actions of a list on the environments of a list.

        Args
        ----
        envs : list of GFlowNetEnv or derived
            A list of instances of the environment

        actions : list
            A list of actions to be executed on each env of envs.

        is_forward : bool
            True if sampling is forward. False if backward.

        temperature : float
            Temperature to adjust the logits by logits /= temperature
        """
        assert len(envs) == len(actions)
        if not isinstance(envs, list):
            envs = [envs]
        if is_forward:
            _, actions, valids = zip(
                *[env.step(action) for env, action in zip(envs, actions)]
            )
        else:
            valids = []
            for env, action in zip(envs, actions):
                parents, parents_a = env.get_parents(action=action)
                if action in parents_a:
                    state_next = parents[parents_a.index(action)]
                    env.set_state(state_next, done=False)
                    env.n_actions -= 1
                    valids.append(True)
                else:
                    valids.append(False)
        return envs, actions, valids

    # @profile
    def sample_batch(
        self, envs, n_samples=None, train=True, model=None, progress=False
    ):
        """
        Builds a batch of data

        if train == True:
            Each item in the batch is a list of 7 elements (all tensors):
                - [0] the state
                - [1] the action
                - [2] all parents of the state, parents
                - [3] actions that lead to the state from each parent, parents_a
                - [4] done [True, False]
                - [5] traj id: identifies each trajectory
                - [6] state id: identifies each state within a traj
                - [7] mask_f: invalid forward actions from that state are 1
                - [8] mask_b: invalid backward actions from that state are 1
        else:
            Each item in the batch is a list of 1 element:
                - [0] the states (state)

        Args
        ----
        """

        def _add_to_batch(batch, envs, actions, valids, train=True):
            for env, action, valid in zip(envs, actions, valids):
                if valid is False:
                    continue
                parents, parents_a = env.get_parents(action=action)
                mask_f = env.get_mask_invalid_actions_forward()
                mask_b = env.get_mask_invalid_actions_backward(
                    env.state, env.done, parents_a
                )
                assert action in parents_a
                if train:
                    batch.append(
                        [
                            self._tfloat([env.state]),
                            self._tfloat([action]),
                            self._tfloat(parents),
                            self._tfloat(parents_a),
                            self._tbool([env.done]),
                            self._tlong([env.id] * len(parents)),
                            self._tlong([env.n_actions]),
                            self._tbool([mask_f]),
                            self._tbool([mask_b]),
                        ]
                    )
                else:
                    if env.done:
                        batch.append(env.state)
            return batch

        times = {
            "all": 0.0,
            "forward_actions": 0.0,
            "backward_actions": 0.0,
            "actions_envs": 0.0,
            "rewards": 0.0,
        }
        t0_all = time.time()
        batch = []
        if isinstance(envs, list):
            envs = [env.reset(idx) for idx, env in enumerate(envs)]
        elif n_samples is not None and n_samples > 0:
            envs = [self.env.copy().reset(idx) for idx in range(n_samples)]
        else:
            return None, None
        # Offline trajectories
        if train:
            n_empirical = int(self.pct_offline * len(envs))
        else:
            n_empirical = 0
        if n_empirical > 0 and self.buffer.train_pkl is not None:
            with open(self.buffer.train_pkl, "rb") as f:
                dict_tr = pickle.load(f)
                # TODO: implement other sampling options besides permutation
                x_tr = self.rng.permutation(dict_tr["x"])
            envs_offline = []
            actions = []
            valids = []
            for idx in range(n_empirical):
                env = envs[idx]
                env = env.set_state(x_tr[idx].tolist(), done=True)
                env.n_actions = env.get_max_traj_len()
                envs_offline.append(env)
                actions.append((env.eos,))
                valids.append(True)
        else:
            envs_offline = []
        while envs_offline:
            # Sample backward actions
            with torch.no_grad():
                actions = self.sample_actions(
                    envs_offline,
                    times,
                    sampling_method="policy",
                    model=self.backward_policy,
                    is_forward=False,
                    temperature=self.temperature_logits,
                    random_action_prob=self.random_action_prob,
                )
            # Add to batch
            batch = _add_to_batch(batch, envs_offline, actions, valids)
            # Update environments with sampled actions
            envs_offline, actions, valids = self.step(
                envs_offline, actions, is_forward=False
            )
            assert all(valids)
            # Filter out finished trajectories
            envs_offline = [env for env in envs_offline if env.state != env.source]
        envs = envs[n_empirical:]
        # Policy trajectories
        while envs:
            # Sample forward actions
            with torch.no_grad():
                if train is False:
                    actions = self.sample_actions(
                        envs,
                        times,
                        sampling_method="policy",
                        model=self.forward_policy,
                        is_forward=True,
                        temperature=1.0,
                        random_action_prob=self.random_action_prob,
                    )
                else:
                    actions = self.sample_actions(
                        envs,
                        times,
                        sampling_method="policy",
                        model=self.forward_policy,
                        is_forward=True,
                        temperature=self.temperature_logits,
                        random_action_prob=self.random_action_prob,
                    )
            # Update environments with sampled actions
            envs, actions, valids = self.step(envs, actions, is_forward=True)
            # Add to batch
            t0_a_envs = time.time()
            batch = _add_to_batch(batch, envs, actions, valids, train)
            # Filter out finished trajectories
            envs = [env for env in envs if not env.done]
            t1_a_envs = time.time()
            times["actions_envs"] += t1_a_envs - t0_a_envs
            if progress and n_samples is not None:
                print(f"{n_samples - len(envs)}/{n_samples} done")
        return batch, times

    def flowmatch_loss(self, it, batch, loginf=1000):
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
        loginf = self._tfloat([loginf])
        # Unpack batch
        (
            states,
            actions,
            parents,
            parents_a,
            done,
            traj_id_parents,
            state_id,
            masks_sf,
            masks_b,
        ) = zip(*batch)
        # Get state/batch id
        parents_batch_id = self._tlong(
            sum([[idx] * len(p) for idx, p in enumerate(parents)], [])
        )
        # Concatenate lists of tensors
        states, parents, parents_a, done, masks_sf = map(
            torch.cat,
            [
                states,
                parents,
                parents_a,
                done,
                masks_sf,
            ],
        )
        parents_a = parents_a.to(int).squeeze()
        # Compute rewards
        rewards = self.env.reward_torchbatch(states, done)
        assert torch.all(rewards[done] > 0)
        # In-flows
        inflow_logits = -loginf * torch.ones(
            (states.shape[0], self.env.policy_output_dim),
            device=self.device,
        )
        inflow_logits[parents_batch_id, parents_a] = self.forward_policy(
            self.env.statetorch2policy(parents)
        )[torch.arange(parents.shape[0]), parents_a]
        inflow = torch.logsumexp(inflow_logits, dim=1)
        # Out-flows
        outflow_logits = self.forward_policy(self.env.statetorch2policy(states))
        outflow_logits[masks_sf] = -loginf
        outflow = torch.logsumexp(outflow_logits, dim=1)
        outflow = outflow * torch.logical_not(done) - loginf * done
        outflow = torch.logaddexp(torch.log(rewards), outflow)
        # Flow matching loss
        loss = (inflow - outflow).pow(2).mean()
        # Isolate loss at terminating nodes and all other nodes
        with torch.no_grad():
            term_loss = ((inflow - outflow) * done).pow(2).sum() / (done.sum() + 1e-20)
            flow_loss = ((inflow - outflow) * torch.logical_not(done)).pow(2).sum() / (
                torch.logical_not(done).sum() + 1e-20
            )
        return (loss, term_loss, flow_loss), rewards[done.eq(1)]

    def trajectorybalance_loss(self, it, batch, loginf=1000):
        """
        Computes the trajectory balance loss of a batch

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

        term_loss : float
            Loss of the terminal nodes only

        flow_loss : float
            Loss of the intermediate nodes only
        """
        loginf = self._tfloat([loginf])
        # Unpack batch
        (
            states,
            actions,
            parents,
            parents_a,
            done,
            traj_id_parents,
            state_id,
            masks_sf,
            masks_b,
        ) = zip(*batch)
        # Keep only parents in trajectory
        parents = [
            p[torch.where(torch.all(torch.eq(a, p_a), axis=1))]
            for a, p, p_a in zip(actions, parents, parents_a)
        ]
        traj_id = torch.cat([el[:1] for el in traj_id_parents])
        # Concatenate lists of tensors
        states, actions, parents, done, state_id, masks_sf, masks_b = map(
            torch.cat,
            [
                states,
                actions,
                parents,
                done,
                state_id,
                masks_sf,
                masks_b,
            ],
        )
        # Shift state_id to [1, 2, ...]
        for tid in traj_id.unique():
            state_id[traj_id == tid] -= state_id[traj_id == tid].min() + 1
        # Compute rewards
        rewards = self.env.reward_torchbatch(states, done)
        # Build parents forward masks from state masks
        masks_f = torch.cat(
            [
                masks_sf[torch.where((state_id == sid - 1) & (traj_id == pid))]
                if sid > 1
                else self.mask_source
                for sid, pid in zip(state_id, traj_id)
            ]
        )
        # Forward trajectories
        policy_output_f = self.forward_policy(self.env.statetorch2policy(parents))
        logprobs_f = self.env.get_logprobs(
            policy_output_f, True, actions, states, masks_f, loginf
        )
        sumlogprobs_f = torch.zeros(
            len(torch.unique(traj_id, sorted=True)),
            dtype=self.float,
            device=self.device,
        ).index_add_(0, traj_id, logprobs_f)
        # Backward trajectories
        policy_output_b = self.backward_policy(self.env.statetorch2policy(states))
        logprobs_b = self.env.get_logprobs(
            policy_output_b, False, actions, parents, masks_b, loginf
        )
        sumlogprobs_b = torch.zeros(
            len(torch.unique(traj_id, sorted=True)),
            dtype=self.float,
            device=self.device,
        ).index_add_(0, traj_id, logprobs_b)
        # Sort rewards of done states by ascending traj id
        rewards = rewards[done.eq(1)][torch.argsort(traj_id[done.eq(1)])]
        # Trajectory balance loss
        loss = (
            (self.logZ.sum() + sumlogprobs_f - sumlogprobs_b - torch.log(rewards))
            .pow(2)
            .mean()
        )
        return (loss, loss, loss), rewards

    def unpack_terminal_states(self, batch):
        """
        Unpacks the terminating states and trajectories of a batch and converts them
        to Python lists/tuples.
        """
        # TODO: make sure that unpacked states and trajs are sorted by traj_id (like
        # rewards will be)
        trajs = [[] for _ in range(self.batch_size)]
        states = [None] * self.batch_size
        for el in batch:
            traj_id = el[5][:1].item()
            state_id = el[6][:1].item()
            trajs[traj_id].append(tuple(el[1][0].tolist()))
            if bool(el[4].item()):
                states[traj_id] = tuple(el[0][0].tolist())
        trajs = [tuple(el) for el in trajs]
        return states, trajs

    def train(self):
        # Metrics
        all_losses = []
        all_visited = []
        loss_term_ema = None
        loss_flow_ema = None
        # Generate list of environments
        envs = [self.env.copy().reset() for _ in range(self.batch_size)]
        # Train loop
        pbar = tqdm(range(1, self.n_train_steps + 1), disable=not self.logger.progress)
        for it in pbar:
            # Test
            if self.logger.do_test(it):
                self.l1, self.kl, self.jsd, figs = self.test()
                self.logger.log_test_metrics(
                    self.l1, self.kl, self.jsd, it, self.use_context
                )
                self.logger.log_plots(figs, it, self.use_context)
            t0_iter = time.time()
            data = []
            for j in range(self.sttr):
                batch, times = self.sample_batch(envs)
                data += batch
            for j in range(self.ttsr):
                if self.loss == "flowmatch":
                    losses, rewards = self.flowmatch_loss(
                        it * self.ttsr + j, data
                    )  # returns (opt loss, *metrics)
                elif self.loss == "trajectorybalance":
                    losses, rewards = self.trajectorybalance_loss(
                        it * self.ttsr + j, data
                    )  # returns (opt loss, *metrics)
                else:
                    print("Unknown loss!")
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
            states_term, trajs_term = self.unpack_terminal_states(batch)
            proxy_vals = self.env.reward2proxy(rewards).tolist()
            rewards = rewards.tolist()
            self.buffer.add(states_term, trajs_term, rewards, proxy_vals, it)
            self.buffer.add(
                states_term, trajs_term, rewards, proxy_vals, it, buffer="replay"
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
                batch_size=len(data),
                logz=self.logZ,
                learning_rates=self.lr_scheduler.get_last_lr(),
                step=it,
                use_context=self.use_context,
            )
            t1_log = time.time()
            times.update({"log": t1_log - t0_log})
            # Save intermediate models
            t0_model = time.time()
            self.logger.save_models(self.forward_policy, self.backward_policy, step=it)
            t1_model = time.time()
            times.update({"save_interim_model": t1_model - t0_model})

            # Moving average of the loss for early stopping
            if loss_term_ema and loss_flow_ema:
                loss_term_ema = (
                    self.ema_alpha * losses[1] + (1.0 - self.ema_alpha) * loss_term_ema
                )
                loss_flow_ema = (
                    self.ema_alpha * losses[2] + (1.0 - self.ema_alpha) * loss_flow_ema
                )
                if (
                    loss_term_ema < self.early_stopping
                    and loss_flow_ema < self.early_stopping
                ):
                    break
            else:
                loss_term_ema = losses[1]
                loss_flow_ema = losses[2]
            # Log times
            t1_iter = time.time()
            times.update({"iter": t1_iter - t0_iter})
            self.logger.log_time(times, use_context=self.use_context)

        # Save final model
        self.logger.save_models(self.forward_policy, self.backward_policy, final=True)
        # Close logger
        if self.use_context == False:
            self.logger.end()

    def test(self, **plot_kwargs):
        """
        Computes metrics by sampling trajectories from the forward policy.
        """
        if self.buffer.test_pkl is None:
            return self.l1, self.kl, self.jsd, (None,)
        with open(self.buffer.test_pkl, "rb") as f:
            dict_tt = pickle.load(f)
            x_tt = dict_tt["x"]
        x_sampled, _ = self.sample_batch(self.env, self.logger.test.n, train=False)
        if self.buffer.test_type is not None and self.buffer.test_type == "all":
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
        elif self.continuous:
            x_sampled = torch2np(self.env.statebatch2proxy(x_sampled))
            x_tt = torch2np(self.env.statebatch2proxy(x_tt))
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
                x_from_reward = torch2np(self.env.statetorch2proxy(x_from_reward))
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
            raise NotImplementedError
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
        return l1, kl, jsd, [fig_reward_samples, fig_kde_pred, fig_kde_true]

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
            self.env, self.oracle_n, train=False
        )

        if not self.logger.lightweight:
            self.logger.log_metric(
                "unique_states",
                np.unique(all_visited).shape[0],
                step=it,
                use_context=self.use_context,
            )


class Policy:
    def __init__(self, config, env, device, float_precision, base=None):
        # If config is null, default to uniform
        if config is None:
            config = OmegaConf.create()
            config.type = "uniform"
        # Device and float precision
        self.device = device
        self.float = float_precision
        # Input and output dimensions
        self.state_dim = env.policy_input_dim
        self.fixed_output = torch.tensor(env.fixed_policy_output).to(
            dtype=self.float, device=self.device
        )
        self.random_output = torch.tensor(env.random_policy_output).to(
            dtype=self.float, device=self.device
        )
        self.output_dim = len(self.fixed_output)
        if "shared_weights" in config:
            self.shared_weights = config.shared_weights
        else:
            self.shared_weights = False
        self.base = base
        if "n_hid" in config:
            self.n_hid = config.n_hid
        else:
            self.n_hid = None
        if "n_layers" in config:
            self.n_layers = config.n_layers
        else:
            self.n_layers = None
        if "tail" in config:
            self.tail = config.tail
        else:
            self.tail = []
        if "type" in config:
            self.type = config.type
        elif self.shared_weights:
            self.type = self.base.type
        else:
            raise "Policy type must be defined if shared_weights is False"
        # Instantiate policy
        if self.type == "fixed":
            self.model = self.fixed_distribution
            self.is_model = False
        elif self.type == "uniform":
            self.model = self.uniform_distribution
            self.is_model = False
        elif self.type == "mlp":
            self.model = self.make_mlp(nn.LeakyReLU())
            self.is_model = True
        else:
            raise "Policy model type not defined"
        if self.is_model:
            self.model.to(self.device)

    def __call__(self, states):
        return self.model(states)

    def make_mlp(self, activation):
        """
        Defines an MLP with no top layer activation
        If share_weight == True,
            baseModel (the model with which weights are to be shared) must be provided
        Args
        ----
        layers_dim : list
            Dimensionality of each layer
        activation : Activation
            Activation function
        """
        if self.shared_weights == True and self.base is not None:
            mlp = nn.Sequential(
                self.base.model[:-1],
                nn.Linear(
                    self.base.model[-1].in_features, self.base.model[-1].out_features
                ),
            )
            return mlp
        elif self.shared_weights == False:
            layers_dim = (
                [self.state_dim] + [self.n_hid] * self.n_layers + [(self.output_dim)]
            )
            mlp = nn.Sequential(
                *(
                    sum(
                        [
                            [nn.Linear(idim, odim)]
                            + ([activation] if n < len(layers_dim) - 2 else [])
                            for n, (idim, odim) in enumerate(
                                zip(layers_dim, layers_dim[1:])
                            )
                        ],
                        [],
                    )
                    + self.tail
                )
            )
            return mlp
        else:
            raise ValueError(
                "Base Model must be provided when shared_weights is set to True"
            )

    def fixed_distribution(self, states):
        """
        Returns the fixed distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.fixed_output, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )

    def random_distribution(self, states):
        """
        Returns the random distribution specified by the environment.
        Args: states: tensor
        """
        return torch.tile(self.random_output, (len(states), 1)).to(
            dtype=self.float, device=self.device
        )

    def uniform_distribution(self, states):
        """
        Return action logits (log probabilities) from a uniform distribution
        Args: states: tensor
        """
        return torch.ones(
            (len(states), self.output_dim), dtype=self.float, device=self.device
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


def logq(traj_list, actions_list, model, env, loginf=1000):
    # TODO: this method is probably suboptimal, since it may repeat forward calls for
    # the same nodes.
    log_q = torch.tensor(1.0)
    loginf = self._tfloat([loginf])
    for traj, actions in zip(traj_list, actions_list):
        traj = traj[::-1]
        actions = actions[::-1]
        masks = self._tbool(
            [env.get_mask_invalid_actions_forward(state, 0) for state in traj]
        )
        with torch.no_grad():
            logits_traj = model(self._tfloat(env.statebatch2policy(traj)))
        logits_traj[masks] = -loginf
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
