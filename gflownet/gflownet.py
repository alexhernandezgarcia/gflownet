"""
GFlowNet
TODO:
    - Seeds
"""
import copy
import pickle
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from scipy.special import logsumexp
from torch.distributions import Bernoulli
from tqdm import tqdm

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.batch import Batch
from gflownet.utils.buffer import Buffer
from gflownet.utils.common import (
    set_device,
    set_float_precision,
    tbool,
    tfloat,
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

    def parameters(self):
        if self.backward_policy.is_model is False:
            return list(self.forward_policy.model.parameters())
        elif self.loss == "trajectorybalance":
            return list(self.forward_policy.model.parameters()) + list(
                self.backward_policy.model.parameters()
            )
        else:
            raise ValueError("Backward Policy cannot be a nn in flowmatch.")

    def sample_actions(
        self,
        envs: List[GFlowNetEnv],
        batch: Optional[Batch] = None,
        sampling_method: Optional[str] = "policy",
        backward: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        random_action_prob: Optional[float] = 0.0,
        no_random: Optional[bool] = True,
        times: Optional[dict] = None,
    ) -> List[Tuple]:
        """
        Samples one action on each environment of the list envs, according to the
        sampling method specify by sampling_method.

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
            A batch from which obtain required variables (e.g. masks) to avoid
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
            Probability of sampling random actions. If None, self.random_action_prob is used.

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
            if temperature is not None:
                temperature = self.temperature_logits
            if random_action_prob is not None:
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
        # it is not available and stores it to the batch)
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
                policy_outputs[idx_norandom, :] = model(
                    tfloat(
                        self.env.statebatch2policy(
                            [s for s, do in zip(states, idx_norandom) if do]
                        ),
                        device=self.device,
                        float_type=self.float,
                    )
                )
        else:
            raise NotImplementedError

        # Sample actions from policy outputs
        # TODO: consider adding logprobs to batch
        actions, logprobs = self.env.sample_actions(
            policy_outputs,
            sampling_method,
            mask_invalid_actions,
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

    # @profile
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
                dict_tr = pickle.load(f)
                # TODO: implement other sampling options besides permutation
                # TODO: this converts to numpy
                x_tr = self.rng.permutation(dict_tr["x"])
            actions = []
            valids = []
            for idx, env in enumerate(envs):
                env.set_state(x_tr[idx], done=True)
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
                # TODO: implement other sampling options besides permutation
                if n_replay > 0:
                    x_replay = [x for x in dict_replay["x"].values()]
                    x_replay = [x_replay[idx] for idx in self.rng.permutation(n_replay)]
            actions = []
            valids = []
            for idx, env in enumerate(envs):
                env.set_state(x_replay[idx], done=True)
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
            assert all(valids)
            # Filter out finished trajectories
            envs = [env for env in envs if not env.equal(env.state, env.source)]
        times["replay_actions"] = time.time() - t0_replay

        # Merge forward and backward batches
        batch = batch.merge([batch_forward, batch_train, batch_replay])

        times["all"] = time.time() - t0_all

        return batch, times

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
        assert batch.is_valid()
        # Get necessary tensors from batch
        states = batch.get_states(policy=True)
        actions = batch.get_actions()
        parents = batch.get_parents(policy=True)
        traj_indices = batch.get_trajectory_indices()
        masks_f = batch.get_masks_forward(of_parents=True)
        masks_b = batch.get_masks_backward()
        rewards = batch.get_terminating_rewards(sort_by="trajectory")

        # Forward trajectories
        policy_output_f = self.forward_policy(parents)
        logprobs_f = self.env.get_logprobs(
            policy_output_f, True, actions, states, masks_f
        )
        sumlogprobs_f = torch.zeros(
            batch.get_n_trajectories(),
            dtype=self.float,
            device=self.device,
        ).index_add_(0, traj_indices, logprobs_f)
        # Backward trajectories
        policy_output_b = self.backward_policy(states)
        logprobs_b = self.env.get_logprobs(
            policy_output_b, False, actions, parents, masks_b
        )
        sumlogprobs_b = torch.zeros(
            batch.get_n_trajectories(),
            dtype=self.float,
            device=self.device,
        ).index_add_(0, traj_indices, logprobs_b)
        # Trajectory balance loss
        loss = (
            (self.logZ.sum() + sumlogprobs_f - sumlogprobs_b - torch.log(rewards))
            .pow(2)
            .mean()
        )
        return loss, loss, loss

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
            if self.logger.do_test(it):
                self.l1, self.kl, self.jsd, figs = self.test()
                self.logger.log_test_metrics(
                    self.l1, self.kl, self.jsd, it, self.use_context
                )
                self.logger.log_plots(figs, it, self.use_context)
            t0_iter = time.time()
            batch = Batch(env=self.env, device=self.device, float_type=self.float)
            for j in range(self.sttr):
                sub_batch, times = self.sample_batch(
                    n_forward=self.batch_size.forward,
                    n_train=self.batch_size.train,
                    n_replay=self.batch_size.replay,
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
            self.logger.save_models(self.forward_policy, self.backward_policy, step=it)
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
        self.logger.save_models(self.forward_policy, self.backward_policy, final=True)
        # Close logger
        if self.use_context is False:
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
        batch, _ = self.sample_batch(n_forward=self.logger.test.n, train=False)
        assert batch.is_valid()
        x_sampled = batch.get_terminating_states()
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
            # TODO make it work with conditional env
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
            n_forward=self.oracle_n, train=False
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
        if self.shared_weights is True and self.base is not None:
            mlp = nn.Sequential(
                self.base.model[:-1],
                nn.Linear(
                    self.base.model[-1].in_features, self.base.model[-1].out_features
                ),
            )
            return mlp
        elif self.shared_weights is False:
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
                    env.statebatch2policy(traj),
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
