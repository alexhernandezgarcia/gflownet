"""
GFlowNet
TODO:
    - Seeds
"""

import copy
import pickle
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from torchtyping import TensorType
from tqdm import tqdm, trange

from gflownet.envs.base import GFlowNetEnv
from gflownet.evaluator.base import BaseEvaluator
from gflownet.proxy.base import Proxy
from gflownet.utils.batch import Batch
from gflownet.utils.buffer import Buffer
from gflownet.utils.common import (
    bootstrap_samples,
    set_device,
    set_float_precision,
    tbool,
    tfloat,
    tlong,
)


class GFlowNetAgent:
    def __init__(
        self,
        env_maker: partial,
        proxy,
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
        evaluator,
        state_flow=None,
        use_context=False,
        replay_sampling="permutation",
        train_sampling="permutation",
        **kwargs,
    ):
        """
        Main class of this repository. Handles the training logic for a GFlowNet model.

        Parameters
        ----------
        env : GFlowNetEnv
            The environment to be used for training, i.e. the DAG, action space and
            reward function.
        seed : int
            Random seed to be used for reproducibility.
        device : str
            Device to be used for training and inference, e.g. "cuda" or "cpu".
        float_precision : int
            Precision of the floating point numbers, e.g. 32 or 64.
        optimizer : dict
            Optimizer config dictionary. See gflownet.yaml:optimizer for details.
        buffer : dict
            Buffer config dictionary. See gflownet.yaml:buffer for details.
        forward_policy : gflownet.policy.base.Policy
            The forward policy to be used for training. Parameterized from
            `gflownet.yaml:forward_policy` and parsed with
            `gflownet/utils/policy.py:set_policy`.
        backward_policy : gflownet.policy.base.Policy
            Same as forward_policy, but for the backward policy.
        mask_invalid_actions : bool
            Whether to mask invalid actions in the policy outputs.
        temperature_logits : float
            Temperature to adjust the logits by logits /= temperature. If None,
            self.temperature_logits is used.
        random_action_prob : float
            Probability of sampling random actions. If None (default),
            self.random_action_prob is used, unless its value is forced to either 0.0 or
            1.0 by other arguments (sampling_method or no_random).
        pct_offline : float
            Percentage of offline data to be used for training.
        logger : gflownet.utils.logger.Logger
            Logger object to be used for logging and saving checkpoints
            (`gflownet/utils/logger.py:Logger`).
        num_empirical_loss : int
            Number of empirical loss samples to be used for training.
        evaluator : gflownet.evaluator.base.BaseEvaluator
            :py:mod:`~gflownet.evaluator` ``Evaluator`` instance.
        state_flow : dict, optional
            State flow config dictionary. See `gflownet.yaml:state_flow` for details. By
            default None.
        use_context : bool, optional
            Whether the logger will use its context in metrics names. Formerly the
            `active_learning: bool` flag. By default False.
        replay_sampling : str, optional
            Type of sampling for the replay buffer. See
            :meth:`~gflownet.utils.buffer.select`. By default "permutation".
        train_sampling : str, optional
            Type of sampling for the train buffer (offline backward trajectories). See
            :meth:`~gflownet.utils.buffer.select`. By default "permutation".

        Raises
        ------
        Exception
            If the loss is flowmatch/flowmatching and the environment is continuous.
        """
        # Seed
        self.rng = np.random.default_rng(seed)
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Environment
        self.env_maker = env_maker
        self.env = self.env_maker()
        # Proxy
        self.proxy = proxy
        self.proxy.setup(self.env)
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
        # Buffers
        self.replay_sampling = replay_sampling
        self.train_sampling = train_sampling
        self.buffer = Buffer(
            **buffer,
            env=self.env,
            proxy=self.proxy,
            logger=logger,
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
            print("\nTrain data")
            print(f"\tMean score: {energies_stats_tr[2]}")
            print(f"\tStd score: {energies_stats_tr[3]}")
            print(f"\tMin score: {energies_stats_tr[0]}")
            print(f"\tMax score: {energies_stats_tr[1]}")
        else:
            energies_stats_tr = None
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

        # Evaluator
        self.evaluator = evaluator
        self.evaluator.set_agent(self)

        self.n_train_steps = optimizer.n_train_steps
        self.batch_size = optimizer.batch_size
        self.batch_size_total = sum(self.batch_size.values())
        self.ttsr = max(int(optimizer.train_to_sample_ratio), 1)
        self.sttr = max(int(1 / optimizer.train_to_sample_ratio), 1)
        self.clip_grad_norm = optimizer.clip_grad_norm
        self.tau = optimizer.bootstrap_tau
        self.ema_alpha = optimizer.ema_alpha
        self.early_stopping = optimizer.early_stopping
        self.use_context = use_context
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
        env_cond: Optional[GFlowNetEnv] = None,
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

        env_cond : GFlowNetEnv or derived
            An environment to do conditional sampling, that is restrict the action
            space via the masks of the main environments.

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

        # Obtain masks of invalid actions
        mask_invalid_actions = self._get_masks(envs, batch, env_cond, backward)

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

    def _get_masks(
        self,
        envs: List[GFlowNetEnv],
        batch: Optional[Batch] = None,
        env_cond: Optional[GFlowNetEnv] = None,
        backward: Optional[bool] = False,
    ) -> List[List[bool]]:
        """
        Given a batch and/or a list of environments, obtains the mask of invalid
        actions of each environment's current state.

        Note that batch.get_item("mask_*") computes the mask if it is not available and
        stores it in the batch.

        If env_cond is not None, then the masks will be adjusted according to the
        restrictions imposed by the conditioning environment, env_cond (see
        GFlowNetEnv.mask_conditioning()).

        Args
        ----
        envs : list of GFlowNetEnv or derived
            A list of instances of the environment

        batch_forward : Batch
            A batch from which to obtain the masks to avoid recomputing them.

        env_cond : GFlowNetEnv or derived
            An environment to do conditional sampling, that is restrict the action
            space via the masks of the main environments. Ignored if None.

        backward : bool
            True if sampling is backward. False (forward) by default.

        Returns
        -------
        A list of boolean lists containing the masks of invalid actions of each
        environment.
        """
        if not self.mask_invalid_actions:
            return None
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
        # Mask conditioning
        if env_cond is not None:
            mask_invalid_actions = tbool(
                [
                    env.mask_conditioning(mask, env_cond, backward)
                    for env, mask in zip(envs, mask_invalid_actions)
                ],
                device=self.device,
            )
        return mask_invalid_actions

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
        env_cond: Optional[GFlowNetEnv] = None,
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
        envs = [self.env_maker().set_id(idx) for idx in range(n_forward)]
        batch_forward = Batch(
            env=self.env, proxy=self.proxy, device=self.device, float_type=self.float
        )
        while envs:
            # Sample actions
            t0_a_envs = time.time()
            actions = self.sample_actions(
                envs,
                batch_forward,
                env_cond,
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
        envs = [self.env_maker().set_id(idx) for idx in range(n_train)]
        batch_train = Batch(
            env=self.env, proxy=self.proxy, device=self.device, float_type=self.float
        )
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
                env_cond,
                backward=True,
                no_random=not train,
                times=times,
            )
            times["actions_envs"] += time.time() - t0_a_envs
            # Update environments with sampled actions
            envs, actions, valids = self.step(envs, actions, backward=True)
            # Add to batch
            batch_train.add_to_batch(envs, actions, valids, backward=True, train=train)
            # Filter out finished trajectories
            envs = [env for env in envs if not env.equal(env.state, env.source)]
        times["train_actions"] = time.time() - t0_train

        # REPLAY BACKWARD trajectories
        t0_replay = time.time()
        batch_replay = Batch(
            env=self.env, proxy=self.proxy, device=self.device, float_type=self.float
        )
        if n_replay > 0 and self.buffer.replay_pkl is not None:
            with open(self.buffer.replay_pkl, "rb") as f:
                dict_replay = pickle.load(f)
                n_replay = min(n_replay, len(dict_replay["x"]))
                envs = [self.env_maker().set_id(idx) for idx in range(n_replay)]
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
                env_cond,
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
        logrewards = batch.get_rewards(log=True)
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
        loss_term = (inflow[done] - logrewards[done]).pow(2).mean()
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
        logrewards = batch.get_terminating_rewards(log=True, sort_by="trajectory")

        # Trajectory balance loss
        loss = (self.logZ.sum() + logprobs_f - logprobs_b - logrewards).pow(2).mean()
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
        logrewards = batch.get_terminating_rewards(log=True, sort_by="insertion")

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
        logflows_states[done.eq(1)] = logrewards
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
        logrewards_states = batch.get_rewards(log=True, do_non_terminating=True)
        logrewards_parents = batch.get_rewards_parents(log=True)
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
        energies_transitions = logrewards_parents - logrewards_states

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
        pbar = tqdm(
            total=n_states,
            disable=not self.logger.progress,
            leave=False,
            desc="Sampling backward actions from test data to estimate logprobs",
        )
        pbar2 = trange(
            end_batch * n_trajectories,
            disable=not self.logger.progress,
            leave=False,
            desc="Setting env terminal states",
        )
        while init_batch < n_states:
            batch = Batch(
                env=self.env,
                proxy=self.proxy,
                device=self.device,
                float_type=self.float,
            )
            # Create an environment for each data point and trajectory and set the state
            envs = []
            pbar2.reset((end_batch - init_batch) * n_trajectories)
            for state_idx in range(init_batch, end_batch):
                for traj_idx in range(n_trajectories):
                    idx = int(mult_indices * state_idx + traj_idx)
                    env = self.env_maker().set_id(idx)
                    env.set_state(states_term[state_idx], done=True)
                    envs.append(env)
                    pbar2.update(1)
            # Sample trajectories
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
            if n_states > batch_size:
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
        pbar.close()
        pbar2.close()
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
            # Test and log
            if self.evaluator.should_eval(it):
                self.evaluator.eval_and_log(it)
            if self.evaluator.should_eval_top_k(it):
                self.evaluator.eval_and_log_top_k(it)

            t0_iter = time.time()
            batch = Batch(
                env=self.env,
                proxy=self.proxy,
                device=self.device,
                float_type=self.float,
            )
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
            proxy_vals = batch.get_terminating_proxy_values(sort_by="trajectory")
            proxy_vals = proxy_vals.tolist()
            # The batch will typically have the log-rewards available, since they are
            # used to compute the losses. In order to avoid recalculating the proxy
            # values, the natural rewards are computed by taking the exponential of the
            # log-rewards. In case the rewards are available in the batch but not the
            # log-rewards, the latter are computed by taking the log of the rewards.
            # Numerical issues are not critical in this case, since the derived values
            # are only used for reporting purposes.
            if batch.rewards_available(log=False):
                rewards = batch.get_terminating_rewards(sort_by="trajectory")
            if batch.rewards_available(log=True):
                logrewards = batch.get_terminating_rewards(
                    sort_by="trajectory", log=True
                )
            if not batch.rewards_available(log=False):
                assert batch.rewards_available(log=True)
                rewards = torch.exp(logrewards)
            if not batch.rewards_available(log=True):
                assert batch.rewards_available(log=False)
                logrewards = torch.log(rewards)
            rewards = rewards.tolist()
            logrewards = logrewards.tolist()
            actions_trajectories = batch.get_actions_trajectories()
            self.buffer.add(states_term, actions_trajectories, logrewards, it)
            self.buffer.add(
                states_term, actions_trajectories, logrewards, it, buffer="replay"
            )
            t1_buffer = time.time()
            times.update({"buffer": t1_buffer - t0_buffer})
            # Log
            if self.logger.lightweight:
                all_losses = all_losses[-100:]
            else:
                all_visited.extend(states_term)
            # Progress bar
            self.logger.progressbar_update(
                pbar, all_losses, rewards, self.jsd, it, self.use_context
            )
            # Train logs
            t0_log = time.time()
            if self.evaluator.should_log_train(it):
                self.logger.log_train(
                    losses=losses,
                    rewards=rewards,
                    logrewards=logrewards,
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
            if self.evaluator.should_checkpoint(it):
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

    def get_sample_space_and_reward(self):
        """
        Returns samples representative of the env state space with their rewards

        Returns
        -------
        sample_space_batch : tensor
            Repressentative terminating states for the environment
         rewards_sample_space : tensor
            Rewards associated with the tates in sample_space_batch
        """
        if not hasattr(self, "sample_space_batch"):
            if hasattr(self.env, "get_all_terminating_states"):
                self.sample_space_batch = self.env.get_all_terminating_states()
            elif hasattr(self.env, "get_grid_terminating_states"):
                self.sample_space_batch = self.env.get_grid_terminating_states(
                    self.evaluator.config.n_grid
                )
            else:
                raise NotImplementedError(
                    "In order to obtain representative terminating states, the "
                    "environment must implement either get_all_terminating_states() "
                    "or get_grid_terminating_states()"
                )
            self.sample_space_batch = self.env.states2proxy(self.sample_space_batch)
        if not hasattr(self, "rewards_sample_space"):
            self.rewards_sample_space = self.proxy.rewards(self.sample_space_batch)

        return self.sample_space_batch, self.rewards_sample_space

    # TODO: implement other proposal distributions
    # TODO: rethink whether it is needed to convert to reward
    def sample_from_reward(
        self,
        n_samples: int,
        proposal_distribution: str = "uniform",
        epsilon=1e-4,
    ) -> Union[List, Dict, TensorType["n_samples", "state_dim"]]:
        """
        Rejection sampling with proposal the uniform distribution defined over the
        sample space.

        Returns a tensor in GFloNet (state) format.

        Parameters
        ----------
        n_samples : int
            The number of samples to draw from the reward distribution.
        proposal_distribution : str
            Identifier of the proposal distribution. Currently only `uniform` is
            implemented.
        epsilon : float
            Small epsilon parameter for rejection sampling.

        Returns
        -------
        samples_final : list
            The list of samples drawn from the reward distribution in environment
            format.
        """
        samples_final = []
        max_reward = self.proxy.get_max_reward()
        while len(samples_final) < n_samples:
            if proposal_distribution == "uniform":
                # TODO: sample only the remaining number of samples
                samples_uniform = self.env.get_uniform_terminating_states(n_samples)
            else:
                raise NotImplementedError("The proposal distribution must be uniform")
            rewards = self.proxy.proxy2reward(
                self.proxy(self.env.states2proxy(samples_uniform))
            )
            indices_accept = (
                (
                    torch.rand(n_samples, dtype=self.float, device=self.device)
                    * (max_reward + epsilon)
                    < rewards
                )
                .flatten()
                .tolist()
            )
            samples_accepted = [samples_uniform[idx] for idx in indices_accept]
            samples_final.extend(samples_accepted[-(n_samples - len(samples_final)) :])
        return samples_final


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
