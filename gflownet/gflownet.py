"""
GFlowNet
TODO:
    - Seeds
"""

import copy
import gc
import pickle
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType
from tqdm import tqdm, trange

from gflownet.envs.base import GFlowNetEnv
from gflownet.evaluator.base import BaseEvaluator
from gflownet.utils.batch import Batch, compute_logprobs_trajectories
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
        loss,
        optimizer,
        buffer,
        forward_policy,
        backward_policy,
        mask_invalid_actions,
        temperature_logits,
        random_action_prob,
        logger,
        evaluator,
        state_flow=None,
        use_context=False,
        replay_sampling="permutation",
        train_sampling="permutation",
        garbage_collection_period: int = 0,
        collect_reversed_logprobs: bool = False,
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
        loss : Loss
            An instance of a loss class, corresponding to one of the GFlowNet
            objectives, for example Flow Matching or Trajectory Balance.
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
        logger : gflownet.utils.logger.Logger
            Logger object to be used for logging and saving checkpoints
            (`gflownet/utils/logger.py:Logger`).
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
        garbage_collection_period : int
            The periodicity to perform garbage collection and empty the cache of the
            GPU. By default it is 0, so no garbage collection is performed. This is
            because it can incur a large time overhead unnecessarily.
        collect_reversed_logprobs: bool
            If True, reversed logprobs will be computed and collected during sampling batches
            for training

        Raises
        ------
        Exception
            If the environment is continuous and the loss is not well defined for
            continuous GFlowNets.
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
        self.env_cache = []
        # Proxy
        self.proxy = proxy
        self.proxy.setup(self.env)
        # Loss
        self.loss = loss
        if self.loss.requires_log_z:
            self.logZ = nn.Parameter(torch.ones(optimizer.z_dim) * 150.0 / 64)
            self.loss.set_log_z(self.logZ)
        else:
            self.logZ = None
        self.collect_backwards_masks = self.loss.requires_backward_policy()
        self.collect_reversed_logprobs = collect_reversed_logprobs
        # Continuous environments
        self.continuous = hasattr(self.env, "continuous") and self.env.continuous
        if self.continuous and not loss.is_defined_for_continuous():
            raise Exception(
                f"The environment is continuous but the {loss.name} loss is not well "
                "defined for continuous environments. Consider using a different loss."
            )
        # Logging
        self.logger = logger
        # Buffers
        self.replay_sampling = replay_sampling
        self.train_sampling = train_sampling
        self.buffer = buffer
        # Train set statistics and reward normalization constant
        if self.buffer.train is not None:
            scores_stats_tr = [
                self.buffer.min_tr,
                self.buffer.max_tr,
                self.buffer.mean_tr,
                self.buffer.std_tr,
                self.buffer.max_norm_tr,
            ]
            print("\nTrain data")
            print(f"\tMean score: {scores_stats_tr[2]}")
            print(f"\tStd score: {scores_stats_tr[3]}")
            print(f"\tMin score: {scores_stats_tr[0]}")
            print(f"\tMax score: {scores_stats_tr[1]}")
        else:
            scores_stats_tr = None
        # Test set statistics
        if self.buffer.test is not None:
            print("\nTest data")
            print(f"\tMean score: {self.buffer.test['scores'].mean()}")
            print(f"\tStd score: {self.buffer.test['scores'].std()}")
            print(f"\tMin score: {self.buffer.test['scores'].min()}")
            print(f"\tMax score: {self.buffer.test['scores'].max()}")

        # Models
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.state_flow = state_flow

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
        self.use_context = use_context
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # Training
        self.it = 1
        self.mask_invalid_actions = mask_invalid_actions
        self.temperature_logits = temperature_logits
        self.random_action_prob = random_action_prob
        self.garbage_collection_period = garbage_collection_period
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
            if not self.loss.requires_backward_policy():
                raise ValueError(
                    "Backward policy initialized but not required by "
                    f"loss {self.loss.name}."
                )
            parameters += list(self.backward_policy.model.parameters())
        if self.state_flow is not None:
            if not self.loss.requires_state_flow_model():
                raise ValueError(
                    "State flow model initialized but not required by "
                    f"loss {self.loss.name}."
                )
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
        compute_reversed_logprobs: Optional[bool] = False,
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

        compute_reversed_logprobs: bool
            If True, reversed logprobs will be computed. Default is False

        Returns
        -------
        actions : list of tuples
            The sampled actions, one for each environment in envs.
        logprobs : tensor or None
            Log probabilities corresponding to each sampled action. It may be None if
            the environment's sampled_action_batch() method does not calculate the log
            probs while sampling the actions.
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
            model = self.backward_policy
            model_rev = self.forward_policy
        else:
            model = self.forward_policy
            model_rev = self.backward_policy

        if not isinstance(envs, list):
            envs = [envs]
        # Build states and masks
        states = [env.state for env in envs]

        # Obtain masks of invalid actions
        mask_invalid_actions = self._get_masks(
            envs, batch, env_cond, backward, backward
        )

        # Get policy inputs from the states and obtain the policy outputs from the
        # model
        # TODO: get policy states from batch
        states_policy = tfloat(
            self.env.states2policy(states),
            device=self.device,
            float_type=self.float,
        )
        policy_outputs = model(states_policy)

        # Sample actions from policy outputs
        actions = self.env.sample_actions_batch(
            policy_outputs=policy_outputs,
            mask=mask_invalid_actions,
            states_from=states,
            is_backward=backward,
            sampling_method=sampling_method,
            random_action_prob=random_action_prob,
            temperature_logits=temperature,
        )
        # Compute logprobs from policy outputs
        actions_tensor = tfloat(actions, device=self.device, float_type=self.float)
        logprobs = self.env.get_logprobs(
            policy_outputs=policy_outputs,
            actions=actions_tensor,
            mask=mask_invalid_actions,
            states_from=states,
            is_backward=backward,
        )

        if compute_reversed_logprobs:
            logprobs_rev = torch.zeros_like(logprobs)
            actions_rev, actions_rev_valid = batch.get_latest_added_actions(
                envs, backward
            )
            if any(actions_rev_valid):
                if not all(actions_rev_valid):
                    actions_rev = [
                        act for act, val in zip(actions_rev, actions_rev_valid) if val
                    ]
                actions_rev = tfloat(
                    actions_rev, device=self.device, float_type=self.float
                )
                actions_rev_valid = tbool(actions_rev_valid, device=self.device)
                mask_invalid_actions_rev = self._get_masks(
                    envs, batch, env_cond, not backward, backward
                )
                policy_outputs_rev = model_rev(states_policy[actions_rev_valid])
                states_from = [st for st, val in zip(states, actions_rev_valid) if val]
                logrpobs_rev_val = self.env.get_logprobs(
                    policy_outputs=policy_outputs_rev[actions_rev_valid],
                    actions=actions_rev,
                    mask=mask_invalid_actions_rev[actions_rev_valid],
                    states_from=states_from,
                    is_backward=not backward,
                )
                logprobs_rev[actions_rev_valid] = logrpobs_rev_val
        else:
            logprobs_rev = [None] * len(actions)
        return actions, logprobs, logprobs_rev

    def _get_masks(
        self,
        envs: List[GFlowNetEnv],
        batch: Optional[Batch] = None,
        env_cond: Optional[GFlowNetEnv] = None,
        backward_mask: Optional[bool] = False,
        backward_traj: Optional[bool] = False,
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

        backward_mack : bool
            True if mask is for sampling backward. False (forward) by default.

        backward_traj : bool
            True if trajectories in the batch are sampled backward. False (forward) by default.

        Returns
        -------
        A list of boolean lists containing the masks of invalid actions of each
        environment.
        """
        if not self.mask_invalid_actions:
            return None
        if batch is not None:
            if backward_mask:
                mask_invalid_actions = tbool(
                    [
                        batch.get_item("mask_backward", env, backward=backward_traj)
                        for env in envs
                    ],
                    device=self.device,
                )
            else:
                mask_invalid_actions = tbool(
                    [
                        batch.get_item("mask_forward", env, backward=backward_traj)
                        for env in envs
                    ],
                    device=self.device,
                )
        # Compute masks since a batch was not provided
        else:
            if backward_mask:
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
                    env.mask_conditioning(mask, env_cond, backward_mask)
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
        pair, depending on the value of backward.

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

    def get_env_instances(self, nb_env_instances):
        """
        Returns the requested number of instances of the environment

        Args
        ----
        nb_env_instances : int
            Number of instance to return

        Returns
        -------
        A list of environment instances
        """
        # Create new env instances if not enough exist in the cache.
        if len(self.env_cache) < nb_env_instances:
            nb_new_instances_needed = nb_env_instances - len(self.env_cache)
            new_instances = [self.env_maker() for _ in range(nb_new_instances_needed)]
            self.env_cache.extend(new_instances)

        # Return the requested instances
        return self.env_cache[:nb_env_instances]

    # TODO: avoid computing gradients when not needed
    # TODO: extract code from while loop to avoid replication
    def sample_batch(
        self,
        n_forward: int = 0,
        n_train: int = 0,
        n_replay: int = 0,
        env_cond: Optional[GFlowNetEnv] = None,
        train=True,
        progress=False,
        collect_forwards_masks=False,
        collect_backwards_masks=False,
    ):
        """
        TODO: extend docstring.
        Builds a batch of data by sampling online and/or offline trajectories.
        """
        # Obtain the necessary env instances (one per forward/train/replay trajectory)
        # WARNING : These instances must be reset before use.
        nb_env_instances_needed = n_forward + n_train + n_replay
        env_instances = self.get_env_instances(nb_env_instances_needed)

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
        envs = [env_instances.pop().reset(idx) for idx in range(n_forward)]
        batch_forward = Batch(
            env=self.env,
            proxy=self.proxy,
            device=self.device,
            float_type=self.float,
            collect_forwards_masks=collect_forwards_masks,
            collect_backwards_masks=collect_backwards_masks,
        )
        while envs:
            # Sample actions
            t0_a_envs = time.time()
            actions, logprobs, logprobs_rev = self.sample_actions(
                envs,
                batch_forward,
                env_cond,
                no_random=not train,
                times=times,
                compute_reversed_logprobs=self.collect_reversed_logprobs,
            )
            times["actions_envs"] += time.time() - t0_a_envs
            # Update environments with sampled actions
            envs, actions, valids = self.step(envs, actions)
            # Add to batch
            actions_torch = torch.tensor(actions)
            batch_forward.add_to_batch(
                envs, actions, logprobs, logprobs_rev, valids, train=train
            )
            # Filter out finished trajectories
            envs = [env for env in envs if not env.done]
        times["forward_actions"] = time.time() - t0_forward

        # TRAIN BACKWARD trajectories
        t0_train = time.time()
        batch_train = Batch(
            env=self.env,
            proxy=self.proxy,
            device=self.device,
            float_type=self.float,
            collect_forwards_masks=collect_forwards_masks,
            collect_backwards_masks=collect_backwards_masks,
        )
        if n_train > 0 and self.buffer.train is not None:
            envs = [env_instances.pop().reset(idx) for idx in range(n_train)]
            x_train = self.buffer.select(
                self.buffer.train, n_train, self.train_sampling, self.rng
            )["samples"].values.tolist()
            for env, x in zip(envs, x_train):
                env.set_state(x, done=True)
        else:
            envs = []

        while envs:
            # Sample backward actions
            t0_a_envs = time.time()
            actions, logprobs, logprobs_rev = self.sample_actions(
                envs,
                batch_train,
                env_cond,
                backward=True,
                no_random=not train,
                times=times,
                compute_reversed_logprobs=self.collect_reversed_logprobs,
            )
            times["actions_envs"] += time.time() - t0_a_envs
            # Update environments with sampled actions
            envs, actions, valids = self.step(envs, actions, backward=True)
            # Add to batch
            batch_train.add_to_batch(
                envs,
                actions,
                logprobs,
                logprobs_rev,
                valids,
                backward=True,
                train=train,
            )
            # Filter out finished trajectories
            envs = [env for env in envs if not env.equal(env.state, env.source)]
        times["train_actions"] = time.time() - t0_train

        # REPLAY BACKWARD trajectories
        t0_replay = time.time()
        batch_replay = Batch(
            env=self.env,
            proxy=self.proxy,
            device=self.device,
            float_type=self.float,
            collect_forwards_masks=collect_forwards_masks,
            collect_backwards_masks=collect_backwards_masks,
        )
        if (
            n_replay > 0
            and self.buffer.replay is not None
            and len(self.buffer.replay) > 0
        ):
            envs = [env_instances.pop().reset(idx) for idx in range(n_replay)]
            n_replay = min(n_replay, len(self.buffer.replay))
            x_replay = self.buffer.select(
                self.buffer.replay,
                n_replay,
                self.replay_sampling,
                self.rng,
            )["samples"].values.tolist()
            for env, x in zip(envs, x_replay):
                env.set_state(x, done=True)
        else:
            envs = []

        while envs:
            # Sample backward actions
            t0_a_envs = time.time()
            actions, logprobs, logprobs_rev = self.sample_actions(
                envs,
                batch_replay,
                env_cond,
                backward=True,
                no_random=not train,
                times=times,
                compute_reversed_logprobs=self.collect_reversed_logprobs,
            )
            times["actions_envs"] += time.time() - t0_a_envs
            # Update environments with sampled actions
            envs, actions, valids = self.step(envs, actions, backward=True)
            # Add to batch
            batch_replay.add_to_batch(
                envs,
                actions,
                logprobs,
                logprobs_rev,
                valids,
                backward=True,
                train=train,
            )
            # Filter out finished trajectories
            envs = [env for env in envs if not env.equal(env.state, env.source)]
        times["replay_actions"] = time.time() - t0_replay

        # Merge forward and backward batches
        batch = batch.merge([batch_forward, batch_train, batch_replay])

        times["all"] = time.time() - t0_all

        return batch, times

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
            set where the terminating states are stored in key "samples".

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
                states_term = data_dict["samples"]
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
            disable=self.logger.progressbar["skip"],
            leave=False,
            desc="Sampling backward actions from test data to estimate logprobs",
        )
        pbar2 = trange(
            end_batch * n_trajectories,
            disable=self.logger.progressbar["skip"],
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
                actions, logprobs, logprobs_rev = self.sample_actions(
                    envs,
                    batch,
                    backward=True,
                    no_random=True,
                    times=times,
                    compute_reversed_logprobs=True,
                )
                # Update environments with sampled actions
                envs, actions, valids = self.step(envs, actions, backward=True)
                # Add to batch
                batch.add_to_batch(
                    envs,
                    actions,
                    logprobs,
                    logprobs_rev,
                    valids,
                    backward=True,
                    train=True,
                )
                # Filter out finished trajectories
                envs = [env for env in envs if not env.equal(env.state, env.source)]
            # Prepare data structures to compute log probabilities
            traj_indices_batch = tlong(
                batch.get_unique_trajectory_indices(), device=self.device
            )
            data_indices = traj_indices_batch // mult_indices
            traj_indices = traj_indices_batch % mult_indices
            # Compute log probabilities of the trajectories
            logprobs_f[data_indices, traj_indices] = compute_logprobs_trajectories(
                batch, self.env, forward_policy=self.forward_policy, backward=False
            )
            logprobs_b[data_indices, traj_indices] = compute_logprobs_trajectories(
                batch, self.env, backward_policy=self.backward_policy, backward=True
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
        # Train loop
        pbar = tqdm(
            initial=self.it - 1,
            total=self.n_train_steps,
            disable=self.logger.progressbar["skip"],
        )
        for self.it in range(self.it, self.n_train_steps + 1):
            # Test and log
            if self.evaluator.should_eval(self.it):
                self.evaluator.eval_and_log(self.it)
            if self.evaluator.should_eval_top_k(self.it):
                self.evaluator.eval_and_log_top_k(self.it)

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
                    collect_forwards_masks=True,
                    collect_backwards_masks=self.collect_backwards_masks,
                )
                batch.merge(sub_batch)
            for j in range(self.ttsr):
                losses = self.loss.compute(batch, get_sublosses=True)
                # TODO: deal with this in a better way
                if not all([torch.isfinite(loss) for loss in losses.values()]):
                    if self.logger.debug:
                        print("Loss is not finite - skipping iteration")
                else:
                    losses["all"].backward()
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.parameters(), self.clip_grad_norm
                        )
                    self.opt.step()
                    self.lr_scheduler.step()
                    self.opt.zero_grad()

            # Log training iteration: progress bar, buffer, metrics, intermediate
            # models
            times = self.log_train_iteration(pbar, losses, batch, times)

            # Log times
            t1_iter = time.time()
            times.update({"iter": t1_iter - t0_iter})
            self.logger.log_time(times, use_context=self.use_context)

            # Garbage collection and cleanup GPU memory
            if (
                self.garbage_collection_period > 0
                and self.garbage_collection_period % self.it == 0
            ):
                del batch
                gc.collect()
                torch.cuda.empty_cache()

            # Check early stopping
            if self.loss.do_early_stopping(losses["all"]):
                print(
                    "Ending training after meeting early stopping criteria: "
                    f"{self.loss.loss_ema} < {self.loss.early_stopping_th}"
                )
                break

        # Save final model
        self.logger.save_checkpoint(
            forward_policy=self.forward_policy,
            backward_policy=self.backward_policy,
            state_flow=self.state_flow,
            logZ=self.logZ,
            optimizer=self.opt,
            buffer=self.buffer,
            step=self.it,
            final=True,
        )
        # Close logger
        if self.use_context is False:
            self.logger.end()

    @torch.no_grad()
    def log_train_iteration(self, pbar: tqdm, losses: List, batch: Batch, times: dict):
        """
        Carries out the logging operations after the training iteration.

        The operations done by this method include:
            - Updating the main buffer
            - Updating the replay buffer
            - Logging the rewards and scores of the train batch
            - Logging the losses, logZ, learning rate and other metrics of the training
              process
            - Updating the progress bar
            - Save checkpoints

        Parameters
        ----------
        pbar : tqdm
            Progress bar object
        losses : dict
            Dictionary of losses after the training iteration
        batch : Batch
            Training batch
        times : dict
            Dictionary of times
        """
        t0_buffer = time.time()

        states_term = batch.get_terminating_states(sort_by="trajectory")
        proxy_vals = batch.get_terminating_proxy_values(sort_by="trajectory")
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
            logrewards = batch.get_terminating_rewards(sort_by="trajectory", log=True)
        if not batch.rewards_available(log=False):
            assert batch.rewards_available(log=True)
            rewards = torch.exp(logrewards)
        if not batch.rewards_available(log=True):
            assert batch.rewards_available(log=False)
            logrewards = torch.log(rewards)

        # Update main buffer
        actions_trajectories = batch.get_actions_trajectories()
        if self.buffer.use_main_buffer:
            self.buffer.add(
                states_term,
                actions_trajectories,
                rewards,
                self.it,
                buffer="main",
            )

        # Update replay buffer
        self.buffer.add(
            states_term,
            actions_trajectories,
            rewards,
            self.it,
            buffer="replay",
        )
        t1_buffer = time.time()
        times.update({"buffer": t1_buffer - t0_buffer})

        ### Train logs
        t0_log = time.time()

        # TODO: consider moving this into separate method
        if self.evaluator.should_log_train(self.it):

            # logZ
            if self.logZ is not None:
                logz = self.logZ.sum()
            else:
                logz = None

            # Trajectory length
            _, trajectory_lengths = torch.unique(
                batch.get_trajectory_indices(), return_counts=True
            )
            traj_length_mean = torch.mean(trajectory_lengths.to(self.float))
            traj_length_min = torch.min(trajectory_lengths)
            traj_length_max = torch.max(trajectory_lengths)

            # Learning rates
            learning_rates = self.lr_scheduler.get_last_lr()
            if len(learning_rates) == 1:
                learning_rates += [None]

            # Log train rewards and scores
            self.logger.log_rewards_and_scores(
                rewards,
                logrewards,
                proxy_vals,
                step=self.it,
                prefix="Train batch -",
                use_context=self.use_context,
            )

            # Log trajectory lengths, batch size, logZ and learning rates
            self.logger.log_metrics(
                metrics={
                    "step": self.it,
                    "Trajectory lengths mean": traj_length_mean,
                    "Trajectory lengths min": traj_length_min,
                    "Trajectory lengths max": traj_length_max,
                    "Batch size": len(batch),
                    "logZ": logz,
                    "Learning rate": learning_rates[0],
                    "Learning rate logZ": learning_rates[1],
                },
                step=self.it,
                use_context=self.use_context,
            )

            # Log losses
            losses["Loss"] = losses["all"]
            self.logger.log_metrics(
                metrics=losses,
                step=self.it,
                use_context=self.use_context,
            )

            # Log replay buffer rewards
            if self.buffer.replay_updated:
                rewards_replay = self.buffer.replay.rewards
                self.logger.log_rewards_and_scores(
                    rewards_replay,
                    np.log(rewards_replay),
                    scores=None,
                    step=self.it,
                    prefix="Replay buffer -",
                    use_context=self.use_context,
                )

        t1_log = time.time()
        times.update({"log": t1_log - t0_log})

        # Progress bar
        self.logger.progressbar_update(
            pbar, losses["all"].item(), rewards.tolist(), self.jsd, self.use_context
        )

        # Save intermediate models
        t0_model = time.time()
        if self.evaluator.should_checkpoint(self.it):
            self.logger.save_checkpoint(
                forward_policy=self.forward_policy,
                backward_policy=self.backward_policy,
                state_flow=self.state_flow,
                logZ=self.logZ,
                optimizer=self.opt,
                buffer=self.buffer,
                step=self.it,
            )
        t1_model = time.time()
        times.update({"save_interim_model": t1_model - t0_model})

        return times

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

    def load_checkpoint(self, checkpoint: dict):
        """
        Loads the content of a checkpoint dictionary into the corresponding variables
        of the GFlowNet agent.

        Parameters
        ----------
        checkpoint : dict
            A dictionary containing the following keys:
                - "step": The iteration number of the checkpoint,
                - "forward": The state dict of the forward policy model,
                - "backward": The state dict of the backward policy model,
                - "state_flow": The state dict of the state flow model,
                - "logZ": The tensor containing the parameters of logZ,
                - "optimizer": The state dict of the optimizer,
                - "buffer": A dictionary with keys 'train', 'test' and 'replay', with
                  the relative paths of the corresponding data sets.
        """
        # Iteration: increment by one
        self.it = checkpoint["step"] + 1

        # Forward model
        if checkpoint["forward"] is not None:
            assert self.forward_policy.is_model
            self.forward_policy.model.load_state_dict(checkpoint["forward"])

        # Backward model
        if checkpoint["backward"] is not None:
            assert self.backward_policy.is_model
            self.backward_policy.model.load_state_dict(checkpoint["backward"])

        # State flow model
        if checkpoint["state_flow"] is not None:
            assert self.state_flow
            self.state_flow.model.load_state_dict(checkpoint["state_flow"])

        # LogZ
        if checkpoint["logZ"] is not None:
            assert isinstance(self.logZ, torch.nn.Parameter) and self.logZ.requires_grad
            self.logZ.data = checkpoint["logZ"].to(self.device)

        # Optimizer
        self.opt.load_state_dict(checkpoint["optimizer"])

        if self.logger.debug:
            print("\nCheckpoint loaded into GFlowNet agent\n")


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
