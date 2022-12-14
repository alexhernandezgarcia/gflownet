"""
GFlowNet
TODO:
    - Seeds
"""
import sys
import copy
import time
from collections import defaultdict
from itertools import count
from pathlib import Path

from comet_ml import Experiment
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from gflownet.envs.base import Buffer

# Float and Long tensors
_dev = [torch.device("cpu")]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])
tb = lambda x: torch.BoolTensor(x).to(_dev[0])


def process_config(config):
    if "score" not in config.gflownet.test or "nupack" in config.gflownet.test.score:
        config.gflownet.test.score = config.gflownet.func.replace("nupack ", "")
    return config


def set_device(dev):
    _dev[0] = dev


class GFlowNetAgent:
    def __init__(
        self,
        logdir,
        env,
        seed,
        device,
        optimizer,
        logger,
        comet,
        buffer,
        policy,
        mask_invalid_actions,
        temperature_logits,
        pct_batch_empirical,
        proxy=None,
        al_iter=-1,
        data_path=None,
        sample_only=False,
        **kwargs,
    ):
        # Log directory
        self.logdir = Path(logdir)
        # Environment
        self.env = env
        self.mask_source = tb([self.env.get_mask_invalid_actions()])
        # Seed
        self.rng = np.random.default_rng(seed)
        # Device
        self.device_torch = torch.device(device)
        self.device = self.device_torch
        set_device(self.device_torch)
        # Loss
        if optimizer.loss in ["flowmatch"]:
            self.loss = "flowmatch"
            self.Z = None
        elif optimizer.loss in ["trajectorybalance", "tb"]:
            self.loss = "trajectorybalance"
            self.Z = nn.Parameter(torch.ones(16) * 150.0 / 64)
        else:
            print("Unkown loss. Using flowmatch as default")
            self.loss = "flowmatch"
            self.Z = None
        if not sample_only:
            self.loss_eps = torch.tensor(float(1e-5)).to(self.device)
        # Logging (Comet)
        self.debug = logger.debug
        self.lightweight = logger.lightweight
        self.progress = logger.progress
        self.num_empirical_loss = logger.num_empirical_loss
        if comet.project and not comet.skip and not sample_only:
            self.comet = Experiment(project_name=comet.project, display_summary_level=0)
            if comet.tags:
                if isinstance(comet.tags, list):
                    self.comet.add_tags(comet.tags)
                else:
                    self.comet.add_tag(comet.tags)
            #             self.comet.log_parameters(vars(args))
            if self.logdir.exists():
                with open(self.logdir / "comet.url", "w") as f:
                    f.write(self.comet.url + "\n")
        else:
            if isinstance(comet, Experiment):
                self.comet = comet
            else:
                self.comet = None
        self.log_times = comet.log_times
        self.test_period = logger.test.period
        if self.test_period in [None, -1]:
            self.test_period = np.inf
        self.oracle_period = logger.oracle.period
        self.oracle_n = logger.oracle.n
        self.oracle_k = logger.oracle.k
        # Buffers
        self.buffer = Buffer(**buffer, env=self.env, make_train_test=not sample_only)
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
        self.forward_policy = Policy(
            policy.forward,
            self.env.obs_dim,
            len(self.env.action_space),
        )
        if policy.forward.checkpoint:
            if self.logdir.exists():
                if (self.logdir / "ckpts").exists():
                    self.policy_forward_path = (
                        self.logdir / "ckpts" / policy.forward.checkpoint
                    )
                else:
                    self.policy_forward_path = self.logdir / policy.forward.checkpoint
            else:
                self.policy_forward_path = policy.forward.checkpoint
            if self.policy_forward_path.exists() and policy.forward.reload_ckpt:
                self.forward_policy.load_state_dict(
                    torch.load(self.policy_forward_path)
                )
                print("Reloaded GFN forward policy model Checkpoint")
        else:
            self.policy_forward_path = None
        if policy.backward:
            self.backward_policy = Policy(
                policy.backward,
                self.env.obs_dim,
                len(self.env.action_space),
                base=self.forward_policy,
            )
        else:
            self.backward_policy = None
        if self.backward_policy and policy.backward.checkpoint:
            if self.logdir.exists():
                if (self.logdir / "ckpts").exists():
                    self.policy_backward_path = (
                        self.logdir / "ckpts" / policy.backward.checkpoint
                    )
                else:
                    self.policy_backward_path = self.logdir / policy.backward.checkpoint
            else:
                self.policy_backward_path = policy.backward.checkpoint
            if self.policy_backward_path.exists() and policy.backward.reload_ckpt:
                self.backward_policy.load_state_dict(
                    torch.load(self.policy_backward_path)
                )
                print("Reloaded GFN backward policy model Checkpoint")
        else:
            self.policy_backward_path = None
        if self.backward_policy and self.backward_policy.is_model:
            self.backward_policy.model.to(self.device)
        self.ckpt_period = policy.ckpt_period
        if self.ckpt_period in [None, -1]:
            self.ckpt_period = np.inf
        # Optimizer
        if self.forward_policy.is_model:
            self.forward_policy.model.to(self.device)
            self.target = copy.deepcopy(self.forward_policy.model)
            self.opt, self.lr_scheduler = make_opt(self.parameters(), self.Z, optimizer)
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
        if al_iter >= 0:
            self.al_iter = "_iter{}".format(al_iter)
        else:
            self.al_iter = ""
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # Training
        self.mask_invalid_actions = mask_invalid_actions
        self.temperature_logits = temperature_logits
        self.pct_batch_empirical = pct_batch_empirical

    def parameters(self):
        if self.backward_policy is None or self.backward_policy.is_model == False:
            return list(self.forward_policy.model.parameters())
        elif self.loss == "trajectorybalance":
            return list(self.forward_policy.model.parameters()) + list(
                self.backward_policy.model.parameters()
            )
        else:
            raise ValueError("Backward Policy cannot be a nn in flowmatch.")

    def forward_sample(
        self, envs, times, sampling_method="policy", model=None, temperature=1.0
    ):
        """
        Performs a forward action on each environment of a list.

        Args
        ----
        env : list of GFlowNetEnv or derived
            A list of instances of the environment

        times : dict
            Dictionary to store times

        sampling_method : string
            - model: uses current forward to obtain the sampling probabilities.
            - uniform: samples uniformly from the action space.

        model : torch model
            Model to use as policy if sampling_method="policy"

        temperature : float
            Temperature to adjust the logits by logits /= temperature
        """
        if not isinstance(envs, list):
            envs = [envs]
        states = [env.state2obs() for env in envs]
        mask_invalid_actions = tb([env.get_mask_invalid_actions() for env in envs])
        random_action = self.rng.uniform()
        t0_a_model = time.time()
        if sampling_method == "policy":
            with torch.no_grad():
                action_logits = model(tf(states))
            action_logits /= temperature
        elif sampling_method == "uniform":
            action_logits = tf(np.zeros(len(states)), len(self.env.action_space) + 1)
        else:
            raise NotImplemented
        if self.mask_invalid_actions:
            action_logits[mask_invalid_actions] = -1000
        if all(torch.isfinite(action_logits).flatten()):
            actions = Categorical(logits=action_logits).sample().tolist()
        else:
            if self.debug:
                raise ValueError("Action could not be sampled from model!")
        t1_a_model = time.time()
        times["actions_model"] += t1_a_model - t0_a_model
        assert len(envs) == len(actions)
        # Execute actions
        _, actions, valids = zip(
            *[env.step(action) for env, action in zip(envs, actions)]
        )
        return envs, actions, valids

    def backward_sample(
        self, env, sampling_method="policy", model=None, temperature=1.0, done=False
    ):
        """
        Performs a backward action on one environment.

        Args
        ----
        env : GFlowNetEnv or derived
            An instance of the environment

        sampling_method : string
            - model: uses the current backward policy to obtain the sampling probabilities.
            - uniform: samples uniformly from the parents of the current state.

        model : torch model
            Model to use as policy if sampling_method="policy"

        temperature : float
            Temperature to adjust the logits by logits /= temperature

        done : bool
            If True, it will sample eos action
        """
        # TODO: If sampling method is policy
        # Change backward sampling with a mask for parents
        # As we can use the backward policy to get model(states) but not all of those actions are likely
        # Need to compute backward_masks, amsk those actions and then get the categorical distribution.
        parents, parents_a = env.get_parents(env.state, done)
        if sampling_method == "policy":
            with torch.no_grad():
                action_logits = model(tf(parents))[
                    torch.arange(len(parents)), parents_a
                ]
            action_logits /= temperature
            if all(torch.isfinite(action_logits).flatten()):
                action_idx = Categorical(logits=action_logits).sample().item()
            else:
                if self.debug:
                    raise ValueError("Action could not be sampled from model!")
        elif sampling_method == "uniform":
            action_idx = self.rng.integers(low=0, high=len(parents_a))
        else:
            raise NotImplemented
        action = parents_a[action_idx]
        env.set_state(env.obs2state((parents)[action_idx]), done=False)
        return env, env.state, action, parents, parents_a

    def sample_batch(
        self, envs, n_samples=None, train=True, model=None, progress=False
    ):
        """
        Builds a batch of data

        if train == True:
            Each item in the batch is a list of 7 elements (all tensors):
                - [0] the state, as state2obs(state)
                - [1] the action
                - [2] reward of the state
                - [3] all parents of the state, parents
                - [4] actions that lead to the state from each parent, parents_a
                - [5] done [True, False]
                - [6] traj id: identifies each trajectory
                - [7] state id: identifies each state within a trajectory
                - [8] mask: invalid actions from that state are 1
        else:
            Each item in the batch is a list of 1 element:
                - [0] the states (state)

        Args
        ----
        """
        times = {
            "all": 0.0,
            "actions_model": 0.0,
            "actions_envs": 0.0,
            "rewards": 0.0,
        }
        t0_all = time.time()
        batch = []
        if model is None:
            model = self.forward_policy
        if isinstance(envs, list):
            envs = [env.reset(idx) for idx, env in enumerate(envs)]
        elif n_samples is not None and n_samples > 0:
            envs = [copy.deepcopy(envs).reset(idx) for idx in range(n_samples)]
        else:
            return None, None
        # Offline trajectories
        # TODO: Replay Buffer
        if train:
            n_empirical = int(self.pct_batch_empirical * len(envs))
            for env in envs[:n_empirical]:
                readable = self.rng.permutation(self.buffer.train.samples.values)[0]
                env = env.set_state(env.readable2state(readable), done=True)
                action = env.eos
                parents = [env.state2obs(env.state)]
                parents_a = [action]
                mask = env.get_mask_invalid_actions()
                n_actions = 0
                while len(env.state) > 0:
                    batch.append(
                        [
                            tf([env.state2obs(env.state)]),
                            tl([action]),
                            env.state,
                            tf(parents),
                            tl(parents_a),
                            env.done,
                            tl([env.id] * len(parents)),
                            tl([n_actions]),
                            tb([mask]),
                        ]
                    )
                    # Backward sampling
                    env, state, action, parents, parents_a = self.backward_sample(
                        env,
                        sampling_method="policy",
                        model=self.backward_policy,
                        temperature=self.temperature_logits,
                    )
                    n_actions += 1
            envs = envs[n_empirical:]
        # Policy trajectories
        while envs:
            # Forward sampling
            if train is False:
                envs, actions, valids = self.forward_sample(
                    envs,
                    times,
                    sampling_method="policy",
                    model=self.forward_policy,
                    temperature=1.0,
                )
            else:
                envs, actions, valids = self.forward_sample(
                    envs,
                    times,
                    sampling_method="policy",
                    model=self.forward_policy,
                    temperature=self.temperature_logits,
                )
            t0_a_envs = time.time()
            # Add to batch
            for env, action, valid in zip(envs, actions, valids):
                if valid:
                    parents, parents_a = env.get_parents()
                    mask = env.get_mask_invalid_actions()
                    assert action in parents_a
                    if train:
                        batch.append(
                            [
                                tf([env.state2obs()]),
                                tl([action]),
                                env.state,
                                tf(parents),
                                tl(parents_a),
                                env.done,
                                tl([env.id] * len(parents)),
                                tl([env.n_actions - 1]),
                                tb([mask]),
                            ]
                        )
                    else:
                        if env.done:
                            batch.append(env.state)
            # Filter out finished trajectories
            envs = [env for env in envs if not env.done]
            t1_a_envs = time.time()
            times["actions_envs"] += t1_a_envs - t0_a_envs
            if progress and n_samples is not None:
                print(f"{n_samples - len(envs)}/{n_samples} done")
        if train:
            # Compute rewards
            (
                obs,
                actions,
                states,
                parents,
                parents_a,
                done,
                traj_id,
                state_id,
                masks,
            ) = zip(*batch)
            t0_rewards = time.time()
            rewards = env.reward_batch(states, done)
            t1_rewards = time.time()
            times["rewards"] += t1_rewards - t0_rewards
            rewards = [tf([r]) for r in rewards]
            done = [tl([d]) for d in done]
            batch = list(
                zip(
                    obs,
                    actions,
                    rewards,
                    parents,
                    parents_a,
                    done,
                    traj_id,
                    state_id,
                    masks,
                )
            )
        t1_all = time.time()
        times["all"] += t1_all - t0_all
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
        loginf = tf([loginf])
        batch_idxs = tl(
            sum(
                [
                    [i] * len(parents)
                    for i, (_, _, _, parents, _, _, _, _, _) in enumerate(batch)
                ],
                [],
            )
        )
        sp, _, r, parents, actions, done, _, _, masks = map(torch.cat, zip(*batch))
        # Sanity check if negative rewards
        if self.debug and torch.any(r < 0):
            neg_r_idx = torch.where(r < 0)[0].tolist()
            for idx in neg_r_idx:
                obs = sp[idx].tolist()
                state = self.env.obs2state(obs)
                state_oracle = self.env.state2oracle([state])
                output_proxy = self.env.proxy(state_oracle)
                reward = self.env.proxy2reward(output_proxy)
                import ipdb

                ipdb.set_trace()

        # Q(s,a)
        parents_Qsa = self.forward_policy(parents)[
            torch.arange(parents.shape[0]), actions
        ]

        # log(eps + exp(log(Q(s,a)))) : qsa
        in_flow = torch.log(
            self.loss_eps
            + tf(torch.zeros((sp.shape[0],))).index_add_(
                0, batch_idxs, torch.exp(parents_Qsa)
            )
        )
        # the following with work if autoregressive
        #         in_flow = torch.logaddexp(parents_Qsa[batch_idxs], torch.log(self.loss_eps))
        if self.tau > 0 and self.target is not None:
            with torch.no_grad():
                next_q = self.target(sp)
        else:
            next_q = self.forward_policy(sp)
        next_q[masks] = -loginf
        qsp = torch.logsumexp(next_q, 1)
        # qsp: qsp if not done; -loginf if done
        qsp = qsp * (1 - done) - loginf * done
        out_flow = torch.logaddexp(torch.log(r + self.loss_eps), qsp)
        loss = (in_flow - out_flow).pow(2).mean()

        with torch.no_grad():
            term_loss = ((in_flow - out_flow) * done).pow(2).sum() / (
                done.sum() + 1e-20
            )
            flow_loss = ((in_flow - out_flow) * (1 - done)).pow(2).sum() / (
                (1 - done).sum() + 1e-20
            )

        if self.tau > 0 and self.target is not None:
            for a, b in zip(
                self.forward_policy.model.parameters(), self.target.parameters()
            ):
                b.data.mul_(1 - self.tau).add_(self.tau * a)

        return loss, term_loss, flow_loss

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
        loginf = tf([loginf])
        # Unpack batch
        (
            states,
            actions,
            rewards,
            parents,
            parents_a,
            done,
            traj_id_parents,
            state_id,
            masks,
        ) = zip(*batch)
        # Keep only parents in trajectory
        parents = [
            p[torch.where(a == p_a)] for a, p, p_a in zip(actions, parents, parents_a)
        ]
        traj_id = torch.cat([el[:1] for el in traj_id_parents])
        # Concatenate lists of tensors
        states, actions, rewards, parents, done, state_id, masks = map(
            torch.cat,
            [
                states,
                actions,
                rewards,
                parents,
                done,
                state_id,
                masks,
            ],
        )
        # Build forward masks from state masks
        masks_f = torch.cat(
            [
                masks[torch.where((state_id == sid - 1) & (traj_id == pid))]
                if sid > 0
                else self.mask_source
                for sid, pid in zip(state_id, traj_id)
            ]
        )
        # Build backward masks from parents actions
        masks_b = torch.ones(masks.shape, dtype=bool)
        # TODO: this should be possible with a matrix operation
        for idx, pa in enumerate(parents_a):
            masks_b[idx, pa] = False
        # Forward trajectories
        logits_f = self.forward_policy(parents)
        logits_f[masks_f] = -loginf
        logprobs_f = self.logsoftmax(logits_f)[torch.arange(logits_f.shape[0]), actions]
        sumlogprobs_f = tf(
            torch.zeros(len(torch.unique(traj_id, sorted=True)))
        ).index_add_(0, traj_id, logprobs_f)
        # Backward trajectories
        logits_b = self.backward_policy(states)
        logits_b[masks_b] = -loginf
        logprobs_b = self.logsoftmax(logits_b)[torch.arange(logits_b.shape[0]), actions]
        sumlogprobs_b = tf(
            torch.zeros(len(torch.unique(traj_id, sorted=True)))
        ).index_add_(0, traj_id, logprobs_b)
        # Sort rewards of done states by ascending traj_id
        rewards = rewards[done.eq(1)][torch.argsort(traj_id[done.eq(1)])]
        # Trajectory balance loss
        loss = (
            (self.Z.sum() + sumlogprobs_f - sumlogprobs_b - torch.log(rewards))
            .pow(2)
            .mean()
        )
        return loss, loss, loss

    def unpack_terminal_states(self, batch):
        trajs = [[] for _ in range(self.batch_size)]
        states = [None] * self.batch_size
        rewards = [None] * self.batch_size
        #         state_ids = [[-1] for _ in range(self.batch_size)]
        for el in batch:
            traj_id = el[6][:1].item()
            state_id = el[7][:1].item()
            #             assert state_ids[traj_id][-1] + 1 == state_id
            #             state_ids[traj_id].append(state_id)
            trajs[traj_id].append(el[1][0].item())
            if bool(el[5].item()):
                states[traj_id] = tuple(self.env.obs2state(el[0][0]))
                rewards[traj_id] = el[2][0].item()
        trajs = [tuple(el) for el in trajs]
        return states, trajs, rewards

    def train(self):
        # Metrics
        all_losses = []
        all_visited = []
        loss_term_ema = None
        loss_flow_ema = None
        # Generate list of environments
        envs = [copy.deepcopy(self.env).reset() for _ in range(self.batch_size)]
        # Train loop
        for it in tqdm(range(self.n_train_steps + 1), disable=not self.progress):
            t0_iter = time.time()
            data = []
            for j in range(self.sttr):
                batch, times = self.sample_batch(envs)
                data += batch
            for j in range(self.ttsr):
                if self.loss == "flowmatch":
                    losses = self.flowmatch_loss(
                        it * self.ttsr + j, data
                    )  # returns (opt loss, *metrics)
                elif self.loss == "trajectorybalance":
                    losses = self.trajectorybalance_loss(
                        it * self.ttsr + j, data
                    )  # returns (opt loss, *metrics)
                else:
                    print("Unknown loss!")
                if not all([torch.isfinite(loss) for loss in losses]):
                    if self.debug:
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
            states_term, trajs_term, rewards = self.unpack_terminal_states(batch)
            proxy_vals = self.env.reward2proxy(rewards)
            self.buffer.add(states_term, trajs_term, rewards, proxy_vals, it)
            self.buffer.add(
                states_term, trajs_term, rewards, proxy_vals, it, buffer="replay"
            )
            # Log
            idx_best = np.argmax(rewards)
            state_best = "".join(self.env.state2readable(states_term[idx_best]))
            if self.lightweight:
                all_losses = all_losses[-100:]
                all_visited = states_term

            else:
                all_visited.extend(states_term)
            if self.comet:
                self.comet.log_text(
                    state_best + " / proxy: {}".format(proxy_vals[idx_best]), step=it
                )
                self.comet.log_metrics(
                    dict(
                        zip(
                            [
                                "mean_reward{}".format(self.al_iter),
                                "max_reward{}".format(self.al_iter),
                                "mean_proxy{}".format(self.al_iter),
                                "min_proxy{}".format(self.al_iter),
                                "max_proxy{}".format(self.al_iter),
                                "mean_seq_length{}".format(self.al_iter),
                                "batch_size{}".format(self.al_iter),
                            ],
                            [
                                np.mean(rewards),
                                np.max(rewards),
                                np.mean(proxy_vals),
                                np.min(proxy_vals),
                                np.max(proxy_vals),
                                np.mean([len(state) for state in states_term]),
                                len(data),
                            ],
                        )
                    ),
                    step=it,
                )
            # Test set metrics
            if not it % self.test_period and self.buffer.test is not None:
                data_logq = []
                times.update(
                    {
                        "test_trajs": 0.0,
                        "test_logq": 0.0,
                    }
                )
                # TODO: this could be done just once and store it
                for statestr, score in tqdm(
                    zip(self.buffer.test.samples, self.buffer.test["energies"]),
                    disable=self.test_period < 10,
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
                    data_logq.append(
                        logq(traj_list, actions, self.forward_policy, self.env)
                    )
                    t1_test_logq = time.time()
                    times["test_logq"] += t1_test_logq - t0_test_logq
                corr = np.corrcoef(data_logq, self.buffer.test["energies"])
                if self.comet:
                    self.comet.log_metrics(
                        dict(
                            zip(
                                [
                                    "test_corr_logq_score{}".format(self.al_iter),
                                    "test_mean_logq{}".format(self.al_iter),
                                ],
                                [
                                    corr[0, 1],
                                    np.mean(data_logq),
                                ],
                            )
                        ),
                        step=it,
                    )
            # Oracle metrics (for monitoring)
            if not it % self.oracle_period and self.debug:
                oracle_batch, oracle_times = self.sample_batch(
                    self.env, self.oracle_n, train=False
                )
                oracle_dict, oracle_times = batch2dict(
                    oracle_batch, self.env, get_uncertainties=False
                )
                energies = oracle_dict["energies"]
                energies_sorted = np.sort(energies)
                dict_topk = {}
                for k in self.oracle_k:
                    mean_topk = np.mean(energies_sorted[:k])
                    dict_topk.update(
                        {"oracle_mean_top{}{}".format(k, self.al_iter): mean_topk}
                    )
                    if self.comet:
                        self.comet.log_metrics(dict_topk)
            if not it % 100:
                if not self.lightweight:
                    l1_error, kl_div = empirical_distribution_error(
                        self.env, all_visited[-self.num_empirical_loss :]
                    )
                else:
                    l1_error, kl_div = 1, 100
                if self.progress:
                    if self.debug:
                        print("Empirical L1 distance", l1_error, "KL", kl_div)
                        if len(all_losses):
                            print(
                                *[
                                    f"{np.mean([i[j] for i in all_losses[-100:]]):.5f}"
                                    for j in range(len(all_losses[0]))
                                ]
                            )
                if self.comet:
                    self.comet.log_metrics(
                        dict(
                            zip(
                                [
                                    "loss{}".format(self.al_iter),
                                    "term_loss{}".format(self.al_iter),
                                    "flow_loss{}".format(self.al_iter),
                                    "l1{}".format(self.al_iter),
                                    "kl{}".format(self.al_iter),
                                ],
                                [loss.item() for loss in losses] + [l1_error, kl_div],
                            )
                        ),
                        step=it,
                    )
                    if not self.lightweight:
                        self.comet.log_metric(
                            "unique_states{}".format(self.al_iter),
                            np.unique(all_visited).shape[0],
                            step=it,
                        )
            # Save intermediate models
            if not it % self.ckpt_period:
                if self.policy_forward_path:
                    path = self.policy_forward_path.parent / Path(
                        self.model_path.stem
                        + "{}_iter{:06d}".format(self.al_iter, it)
                        + self.policy_forward_path.suffix
                    )
                    torch.save(self.forward_policy.model.state_dict(), path)
                if self.policy_backward_path:
                    path = self.policy_backward_path.parent / Path(
                        self.model_path.stem
                        + "{}_iter{:06d}".format(self.al_iter, it)
                        + self.policy_backward_path.suffix
                    )
                    torch.save(self.backward_policy.model.state_dict(), path)
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
            times = {"time_{}{}".format(k, self.al_iter): v for k, v in times.items()}
            if self.comet and self.log_times:
                self.comet.log_metrics(times, step=it)
        # Save final model
        if self.policy_forward_path:
            path = self.policy_forward_path.parent / Path(
                self.policy_forward_path.stem
                + "_final"
                + self.policy_forward_path.suffix
            )
            torch.save(self.forward_policy.model.state_dict(), path)
            torch.save(self.forward_policy.model.state_dict(), self.policy_forward_path)
        if self.policy_backward_path:
            path = self.policy_backward_path.parent / Path(
                self.policy_backward_path.stem
                + "_final"
                + self.policy_backward_path.suffix
            )
            torch.save(self.backward_policy.model.state_dict(), path)
            torch.save(
                self.backward_policy.model.state_dict(), self.policy_backward_path
            )

        # Close comet
        if self.comet and self.al_iter == -1:
            self.comet.end()


class Policy:
    def __init__(self, config, state_dim, n_actions, base=None):
        self.state_dim = state_dim
        self.n_actions = n_actions
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
        if self.type == "uniform":
            self.model = self.uniform_distribution
            self.is_model = False
        elif self.type == "mlp":
            self.model = self.make_mlp(nn.LeakyReLU())
            self.is_model = True
        else:
            raise "Policy model type not defined"

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
                [self.state_dim] + [self.n_hid] * self.n_layers + [(self.n_actions + 1)]
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

    def uniform_distribution(self, states):
        """
        Return action logits (log probabilities) from a uniform distribution
        Args: states: tensor
        """
        return tf(np.ones((len(states), self.n_actions + 1)))


def batch2dict(batch, env, get_uncertainties=False, query_function="Both"):
    batch = np.asarray(env.state2oracle(batch))
    t0_proxy = time.time()
    if get_uncertainties:
        if query_function == "fancy_acquisition":
            scores, proxy_vals, uncertainties = env.proxy(batch, query_function)
        else:
            proxy_vals, uncertainties = env.proxy(batch, query_function)
            scores = proxy_vals
    else:
        proxy_vals = env.proxy(batch)
        uncertainties = None
        scores = proxy_vals
    t1_proxy = time.time()
    times = {"proxy": t1_proxy - t0_proxy}
    samples = {
        "samples": batch.astype(np.int64),
        "scores": scores,
        "energies": proxy_vals,
        "uncertainties": uncertainties,
    }
    return samples, times


def make_opt(params, Z, config):
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
        if Z is not None:
            opt.add_param_group(
                {
                    "params": Z,
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


def empirical_distribution_error(env, visited, epsilon=1e-9):
    """
    Computes the empirical distribution errors, as the mean L1 error and the KL
    divergence between the true density of the space and the estimated density from all
    states visited.
    """
    true_density, _, states_term = env.true_density()
    if true_density is None:
        return None, None
    true_density = tf(true_density)
    if not len(visited):
        return 1, 100
    hist = defaultdict(int)
    for s in visited:
        hist[s] += 1
    Z = sum([hist[s] for s in states_term]) + epsilon
    estimated_density = tf([hist[s] / Z for s in states_term])
    # L1 error
    l1 = abs(estimated_density - true_density).mean().item()
    # KL divergence
    kl = (true_density * torch.log(estimated_density / true_density)).sum().item()
    return l1, kl


def logq(traj_list, actions_list, model, env, loginf=1000):
    # TODO: this method is probably suboptimal, since it may repeat forward calls for
    # the same nodes.
    log_q = torch.tensor(1.0)
    loginf = tf([loginf])
    for traj, actions in zip(traj_list, actions_list):
        traj = traj[::-1]
        actions = actions[::-1]
        traj_obs = np.asarray([env.state2obs(state) for state in traj])
        masks = tb([env.get_mask_invalid_actions(state, 0) for state in traj])
        with torch.no_grad():
            logits_traj = model(tf(traj_obs))
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
