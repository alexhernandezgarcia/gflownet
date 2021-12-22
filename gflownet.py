"""
GFlowNet
TODO:
    - Seeds
"""
from comet_ml import Experiment
from argparse import ArgumentParser
import copy
import gzip
import heapq
import itertools
import os
import pickle
from collections import defaultdict
from itertools import count, product
from pathlib import Path
import yaml
import time

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from oracles import linearToy, toyHamiltonian, PottsEnergy, seqfoldScore, nupackScore
from utils import get_config, namespace2dict, numpy2python

# Float and Long tensors
_dev = [torch.device("cpu")]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """
    args2config = {}
    # YAML config
    parser.add_argument(
        "-y",
        "--yaml_config",
        default=None,
        type=str,
        help="YAML configuration file",
    )
    args2config.update({"yaml_config": ["yaml_config"]})
    # General
    parser.add_argument("--workdir", default=None, type=str)
    args2config.update({"workdir": ["workdir"]})
    parser.add_argument("--device", default="cpu", type=str)
    args2config.update({"device": ["gflownet", "device"]})
    parser.add_argument("--progress", action="store_true")
    args2config.update({"progress": ["gflownet", "progress"]})
    parser.add_argument("--debug", action="store_true")
    args2config.update({"debug": ["debug"]})
    parser.add_argument("--model_ckpt", default=None, type=str)
    args2config.update({"model_ckpt": ["gflownet", "model_ckpt"]})
    parser.add_argument("--ckpt_period", default=None, type=int)
    args2config.update({"ckpt_period": ["gflownet", "ckpt_period"]})
    # Training hyperparameters
    parser.add_argument(
        "--early_stopping",
        default=0.01,
        help="Threshold loss for early stopping",
        type=float,
    )
    args2config.update({"early_stopping": ["gflownet", "early_stopping"]})
    parser.add_argument(
        "--ema_alpha",
        default=0.5,
        help="alpha coefficient for exponential moving average",
        type=float,
    )
    args2config.update({"ema_alpha": ["gflownet", "ema_alpha"]})
    parser.add_argument(
        "--learning_rate", default=1e-4, help="Learning rate", type=float
    )
    args2config.update({"learning_rate": ["gflownet", "learning_rate"]})
    parser.add_argument("--opt", default="adam", type=str)
    args2config.update({"opt": ["gflownet", "opt"]})
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    args2config.update({"adam_beta1": ["gflownet", "adam_beta1"]})
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    args2config.update({"adam_beta2": ["gflownet", "adam_beta2"]})
    parser.add_argument(
        "--reward_beta_init",
        default=1,
        type=float,
        help="Initial beta for exponential reward scaling",
    )
    args2config.update({"reward_beta_init": ["gflownet", "reward_beta_init"]})
    parser.add_argument(
        "--reward_max",
        default=1e6,
        type=float,
        help="Max reward to prevent numerical issues",
    )
    args2config.update({"reward_max": ["gflownet", "reward_max"]})
    parser.add_argument(
        "--reward_beta_mult",
        default=1.25,
        type=float,
        help="Multiplier for rescaling beta during training",
    )
    args2config.update({"reward_beta_mult": ["gflownet", "reward_beta_mult"]})
    parser.add_argument(
        "--reward_beta_period",
        default=-1,
        type=float,
        help="Period (number of iterations) for beta rescaling",
    )
    args2config.update({"reward_beta_period": ["gflownet", "reward_beta_period"]})
    parser.add_argument("--momentum", default=0.9, type=float)
    args2config.update({"momentum": ["gflownet", "momentum"]})
    parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
    args2config.update({"mbsize": ["gflownet", "mbsize"]})
    parser.add_argument("--train_to_sample_ratio", default=1, type=float)
    args2config.update({"train_to_sample_ratio": ["gflownet", "train_to_sample_ratio"]})
    parser.add_argument("--n_hid", default=256, type=int)
    args2config.update({"n_hid": ["gflownet", "n_hid"]})
    parser.add_argument("--n_layers", default=2, type=int)
    args2config.update({"n_layers": ["gflownet", "n_layers"]})
    parser.add_argument("--n_train_steps", default=20000, type=int)
    args2config.update({"n_train_steps": ["gflownet", "n_iter"]})
    parser.add_argument(
        "--num_empirical_loss",
        default=200000,
        type=int,
        help="Number of samples used to compute the empirical distribution loss",
    )
    args2config.update({"num_empirical_loss": ["gflownet", "num_empirical_loss"]})
    parser.add_argument("--clip_grad_norm", default=0.0, type=float)
    args2config.update({"clip_grad_norm": ["gflownet", "clip_grad_norm"]})
    parser.add_argument("--random_action_prob", default=0.0, type=float)
    args2config.update({"random_action_prob": ["gflownet", "random_action_prob"]})
    # Environment
    parser.add_argument("--func", default="arbitrary_i")
    args2config.update({"func": ["gflownet", "func"]})
    parser.add_argument(
        "--horizon",
        default=42,
        help="Maximum number of episodes; maximum sequence length",
        type=int,
    )
    args2config.update({"horizon": ["gflownet", "horizon"]})
    parser.add_argument("--nalphabet", default=4, type=int)
    args2config.update({"nalphabet": ["gflownet", "nalphabet"]})
    parser.add_argument("--min_word_len", default=1, type=int)
    args2config.update({"min_word_len": ["gflownet", "min_word_len"]})
    parser.add_argument("--max_word_len", default=1, type=int)
    args2config.update({"max_word_len": ["gflownet", "max_word_len"]})
    args2config.update({"learning_rate": ["gflownet", "learning_rate"]})
    # Sampling
    parser.add_argument("--bootstrap_tau", default=0.0, type=float)
    args2config.update({"bootstrap_tau": ["gflownet", "bootstrap_tau"]})
    parser.add_argument("--batch_reward", action="store_true")
    args2config.update({"batch_reward": ["gflownet", "batch_reward"]})
    # Comet
    parser.add_argument("--comet_project", default=None, type=str)
    args2config.update({"comet_project": ["gflownet", "comet", "project"]})
    parser.add_argument(
        "-t", "--tags", nargs="*", help="Comet.ml tags", default=[], type=str
    )
    args2config.update({"tags": ["gflownet", "comet", "tags"]})
    return parser, args2config


def set_device(dev):
    _dev[0] = dev


class AptamerSeq:
    """
    Aptamer sequence environment

    Attributes
    ----------
    horizon : int
        Maximum length of the sequences

    nalphabet : int
        Number of letters in the alphabet

    seq : list
        Representation of a sequence (state), as a list of length horizon where each
        element is the index of a letter in the alphabet, from 0 to (nalphabet - 1).

    done : bool
        True if the sequence has reached a terminal state (maximum length, or stop
        action executed.

    func : str
        Name of the reward function

    proxy : lambda
        Proxy model
    """

    def __init__(
        self,
        horizon=42,
        nalphabet=4,
        min_word_len=1,
        max_word_len=1,
        func="default",
        proxy=None,
        allow_backward=False,
        debug=False,
        reward_beta=1,
    ):
        self.horizon = horizon
        self.nalphabet = nalphabet
        self.min_word_len = min_word_len
        self.max_word_len = max_word_len
        self.seq = []
        self.done = False
        self.func = func
        if proxy:
            self.proxy = proxy
        else:
            self.proxy = {
                "default": None,
                "arbitrary_i": self.reward_arbitrary_i,
                "linear": linearToy,
                "innerprod": toyHamiltonian,
                "potts": PottsEnergy,
                "seqfold": seqfoldScore,
                "nupack energy": lambda x: nupackScore(x, returnFunc="energy"),
                "nupack pairs": lambda x: nupackScore(x, returnFunc="pairs"),
                "nupack pins": lambda x: nupackScore(x, returnFunc="hairpins"),
            }[self.func]
        self.reward = (
            lambda x: [0]
            if not self.done
            else self.proxy2reward(self.proxy(self.seq2oracle(x)))
        )
        self.allow_backward = allow_backward
        self._true_density = None
        self.debug = debug
        self.reward_beta = reward_beta
        self.action_space = self.get_actions_space(
            self.nalphabet, np.arange(self.min_word_len, self.max_word_len + 1)
        )
        self.nactions = len(self.action_space)

    def get_actions_space(self, nalphabet, valid_wordlens):
        """
        Constructs with all possible actions
        """
        alphabet = [a for a in range(nalphabet)]
        actions = []
        for r in valid_wordlens:
            actions_r = [el for el in product(alphabet, repeat=r)]
            actions += actions_r
        return actions

    def reward_arbitrary_i(self, seq):
        if len(seq) > 0:
            return (seq[-1] + 1) * len(seq)
        else:
            return 0

    def seq2oracle(self, seq):
        """
        Prepares a sequence in "GFlowNet format" for the oracles.

        Args
        ----
        seq : list of lists
            List of sequences.
        """
        queries = [s + [-1] * (self.horizon - len(s)) for s in seq]
        queries = np.array(queries, dtype=int)
        if queries.ndim == 1:
            queries = queries[np.newaxis, ...]
        queries += 1
        if queries.shape[1] == 1:
            import ipdb

            ipdb.set_trace()
            queries = np.column_stack((queries, np.zeros(queries.shape[0])))
        return queries

    def reward_batch(self, seq, done):
        seq = [s for s, d in zip(seq, done) if d]
        reward = np.zeros(len(done))
        reward[list(done)] = self.proxy2reward(self.proxy(self.seq2oracle(seq)))
        return reward

    def proxy2reward(self, proxy_vals):
        """
        Prepares the output of an oracle for GFlowNet.
        """
        if "pins" in self.func or "pairs" in self.func:
            return np.exp(self.reward_beta * proxy_vals)
        else:
            return np.exp(-self.reward_beta * proxy_vals)

    def reward2proxy(self, reward):
        """
        Converts a "GFlowNet reward" into energy or values as returned by an oracle.
        """
        if "pins" in self.func or "pairs" in self.func:
            return np.log(reward) / self.reward_beta
        else:
            return -np.log(reward) / self.reward_beta

    def seq2obs(self, seq=None):
        """
        Transforms the sequence (state) given as argument (or self.seq if None) into a
        one-hot encoding. The output is a list of length nalphabet * nhorizon, where
        each n-th successive block of nalphabet elements is a one-hot encoding of the
        letter in the n-th position.

        Example:
          - Sequence: AACTG
          - State, seq: [0, 0, 1, 3, 2]
                         A, A, C, T, G
          - seq2obs(seq): [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                          |     A    |      A    |      C    |      T    |      G    |

        If horizon > len(s), the last (horizon - len(s)) blocks are all 0s.
        """
        if seq is None:
            seq = self.seq

        z = np.zeros((self.nalphabet * self.horizon), dtype=np.float32)

        if len(seq) > 0:
            if hasattr(
                seq[0], "device"
            ):  # if it has a device at all, it will be cuda (CPU numpy array has no dev
                seq = [subseq.cpu().detach().numpy() for subseq in seq]

            z[(np.arange(len(seq)) * self.nalphabet + seq)] = 1
        return z

    def obs2seq(self, obs):
        """
        Transforms the one-hot encoding version of a sequence (state) given as argument
        into a a sequence of letter indices.

        Example:
          - Sequence: AACTG
          - obs: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                 |     A    |      A    |      C    |      T    |      G    |
          - seq: [0, 0, 1, 3, 2]
                  A, A, C, T, G
        """
        obs_mat = np.reshape(obs, (self.horizon, self.nalphabet))
        seq = np.where(obs_mat)[1]
        return seq

    def seq2letters(self, seq, alphabet={0: "A", 1: "T", 2: "C", 3: "G"}):
        """
        Transforms a sequence given as a list of indices into a sequence of letters
        according to an alphabet.
        """
        return [alphabet[el] for el in seq]

    def reset(self):
        """
        Resets the environment
        """
        self.seq = []
        self.done = False
        return self

    def parent_transitions(self, seq, action):
        # TODO: valid parents must satisfy horizon constraint!!!
        """
        Determines all parents and actions that lead to sequence (state) seq

        Args
        ----
        seq : list
            Representation of a sequence (state), as a list of length horizon where each
        element is the index of a letter in the alphabet, from 0 to (nalphabet - 1).

        action : int
            Last action performed

        Returns
        -------
        parents : list
            List of parents as seq2obs(seq)

        actions : list
            List of actions that lead to seq for each parent in parents
        """
        if action == self.nactions:
            return [self.seq2obs(seq)], [action]
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                if seq[-len(a) :] == list(a):
                    parents.append(self.seq2obs(seq[: -len(a)]))
                    actions.append(idx)
        return parents, actions

    def step(self, action):
        """
        Define step given action and state.

        See: step_daug()
        See: step_chain()
        """
        if self.allow_backward:
            return self.step_chain(action)
        return self.step_dag(action)

    def step_dag(self, action):
        """
        Executes step given an action

        If action is smaller than nactions (no stop), add action to next
        position.

        See: step_daug()
        See: step_chain()

        Args
        ----
        a : int
            Index of action in the action space. a == nactions indicates "stop action"

        Returns
        -------
        self.seq : list
            The sequence after executing the action

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        if action < self.nactions:
            seq_next = self.seq + list(self.action_space[action])
            if len(seq_next) > self.horizon:
                valid = False
            else:
                self.seq = seq_next
                valid = True
            self.done = len(self.seq) == self.horizon
        else:
            if len(self.seq) == 0:
                valid = False
            else:
                self.done = True
                valid = True

        return self.seq, valid

    def true_density(self, max_states=1e6):
        """
        Computes the reward density (reward / sum(rewards)) of the whole space, if the
        dimensionality is smaller than specified in the arguments.

        Returns
        -------
        Tuple:
          - normalized reward for each state
          - states
          - (un-normalized) reward)
        """
        if self._true_density is not None:
            return self._true_density
        if self.nalphabet ** self.horizon > max_states:
            return (None, None, None)
        seq_all = np.int32(
            list(itertools.product(*[list(range(self.nalphabet))] * self.horizon))
        )
        traj_rewards, seq_end = zip(
            *[
                (self.proxy(seq), seq)
                for seq in seq_all
                if len(self.parent_transitions(seq, 0)[0]) > 0 or sum(seq) == 0
            ]
        )
        traj_rewards = np.array(traj_rewards)
        self._true_density = (
            traj_rewards / traj_rewards.sum(),
            list(map(tuple, seq_end)),
            traj_rewards,
        )
        return self._true_density


def make_mlp(layers_dim, act=nn.LeakyReLU(), tail=[]):
    """
    Defines an MLP with no top layer activation

    Args
    ----
    layers_dim : list
        Dimensionality of each layer

    act : Activation
        Activation function
    """
    return nn.Sequential(
        *(
            sum(
                [
                    [nn.Linear(idim, odim)] + ([act] if n < len(layers_dim) - 2 else [])
                    for n, (idim, odim) in enumerate(zip(layers_dim, layers_dim[1:]))
                ],
                [],
            )
            + tail
        )
    )


class GFlowNetAgent:
    def __init__(self, args, proxy=None):
        # Misc
        self.rng = np.random.RandomState(int(time.time()))
        self.debug = args.debug
        self.device_torch = torch.device(args.gflownet.device)
        self.device = self.device_torch
        set_device(self.device_torch)
        self.lightweight = True
        self.tau = args.gflownet.bootstrap_tau
        self.ema_alpha = 0.5
        self.early_stopping = 0.05
        self.reward_beta = args.gflownet.reward_beta_init
        self.reward_beta_mult = args.gflownet.reward_beta_mult
        self.reward_beta_period = args.gflownet.reward_beta_period
        if self.reward_beta_period in [None, -1]:
            self.reward_beta_period = np.inf
        self.reward_max = args.gflownet.reward_max
        # Comet
        if args.gflownet.comet.project:
            self.comet = Experiment(
                project_name=args.gflownet.comet.project, display_summary_level=0
            )
            if args.gflownet.comet.tags:
                if isinstance(args.gflownet.comet.tags, list):
                    self.comet.add_tags(args.gflownet.comet.tags)
                else:
                    self.comet.add_tag(args.gflownet.comet.tags)
            self.comet.log_parameters(vars(args))
            if "workdir" in args and Path(args.workdir).exists():
                with open(Path(args.workdir) / "comet.url", "w") as f:
                    f.write(self.comet.url + "\n")
        else:
            self.comet = None
        # Environment
        self.env = AptamerSeq(
            args.gflownet.horizon,
            args.gflownet.nalphabet,
            args.gflownet.min_word_len,
            args.gflownet.max_word_len,
            func=args.gflownet.func,
            proxy=proxy,
            allow_backward=False,
            debug=self.debug,
            reward_beta=self.reward_beta,
        )
        self.envs = [
            AptamerSeq(
                args.gflownet.horizon,
                args.gflownet.nalphabet,
                args.gflownet.min_word_len,
                args.gflownet.max_word_len,
                func=args.gflownet.func,
                proxy=proxy,
                allow_backward=False,
                debug=self.debug,
                reward_beta=self.reward_beta,
            )
            for _ in range(args.gflownet.mbsize)
        ]
        self.batch_reward = args.gflownet.batch_reward
        # Model
        self.model = make_mlp(
            [args.gflownet.horizon * args.gflownet.nalphabet]
            + [args.gflownet.n_hid] * args.gflownet.n_layers
            + [self.env.nactions + 1]
        )
        if args.gflownet.model_ckpt:
            if "workdir" in args and Path(args.workdir).exists():
                if (Path(args.workdir) / "ckpts").exists():
                    self.model_path = (
                        Path(args.workdir) / "ckpts" / args.gflownet.model_ckpt
                    )
                else:
                    self.model_path = Path(args.workdir) / args.gflownet.model_ckpt
            else:
                self.model_path = args.gflownet.model_ckpt
            if self.model_path.exists():
                self.model.load_state_dict(torch.load(self.model_path))
        else:
            self.model_path = None
        self.ckpt_period = args.gflownet.ckpt_period
        if self.ckpt_period in [None, -1]:
            self.ckpt_period = np.inf
        self.model.to(self.device_torch)
        self.target = copy.deepcopy(self.model)
        # Training
        self.opt = make_opt(self.parameters(), args)
        self.n_train_steps = args.gflownet.n_iter
        self.mbsize = args.gflownet.mbsize
        self.progress = args.gflownet.progress
        self.clip_grad_norm = args.gflownet.clip_grad_norm
        self.num_empirical_loss = args.gflownet.num_empirical_loss
        self.ttsr = max(int(args.gflownet.train_to_sample_ratio), 1)
        self.sttr = max(int(1 / args.gflownet.train_to_sample_ratio), 1)
        self.random_action_prob = args.gflownet.random_action_prob

    def parameters(self):
        return self.model.parameters()

    def sample_many(self):
        """
        Builds a mini-batch of data

        Each item in the batch is a list of 5 elements:
            - all parents of the state
            - actions that lead to the state from each parent
            - reward of the state
            - the state, as seq2obs(seq)
            - done

        Args
        ----
        mbsize : int
            Mini-batch size
        """
        times = {
            "all": 0.0,
            "actions_model": 0.0,
            "actions_envs": 0.0,
            "rewards": 0.0,
        }
        t0_all = time.time()
        batch = []
        envs = [env.reset() for env in self.envs]
        while envs:
            seqs = [env.seq2obs() for env in envs]
            random_action = self.rng.uniform()
            if random_action > self.random_action_prob:
                with torch.no_grad():
                    t0_a_model = time.time()
                    action_probs = self.model(tf(seqs))
                    t1_a_model = time.time()
                    times["actions_model"] += t1_a_model - t0_a_model
                    if all(torch.isfinite(action_probs).flatten()):
                        actions = Categorical(logits=action_probs).sample()
                    else:
                        random_action = -1
                        if self.debug:
                            print("Action could not be sampled from model!")
            if random_action < self.random_action_prob:
                actions = np.random.randint(
                    low=0, high=action_probs.shape[1], size=action_probs.shape[0]
                )
            t0_a_envs = time.time()
            assert len(envs) == actions.shape[0]
            for env, action in zip(envs, actions):
                seq, valid = env.step(action)
                if valid:
                    parents, parents_a = env.parent_transitions(seq, action)
                    batch.append(
                        [
                            tf(parents),
                            tf(parents_a),
                            seq,
                            tf([env.seq2obs()]),
                            env.done,
                        ]
                    )
            envs = [env for env in envs if not env.done]
            t1_a_envs = time.time()
            times["actions_envs"] += t1_a_envs - t0_a_envs
        parents, parents_a, seqs, obs, done = zip(*batch)
        t0_rewards = time.time()
        rewards = env.reward_batch(seqs, done)
        t1_rewards = time.time()
        times["rewards"] += t1_rewards - t0_rewards
        rewards = [tf([r]) for r in rewards]
        done = [tf([d]) for d in done]
        batch = list(zip(parents, parents_a, rewards, obs, done))
        t1_all = time.time()
        times["all"] += t1_all - t0_all
        return batch, times

    def learn_from(self, it, batch):
        """
        Computes the loss of a batch

        Args
        ----
        it : int
            Iteration

        batch : ndarray
            A batch of data: every row is a state (list), corresponding to all states
            visited in each sequence in the batch.

        Returns
        -------
        loss : float
            Loss, as per Equation 12 of https://arxiv.org/abs/2106.04399v1

        term_loss : float
            Loss of the terminal nodes only

        flow_loss : float
            Loss of the intermediate nodes only
        """
        loginf = tf([1000])
        batch_idxs = tl(
            sum(
                [[i] * len(parents) for i, (parents, _, _, _, _) in enumerate(batch)],
                [],
            )
        )
        parents, actions, r, sp, done = map(torch.cat, zip(*batch))
        if self.debug and torch.any(r < 0):
            neg_r_idx = torch.where(r < 0)[0].tolist()
            for idx in neg_r_idx:
                obs = sp[idx].tolist()
                seq = list(self.env.obs2seq(seq))
                seq_oracle = self.env.seq2oracle([seq])
                output_proxy = self.env.proxy(seq_oracle)
                reward = self.env.proxy2reward(output_proxy)
                print(idx, output_proxy, reward)
                import ipdb

                ipdb.set_trace()
        parents_Qsa = self.model(parents)[
            torch.arange(parents.shape[0]), actions.long()
        ]

        if self.device.type == "cuda":
            in_flow = torch.log(
                torch.zeros((sp.shape[0],))
                .cuda()
                .index_add_(0, batch_idxs, torch.exp(parents_Qsa))
            )
        else:
            in_flow = torch.log(
                torch.zeros((sp.shape[0],)).index_add_(
                    0, batch_idxs, torch.exp(parents_Qsa)
                )
            )
        if self.tau > 0:
            with torch.no_grad():
                next_q = self.target(sp)
        else:
            next_q = self.model(sp)
        next_qd = next_q * (1 - done).unsqueeze(1) + done.unsqueeze(1) * (-loginf)
        out_flow = torch.logsumexp(torch.cat([torch.log(r)[:, None], next_qd], 1), 1)
        loss = (in_flow - out_flow).pow(2).mean()

        with torch.no_grad():
            term_loss = ((in_flow - out_flow) * done).pow(2).sum() / (
                done.sum() + 1e-20
            )
            flow_loss = ((in_flow - out_flow) * (1 - done)).pow(2).sum() / (
                (1 - done).sum() + 1e-20
            )

        if self.tau > 0:
            for a, b in zip(self.model.parameters(), self.target.parameters()):
                b.data.mul_(1 - self.tau).add_(self.tau * a)

        return loss, term_loss, flow_loss

    def train(self):

        # Metrics
        all_losses = []
        all_visited = []
        empirical_distrib_losses = []
        loss_term_ema = None
        loss_flow_ema = None

        # Train loop
        for i in tqdm(range(self.n_train_steps + 1)):  # , disable=not self.progress):
            t0_iter = time.time()
            data = []
            for j in range(self.sttr):
                batch, times = self.sample_many()
                data += batch
            rewards = [d[2][0].item() for d in data if bool(d[4].item())]
            proxy_vals = self.env.reward2proxy(rewards)
            for j in range(self.ttsr):
                losses = self.learn_from(
                    i * self.ttsr + j, data
                )  # returns (opt loss, *metrics)
                if (
                    not all([torch.isfinite(loss) for loss in losses])
                    or np.max(rewards) > self.reward_max
                ):
                    if self.debug:
                        print(
                            "Too large rewards: Skipping backward pass, increasing "
                            "reward temperature from -{:.4f} to -{:.4f} and cancelling "
                            "beta scheduling".format(
                                self.reward_beta,
                                self.reward_beta / self.reward_beta_mult,
                            )
                        )
                    self.reward_beta /= self.reward_beta_mult
                    self.reward_beta_period = np.inf
                    for env in [self.env] + self.envs:
                        env.reward_beta = self.reward_beta
                    all_losses.append([loss for loss in all_losses[-1]])
                else:
                    losses[0].backward()
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.parameters(), self.clip_grad_norm
                        )
                    self.opt.step()
                    self.opt.zero_grad()
                    all_losses.append([i.item() for i in losses])
            # Reward beta scaling
            if not i % self.reward_beta_period and i > 0:
                if self.debug:
                    print(
                        "\tDecreasing reward temperature from -{:.4f} to -{:.4f}".format(
                            self.reward_beta, self.reward_beta * self.reward_beta_mult
                        )
                    )
                self.reward_beta *= self.reward_beta_mult
                for env in [self.env] + self.envs:
                    env.reward_beta = self.reward_beta
            # Log
            seqs_batch = [
                tuple(self.env.obs2seq(d[3][0].tolist()))
                for d in data
                if bool(d[4].item())
            ]
            idx_best = np.argmax(rewards)
            seq_best = "".join(self.env.seq2letters(seqs_batch[idx_best]))
            if self.lightweight:
                all_losses = all_losses[-100:]
                all_visited = seqs_batch

            else:
                all_visited.extend(seqs_batch)
            if self.comet:
                self.comet.log_text(
                    seq_best + " / proxy: {}".format(proxy_vals[idx_best]), step=i
                )
                self.comet.log_metrics(
                    dict(
                        zip(
                            [
                                "mean_reward",
                                "max_reward",
                                "mean_proxy",
                                "min_proxy",
                                "max_proxy",
                                "mean_seq_length",
                                "batch_size",
                                "reward_beta",
                            ],
                            [
                                np.mean(rewards),
                                np.max(rewards),
                                np.mean(proxy_vals),
                                np.min(proxy_vals),
                                np.max(proxy_vals),
                                np.mean([len(seq) for seq in seqs_batch]),
                                len(data),
                                self.reward_beta,
                            ],
                        )
                    ),
                    step=i,
                )
            if not i % 100:
                if not self.lightweight:
                    empirical_distrib_losses.append(
                        compute_empirical_distribution_error(
                            self.env, all_visited[-self.num_empirical_loss :]
                        )
                    )
                else:
                    empirical_distrib_losses.append((None, None))
                if self.progress:
                    k1, kl = empirical_distrib_losses[-1]
                    if self.debug:
                        print("Empirical L1 distance", k1, "KL", kl)
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
                                ["loss", "term_loss", "flow_loss"],
                                [loss.item() for loss in losses],
                            )
                        ),
                        step=i,
                    )
                    if not self.lightweight:
                        self.comet.log_metric(
                            "unique_states", np.unique(all_visited).shape[0], step=i
                        )
                # Save intermediate model
            if not i % self.ckpt_period and self.model_path:
                path = self.model_path.parent / Path(
                    self.model_path.stem
                    + "_iter{:06d}".format(i)
                    + self.model_path.suffix
                )
                torch.save(self.model.state_dict(), path)
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
            times = {"time_{}".format(k): v for k, v in times.items()}
            if self.comet:
                self.comet.log_metrics(times, step=i)
        # Save final model
        if self.model_path:
            path = self.model_path.parent / Path(
                self.model_path.stem + "_final" + self.model_path.suffix
            )
            torch.save(self.model.state_dict(), path)
            torch.save(self.model.state_dict(), self.model_path)

        # Close comet
        if self.comet:
            self.comet.end()

    def sample(self, n_samples, horizon, nalphabet, min_word_len, max_word_len, proxy):
        times = {
            "all": 0.0,
            "actions_model": 0.0,
            "actions_envs": 0.0,
            "proxy": 0.0,
            "sanitycheck": 0.0,
        }
        t0_all = time.time()
        batch = []
        envs = [
            AptamerSeq(horizon, nalphabet, min_word_len, max_word_len, proxy=proxy)
            for i in range(n_samples)
        ]
        envs = [env.reset() for env in envs]
        while envs:
            seqs = [env.seq2obs() for env in envs]
            with torch.no_grad():
                t0_a_model = time.time()
                action_probs = self.model(tf(seqs))
                t1_a_model = time.time()
                times["actions_model"] += t1_a_model - t0_a_model
                if all(torch.isfinite(action_probs).flatten()):
                    actions = Categorical(logits=action_probs).sample()
                else:
                    actions = np.random.randint(
                        low=0, high=action_probs.shape[1], size=action_probs.shape[0]
                    )
                    if self.debug:
                        print("Action could not be sampled from model!")
            t0_a_envs = time.time()
            assert len(envs) == actions.shape[0]
            for env, action in zip(envs, actions):
                seq, valid = env.step(action)
                if valid and env.done:
                    batch.append(env.seq2oracle([seq])[0])
            envs = [env for env in envs if not env.done]
            t1_a_envs = time.time()
            times["actions_envs"] += t1_a_envs - t0_a_envs
        t0_proxy = time.time()
        batch = np.asarray(batch)
        proxy_vals, uncertainties = env.proxy(batch, "Both")
        t1_proxy = time.time()
        times["proxy"] += t1_proxy - t0_proxy
        samples = {
            "samples": batch.astype(np.int64),
            "scores": proxy_vals,
            "energies": proxy_vals,
            "uncertainties": uncertainties,
        }
        # Sanity-check: absolute zero pad
        t0_sanitycheck = time.time()
        zeros = np.where(batch == 0)
        row_unique, row_unique_idx = np.unique(zeros[0], return_index=True)
        for row, idx in zip(row_unique, row_unique_idx):
            if np.sum(batch[row, zeros[1][idx] :]):
                print(f"Found sequence with positive values after last 0, row {row}")
                import ipdb

                ipdb.set_trace()
        t1_sanitycheck = time.time()
        times["sanitycheck"] += t1_sanitycheck - t0_sanitycheck
        t1_all = time.time()
        times["all"] += t1_all - t0_all
        return samples, times


class RandomTrajAgent:
    def __init__(self, args, envs):
        self.mbsize = args.gflownet.mbsize  # mini-batch size
        self.envs = envs
        self.nact = args.ndim + 1
        self.model = None

    def parameters(self):
        return []

    def sample_many(self, mbsize, all_visited):
        batch = []
        [i.reset()[0] for i in self.envs]  # reset envs
        done = [False] * mbsize
        while not all(done):
            acts = np.random.randint(0, self.nact, mbsize)  # actions (?)
            # step : list
            # - For each e in envs, if corresponding done is False
            #   - For each element i in env, and a in acts
            #     - i.step(a)
            step = [
                i.step(a)
                for i, a in zip([e for d, e in zip(done, self.envs) if not d], acts)
            ]
            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            for (_, r, d, sp) in step:
                if d:
                    all_visited.append(tuple(sp))
        return []  # agent is stateful, no need to return minibatch data

    def learn_from(self, it, batch):
        return None


def make_opt(params, args):
    """
    Set up the optimizer
    """
    params = list(params)
    if not len(params):
        return None
    if args.gflownet.opt == "adam":
        opt = torch.optim.Adam(
            params,
            args.gflownet.learning_rate,
            betas=(args.gflownet.adam_beta1, args.gflownet.adam_beta2),
        )
    elif args.gflownet.opt == "msgd":
        opt = torch.optim.SGD(
            params, args.gflownet.learning_rate, momentum=args.gflownet.momentum
        )
    return opt


def compute_empirical_distribution_error(env, visited):
    """
    Computes the empirical distribution errors, as the mean L1 error and the KL
    divergence between the true density of the space and the estimated density from all
    states visited.
    """
    td, end_states, true_r = env.true_density()
    if td is None:
        return None, None
    true_density = tf(td)
    if not len(visited):
        return 1, 100
    hist = defaultdict(int)
    for i in visited:
        hist[i] += 1
    Z = sum([hist[i] for i in end_states])
    estimated_density = tf([hist[i] / Z for i in end_states])
    k1 = abs(estimated_density - true_density).mean().item()
    # KL divergence
    kl = (true_density * torch.log(estimated_density / true_density)).sum().item()
    return k1, kl


def main(args):
    gflownet_agent = GFlowNetAgent(args)
    gflownet_agent.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser, args2config = add_args(parser)
    args = parser.parse_args()
    config = get_config(args, override_args, args2config)
    print("Config file: " + config.yaml_config)
    print("Working dir: " + config.workdir)
    print(
        "Config:\n"
        + "\n".join([f"    {k:20}: {v}" for k, v in vars(config.gflownet).items()])
    )
    if "workdir" in config:
        if not Path(config.workdir).exists():
            Path(config.workdir).mkdir(parents=True, exist_ok=False)
            with open(config.workdir + "/config.yml", "w") as f:
                yaml.dump(numpy2python(namespace2dict(config)), f, default_flow_style=False)
            torch.set_num_threads(1)
            main(config)
        else:
            print(f"workdir {config.workdir} already exists! - Ending run...")
    else:
        print(f"workdir not defined - Ending run...")
