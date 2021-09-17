from comet_ml import Experiment
import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle
from collections import defaultdict
from itertools import count
from pathlib import Path

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from oracles import linearToy, toyHamiltonian, PottsEnergy, seqfoldScore, nupackScore


parser = argparse.ArgumentParser()

parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--progress", action="store_true")
parser.add_argument("--model_ckpt", default=None, type=str)

#
parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
parser.add_argument("--opt", default="adam", type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
parser.add_argument("--train_to_sample_ratio", default=1, type=float)
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=20000, type=int)
parser.add_argument(
    "--num_empirical_loss",
    default=200000,
    type=int,
    help="Number of samples used to compute the empirical distribution loss",
)
parser.add_argument("--clip_grad_norm", default=0.0, type=float)

# Environment
parser.add_argument("--func", default="arbitrary_i")
parser.add_argument(
    "--horizon",
    default=42,
    help="Maximum number of episodes; maximum sequence length",
    type=int,
)
parser.add_argument("--nalphabet", default=4, type=int)

# Flownet
parser.add_argument("--bootstrap_tau", default=0.0, type=float)
parser.add_argument('--batch_reward', type=bool, default=False)

# Comet
parser.add_argument("--comet_project", default=None, type=str)
parser.add_argument(
    "-t", "--tags", nargs="*", help="Comet.ml tags", default=[], type=str
)

# Float and Long tensors
_dev = [torch.device("cpu")]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])


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
        self, horizon=42, nalphabet=4, func="default", proxy=None, allow_backward=False
    ):
        self.horizon = horizon
        self.nalphabet = nalphabet
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
                "nupack": nupackScore,
            }[self.func]
        self.reward = (
            lambda x: [0]
            if not self.done
            else self.energy2reward(self.proxy(self.seq2oracle(x)))
        )
        self.allow_backward = allow_backward
        self._true_density = None

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
            import ipdb; ipdb.set_trace()
            queries = np.column_stack((queries, np.zeros(queries.shape[0])))
        return queries

    def reward_batch(self, seq, done):
        seq = [s for s, d in zip(seq, done) if d]
        reward = np.zeros(len(done))
        reward[list(done)] = self.energy2reward(self.proxy(self.seq2oracle(seq)))
        return reward

    def energy2reward(self, energies, epsilon=1e-9):
        """
        Prepares the output of an oracle for GFlowNet.
        """
        if self.func == "potts":
            energies *= -1
            energies = np.clip(energies, a_min=0.0, a_max=None)
        elif self.func == "seqfold":
            energies -= 5
            energies *= -1
        elif self.func == "nupack":
            energies *= -1
        else:
            pass
        rewards = energies + epsilon
        return rewards

    def reward2energy(self, reward, epsilon=1e-9):
        """
        Converts a "GFlowNet reward" into energy as returned by an oracle.
        """
        energy = reward - epsilon
        if self.func == "potts":
            energy *= -1
        elif self.func == "seqfold":
            energy *= -1
            energy += 5
        elif self.func == "nupack":
            energy *= -1
        else:
            pass
        return energy

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
            if hasattr(seq[0],'device'): # if it has a device at all, it will be cuda (CPU numpy array has no dev
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
        if action == self.nalphabet:
            return [self.seq2obs(seq)], [action]
        else:
            parents = [self.seq2obs(seq[:-1])]
            actions = [action]
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

        If action is smaller than nalphabet (no stop), add action letter to next
        position.

        See: step_daug()
        See: step_chain()

        Args
        ----
        a : int
            Index of letter in the alphabet. a == nalphabet indicates "stop action"

        Returns
        -------
        self.seq : list
            The sequence after executing the action

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        if action < self.nalphabet:
            self.seq = self.seq + [action]
            self.done = len(self.seq) == self.horizon
            valid = True
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
        args.device_torch = torch.device(args.device)
        self.device = args.device_torch
        set_device(args.device_torch)
        # Model
        self.model = make_mlp(
            [args.horizon * args.nalphabet]
            + [args.gflownet.n_hid] * args.gflownet.n_layers
            + [args.nalphabet + 1]
        )
        if args.gflownet.model_ckpt and "workdir" in args:
            if "workdir" in args:
                self.model_path = Path(args.workdir) / "ckpts" / args.gflownet.model_ckpt
            else:
                self.model_path = args.gflownet.model_ckpt
            if self.model_path.exists():
                self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(args.device_torch)
        self.target = copy.deepcopy(self.model)
        self.tau = args.gflownet.bootstrap_tau
        self.ema_alpha = 0.5
        self.early_stopping = 0.05
        # Comet
        if args.gflownet.comet.project:
            self.comet = Experiment(
                project_name=args.gflownet.comet.project, display_summary_level=0
            )
            if args.tags:
                self.comet.add_tags(args.tags)
            self.comet.log_parameters(vars(args))
        else:
            self.comet = None
        # Environment
        self.env = AptamerSeq(
            args.horizon,
            args.nalphabet,
            func=args.func,
            proxy=proxy,
            allow_backward=False,
        )
        self.envs = [
            AptamerSeq(
                args.horizon,
                args.nalphabet,
                func=args.func,
                proxy=proxy,
                allow_backward=False,
            )
            for _ in range(args.gflownet.mbsize)
        ]
        self.batch_reward = args.gflownet.batch_reward
        # Training
        self.opt = make_opt(self.parameters(), args)
        self.n_train_steps = args.gflownet.n_iter
        self.mbsize = args.gflownet.mbsize
        self.gflownet.progress = args.gflownet.progress
        self.clip_grad_norm = args.gflownet.clip_grad_norm
        self.num_empirical_loss = args.gflownet.num_empirical_loss
        self.ttsr = max(int(args.gflownet.train_to_sample_ratio), 1)
        self.sttr = max(int(1 / args.gflownet.train_to_sample_ratio), 1)

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
        batch = []
        for env in self.envs:
            env = env.reset()
            while not env.done:
                with torch.no_grad():
                    action_probs = self.model(tf(env.seq2obs()))
                    if all(torch.isfinite(action_probs)):
                        action = Categorical(logits=action_probs).sample()
                    else:
                        action = np.random.permutation(np.arange(len(action_probs)))[0]
                        print("Action could not be sampled from model!")
                seq, valid = env.step(action)
                if len(seq) > 0:
                    if hasattr(seq[0], 'device'): # if it has a device, it's on cuda
                        seq = [subseq.cpu().detach().numpy() for subseq in seq]
                if valid:
                    parents, parents_a = env.parent_transitions(seq, action)
                    if self.batch_reward:
                        batch.append(
                            [
                                tf(parents),
                                tf(parents_a),
                                seq,
                                tf([env.seq2obs()]),
                                env.done,
                            ]
                        )
                    else:
                        batch.append(
                            [
                                tf(parents),
                                tf(parents_a),
                                tf([env.reward([seq])[0]]),
                                tf([env.seq2obs()]),
                                tf([env.done]),
                            ]
                        )
        if self.batch_reward:
            parents, parents_a, seq, obs, done = zip(*batch)
            rewards = env.reward_batch(seq, done)
            rewards = [tf([r]) for r in rewards]
            done = [tf([d]) for d in done]
            batch = list(zip(parents, parents_a, rewards, obs, done))
        return batch

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
        parents_Qsa = self.model(parents)[
            torch.arange(parents.shape[0]), actions.long()
        ]

        if self.device.type == 'cuda':
            in_flow = torch.log(
                torch.zeros((sp.shape[0],)).cuda().index_add_(
                    0, batch_idxs, torch.exp(parents_Qsa)
                )
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
        loss_ema = -1.0

        # Train loop
        for i in tqdm(range(self.n_train_steps + 1)):#, disable=not self.gflownet.progress):
            data = []
            for j in range(self.sttr):
                data += self.sample_many()
            for j in range(self.ttsr):
                losses = self.learn_from(
                    i * self.ttsr + j, data
                )  # returns (opt loss, *metrics)
                if losses is not None:
                    losses[0].backward()
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.parameters(), self.clip_grad_norm
                        )
                    self.opt.step()
                    self.opt.zero_grad()
                    all_losses.append([i.item() for i in losses])
            all_visited.extend(
                [
                    tuple(self.env.obs2seq(d[3][0].tolist()))
                    for d in data
                    if bool(d[4].item())
                ]
            )
            # Log
            rewards = [d[2][0].item() for d in data if bool(d[4].item())]
            if self.comet:
                self.comet.log_metric("mean_reward", np.mean(rewards), step=i)
                self.comet.log_metric("max_reward", np.max(rewards), step=i)
            if not i % 100:
                empirical_distrib_losses.append(
                    compute_empirical_distribution_error(
                        self.env, all_visited[-self.num_empirical_loss :]
                    )
                )
                if self.gflownet.progress:
                    k1, kl = empirical_distrib_losses[-1]
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
                    self.comet.log_metric(
                        "unique_states", np.unique(all_visited).shape[0], step=i
                    )
            # Moving average of the loss for early stopping
            if loss_ema > 0:
                loss_ema = (
                    self.ema_alpha * losses[0] + (1.0 - self.ema_alpha) * loss_ema
                )
                if loss_ema < self.early_stopping:
                    break
            else:
                loss_ema = losses[0]

        # Save model
        if self.model_path:
            torch.save(self.model.state_dict(), self.model_path)

        # Close comet
        if self.comet:
            self.comet.end()

    def sample(self, n_samples, horizon, nalphabet, proxy):
        envs = [
            AptamerSeq(
                horizon, nalphabet, proxy=proxy
            )
            for i in range(n_samples)
        ]

        batch = np.zeros((n_samples, horizon))
        for idx, env in enumerate(envs):
            env = env.reset()
            while not env.done:
                with torch.no_grad():
                    action_probs = self.model(tf(env.seq2obs()))
                    if all(torch.isfinite(action_probs)):
                        action = Categorical(logits=action_probs).sample()
                    else:
                        action = np.random.permutation(np.arange(len(action_probs)))[0]
                        print("Action could not be sampled from model!")
                seq, valid = env.step(action)

            seq = [s.item() for s in seq]
            batch[idx, :] = env.seq2oracle([seq])
        energies, uncertainties = env.proxy(batch, 'Both')
        samples = {
                'samples': batch.astype(np.int64),
                'scores': energies,
                'energies': energies,
                'uncertainties': uncertainties,
        }
        # Sanity-check 
        import ipdb; ipdb.set_trace()
        return samples



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
            params, args.gflownet.learning_rate, betas=(args.gflownet.adam_beta1, args.gflownet.adam_beta2)
        )
    elif args.gflownet.opt == "msgd":
        opt = torch.optim.SGD(params, args.gflownet.learning_rate, momentum=args.gflownet.momentum)
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
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
