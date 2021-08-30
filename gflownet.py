import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle
from collections import defaultdict
from itertools import count

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from oracles import linearToy, toyHamiltonian, PottsEnergy, seqfoldScore, nupackScore


parser = argparse.ArgumentParser()

parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--save_path", default="results/flow_insp_0.pkl.gz", type=str)
parser.add_argument("--progress", action="store_true")

#
parser.add_argument("--method", default="flownet", type=str)
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

# Environment
parser.add_argument("--func", default="arbitrary_i")
parser.add_argument(
    "--horizon",
    default=42,
    help="Maximum number of episodes; maximum sequence length",
    type=int,
)
parser.add_argument("--nalphabet", default=4, type=int)

# MCMC
parser.add_argument("--bufsize", default=16, help="MCMC buffer size", type=int)

# Flownet
parser.add_argument("--bootstrap_tau", default=0.0, type=float)
parser.add_argument("--replay_strategy", default="none", type=str)  # top_k none
parser.add_argument("--replay_sample_size", default=2, type=int)
parser.add_argument("--replay_buf_size", default=100, type=float)

# PPO
parser.add_argument("--clip_grad_norm", default=0.0, type=float)

# Comet tags
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

    func : lambda
        Reward function [to be confirmed]
    """

    def __init__(self, horizon=42, nalphabet=4, func=None, allow_backward=False):
        self.horizon = horizon
        self.nalphabet = nalphabet
        self.seq = []
        self.done = False
        self.func = {
            "default": None,
            "arbitrary_i": self.reward_arbitrary_i,
            "linear": linearToy,
            "innerprod": toyHamiltonian,
            "potts": PottsEnergy,
            "seqfold": seqfoldScore,
            "nupack": nupackScore,
        }[func]
        self.reward = lambda x: 0 if not self.done else self.energy2reward(self.func(self.seq2oracle(x)))
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
        """
        queries = np.array(seq)
        if queries.ndim == 1:
            queries = queries[np.newaxis, ...]
        queries += 1
        if queries.shape[1] == 1:
            queries = np.column_stack((queries, np.zeros(queries.shape[0])))
        return queries

    def energy2reward(self, energies, epsilon=1e-9):
        """
        Prepares the output of an oracle for GFlowNet.
        """
        if self.func == seqfoldScore:
            energies -= 5
            energies *= -1
        if self.func == nupackScore:
            energies *= -1
        reward = energies + epsilon
        return reward[0]

    def reward2energy(self, reward, epsilon=1e-9):
        """
        Converts a "GFlowNet reward" into energy as returned by an oracle.
        """
        energy = reward - epsilon
        if self.func == seqfoldScore:
            energy *= -1
            energy += 5
        if self.func == nupackScore:
            energy *= -1
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
            z[np.arange(len(seq)) * self.nalphabet + seq] = 1
        return z

    def obs2seq(self, obs):
        """
        Transforms the one-hot encoding version of a sequence (state) given as argument into a
        a sequence of letter indices.

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
                (self.func(seq), seq)
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


class ReplayBuffer:
    def __init__(self, args, env):
        self.buf = []
        self.strat = args.replay_strategy
        self.sample_size = args.replay_sample_size
        self.bufsize = args.replay_buf_size
        self.env = env

    def add(self, x, r_x):
        if self.strat == "top_k":
            if len(self.buf) < self.bufsize or r_x > self.buf[0][0]:
                self.buf = sorted(self.buf + [(r_x, x)])[-self.bufsize :]

    def sample(self):
        if not len(self.buf):
            return []
        idxs = np.random.randint(0, len(self.buf), self.sample_size)
        return sum([self.generate_backward(*self.buf[i]) for i in idxs], [])

    def generate_backward(self, r, s0):
        s = np.int8(s0)
        os0 = self.env.obs(s)
        # If s0 is a forced-terminal state, the the action that leads
        # to it is s0.argmax() which .parents finds, but if it isn't,
        # we must indicate that the agent ended the trajectory with
        # the stop action
        used_stop_action = s.max() < self.env.horizon - 1
        done = True
        # Now we work backward from that last transition
        traj = []
        while s.sum() > 0:
            parents, actions = self.env.parent_transitions(s, used_stop_action)
            # add the transition
            traj.append(
                [tf(i) for i in (parents, actions, [r], [self.env.obs(s)], [done])]
            )
            # Then randomly choose a parent state
            if not used_stop_action:
                i = np.random.randint(0, len(parents))
                a = actions[i]
                s[a] -= 1
            # Values for intermediary trajectory states:
            used_stop_action = False
            done = False
            r = 0
        return traj


class GFlowNetAgent:
    def __init__(self, args):
        # Misc
        args.device_torch = torch.device(args.device)
        set_device(args.device_torch)
        self.save_path = args.save_path
        # Model
        self.model = make_mlp(
            [args.horizon * args.nalphabet]
            + [args.n_hid] * args.n_layers
            + [args.nalphabet + 1]
        )
        self.model.to(args.device_torch)
        self.target = copy.deepcopy(self.model)
        self.tau = args.bootstrap_tau
        # Comet
        self.comet = Experiment(
            project_name=args.comet_project, display_summary_level=0
        )
        if args.tags:
            self.comet.add_tags(args.tags)
        self.comet.log_parameters(vars(args))
        # Environment
        args.is_mcmc = args.method in ["mars", "mcmc"]
        self.env = AptamerSeq(
            args.horizon, args.nalphabet, func=args.func, allow_backward=args.is_mcmc
        )
        self.envs = [
            AptamerSeq(
                args.horizon, args.nalphabet, func=args.func, allow_backward=args.is_mcmc
            )
            for _ in range(args.mbsize)
        ]
        self.replay = ReplayBuffer(args, self.envs[0])
        # Training
        self.opt = make_opt(self.parameters(), args)
        self.n_train_steps = args.n_train_steps
        self.mbsize = args.mbsize
        self.progress = args.progress
        self.clip_grad_norm = args.clip_grad_norm
        self.num_empirical_loss = args.num_empirical_loss
        self.ttsr = max(int(args.train_to_sample_ratio), 1)
        self.sttr = max(int(1 / args.train_to_sample_ratio), 1)

    def parameters(self):
        return self.model.parameters()

    def sample_many(self, mbsize):
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
                if valid:
                    parents, parents_a = env.parent_transitions(seq, action)
                    batch.append(
                        [
                            tf(parents),
                            tf(parents_a),
                            tf([env.reward(seq)]),
                            tf([env.seq2obs()]),
                            tf([env.done]),
                        ]
                    )
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


    def train():

        # Metrics
        all_losses = []
        all_visited = []
        empirical_distrib_losses = []

        # Train loop
        for i in tqdm(range(self.n_train_steps + 1), disable=not self.progress):
            data = []
            for j in range(self.sttr):
                data += self.sample_many(self.mbsize)
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
                    opt.step()
                    opt.zero_grad()
                    all_losses.append([i.item() for i in losses])
            all_visited.extend(
                [tuple(env.obs2seq(d[3][0].tolist())) for d in data if bool(d[4].item())]
            )
            rewards = [d[2][0].item() for d in data if bool(d[4].item())]
            self.comet.log_metric('mean_reward', np.mean(rewards), step=i)
            self.comet.log_metric('max_reward', np.max(rewards), step=i)

            if not i % 100:
                empirical_distrib_losses.append(
                    compute_empirical_distribution_error(
                        env, all_visited[-args.num_empirical_loss :]
                    )
                )
                if self.progress:
                    k1, kl = empirical_distrib_losses[-1]
                    print("Empirical L1 distance", k1, "KL", kl)
                    if len(all_losses):
                        print(
                            *[
                                f"{np.mean([i[j] for i in all_losses[-100:]]):.5f}"
                                for j in range(len(all_losses[0]))
                            ]
                        )
                self.comet.log_metrics(dict(zip(['loss', 'term_loss', 'flow_loss'], [loss.item() for loss in losses])), step=i)
                self.comet.log_metric('unique_states', np.unique(all_visited).shape[0], step=i)

        # Save model and training variables
        root = os.path.split(self.save_path)[0]
        os.makedirs(root, exist_ok=True)
        pickle.dump(
            {
                "losses": np.float32(all_losses),
                #'model': self.model.to('cpu') if self.model else None,
                "params": [i.data.to("cpu").numpy() for i in self.parameters()],
                "visited": [np.int8(seq) for seq in all_visited],
                "emp_dist_loss": empirical_distrib_losses,
                "true_d": env.true_density()[0],
            },
            gzip.open(self.save_path, "wb"),
        )
        torch.save(self.model.state_dict(), self.save_path.replace("pkl.gz", "pt"))

        # Close comet
        self.comet.end()


class RandomTrajAgent:
    def __init__(self, args, envs):
        self.mbsize = args.mbsize  # mini-batch size
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
    if args.opt == "adam":
        opt = torch.optim.Adam(
            params, args.learning_rate, betas=(args.adam_beta1, args.adam_beta2)
        )
    elif args.opt == "msgd":
        opt = torch.optim.SGD(params, args.learning_rate, momentum=args.momentum)
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
    gflownet_agent = GFlowNet(args)
    gflownet_agent.train()


if __name__ == "__main__":
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
