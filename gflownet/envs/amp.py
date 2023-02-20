"""
Classes to represent aptamers environments
"""
from typing import List, Tuple
import itertools
import numpy as np
import pandas as pd
from gflownet.envs.base import GFlowNetEnv
import time
from sklearn.model_selection import GroupKFold, train_test_split
import itertools
from clamp_common_eval.defaults import get_default_data_splits
from polyleven import levenshtein
import numpy.typing as npt
from torchtyping import TensorType
import torch
import matplotlib.pyplot as plt


class AMP(GFlowNetEnv):
    """
    Anti-microbial peptide  sequence environment

    Attributes
    ----------
    max_seq_length : int
        Maximum length of the sequences

    min_seq_length : int
        Minimum length of the sequences

    nalphabet : int
        Number of letters in the alphabet

    state : list
        Representation of a sequence (state), as a list of length max_seq_length where
        each element is the index of a letter in the alphabet, from 0 to (nalphabet -
        1).

    done : bool
        True if the sequence has reached a terminal state (maximum length, or stop
        action executed.

    func : str
        Name of the reward function

    n_actions : int
        Number of actions applied to the sequence

    proxy : lambda
        Proxy model
    """

    def __init__(
        self,
        max_seq_length=50,
        min_seq_length=1,
        n_alphabet=20,
        min_word_len=1,
        max_word_len=1,
        # env_id=None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.n_alphabet = n_alphabet
        # TODO: self.proxy_input_dim = self.n_alphabet * self.max_seq_length
        self.min_word_len = min_word_len
        self.max_word_len = max_word_len
        # TODO: eos re-initalised in ger_actions_space
        self.eos = self.n_alphabet
        self.action_space = self.get_actions_space()
        self.fixed_policy_output = self.get_fixed_policy_output()
        self.random_policy_output = self.get_fixed_policy_output()
        self.policy_output_dim = len(self.fixed_policy_output)
        self.policy_input_dim = len(self.state2policy())
        self.max_traj_len = self.get_max_traj_len()
        self.vocab = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]
        # TODO: Change depening on how it is assigned for the torus
        if self.proxy_state_format == "ohe":
            self.statebatch2proxy = self.statebatch2policy
            self.statetorch2proxy = self.statetorch2policy
        elif self.proxy_state_format == "oracle":
            self.statebatch2proxy = self.statebatch2oracle
            self.statetorch2proxy = self.statetorch2oracle
        else:
            raise ValueError(
                "Invalid proxy_state_format: {}".format(self.proxy_state_format)
            )
        self.alphabet = dict((i, a) for i, a in enumerate(self.vocab))
        self.reset()
        # because -1 is for fidelity not being chosen
        self.invalid_state_element = -2
        # self.proxy_factor = 1.0
        if self.do_state_padding:
            assert (
                self.invalid_state_element is not None
            ), "Padding value of state not defined"

    def get_actions_space(self):
        """
        Constructs list with all possible actions
        If min_word_len = n_alphabet = 2, actions: [(0, 0,), (1, 1)] and so on
        """
        assert self.max_word_len >= self.min_word_len
        valid_wordlens = np.arange(self.min_word_len, self.max_word_len + 1)
        alphabet = [a for a in range(self.n_alphabet)]
        actions = []
        for r in valid_wordlens:
            actions_r = [el for el in itertools.product(alphabet, repeat=r)]
            actions += actions_r
        # Add "eos" action
        # eos != n_alphabet in the init because it would break if max_word_len >1
        actions = actions + [(len(actions),)]
        return actions

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        Returns a vector of length the action space (where action space includes eos): True if action is invalid
        given the current state, False otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(len(self.action_space))]
        mask = [False for _ in range(len(self.action_space))]
        seq_length = len(state)
        if seq_length < self.min_seq_length:
            mask[self.eos] = True
        # Iterate till before the eos action
        # TODO: ensure it does not break in multi-fidelity
        for idx, a in enumerate(self.action_space[:-1]):
            if seq_length + len(list(a)) > self.max_seq_length:
                mask[idx] = True
        return mask

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
        if self.n_alphabet**self.max_seq_length > max_states:
            return (None, None, None)
        state_all = np.int32(
            list(
                itertools.product(*[list(range(self.n_alphabet))] * self.max_seq_length)
            )
        )
        traj_rewards, state_end = zip(
            *[
                (self.proxy(state), state)
                for state in state_all
                if len(self.get_parents(state, False)[0]) > 0 or sum(state) == 0
            ]
        )
        traj_rewards = np.array(traj_rewards)
        self._true_density = (
            traj_rewards / traj_rewards.sum(),
            list(map(tuple, state_end)),
            traj_rewards,
        )
        return self._true_density

    def state2oracle(self, state: List = None):
        return "".join(self.state2readable(state))

    def statebatch2oracle(self, state_batch: List[List]):
        """
        Prepares a sequence in "GFlowNet format" for the oracles.

        Args
        ----
        state_list : list of lists
            List of sequences.
        """
        state_oracle = [self.state2oracle(state) for state in state_batch]
        return state_oracle

    def get_max_traj_len(
        self,
    ):
        return self.max_seq_length / self.min_word_len + 1

    def statetorch2oracle(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_proxy_dim"]:
        state_oracle = []
        for state in states:
            if state[-1] == self.invalid_state_element:
                state = state[: torch.where(state == self.invalid_state_element)[0][0]]
            state_numpy = state.detach().cpu().numpy()
            state_oracle.append(self.state2oracle(state_numpy))
        return state_oracle

    def state2policy(self, state=None):
        """
        Transforms the sequence (state) given as argument (or self.state if None) into a
        one-hot encoding. The output is a list of length nalphabet * max_seq_length,
        where each n-th successive block of nalphabet elements is a one-hot encoding of
        the letter in the n-th position.

        Example:
          - Sequence: AATGC
          - state: [0, 1, 3, 2]
                    A, T, G, C
          - state2obs(state): [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                              |     A    |      T    |      G    |      C    |

        If max_seq_length > len(state), the last (max_seq_length - len(state)) blocks are all
        0s.
        """
        if state is None:
            state = self.state.copy()
        state_policy = np.zeros(self.max_seq_length * self.n_alphabet, dtype=np.float32)
        if len(state) > 0:
            # state = [subseq.cpu().detach().numpy() for subseq in state]
            state_policy[(np.arange(len(state)) * self.n_alphabet + state)] = 1
        return state_policy

    def statebatch2policy(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Transforms a batch of states into the policy model format. The output is a numpy
        array of shape [n_states, n_alphabet * max_seq_len].

        See state2policy().
        """
        # TODO: ensure that un-padded state is fed here
        # if not modify to implementation similar to that of statetorch2policy
        state_policy = np.zeros(
            (len(states), self.n_alphabet * self.max_seq_length), dtype=np.float32
        )
        for idx, state in enumerate(states):
            state_policy[idx] = self.state2policy(state)
        # TODO fix for when some states are []
        # if list(map(len, states)) != [0 for s in states]:
        # cols, lengths = zip(
        #     *[
        #         (state + np.arange(len(state)) * self.n_alphabet, len(state))
        #         if len(state) > 0
        #         else
        #         (np.array([self.max_seq_length * self.n_alphabet]), 0)
        #         for state in states
        #     ]
        # )
        # rows = np.repeat(np.arange(len(states)), lengths)
        # if rows.shape[0] > 0:
        #     state_policy[rows, np.concatenate(cols)] = 1.0
        return state_policy

    def statetorch2policy(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "policy_output_dim"]:
        # TODO: prettify and verify logic
        cols, lengths = zip(
            *[
                (
                    (
                        state[
                            : torch.where(state == self.invalid_state_element)[0][0]
                            if state[-1] == self.invalid_state_element
                            else len(state)
                        ].to(self.device)
                        + torch.arange(
                            len(
                                state[
                                    : torch.where(state == self.invalid_state_element)[
                                        0
                                    ][0]
                                    if state[-1] == self.invalid_state_element
                                    else len(state)
                                ]
                            )
                        ).to(self.device)
                        * self.n_alphabet
                    ),
                    torch.where(state == self.invalid_state_element)[0][0]
                    if state[-1] == self.invalid_state_element
                    else len(state),
                )
                for state in states
            ]
        )
        lengths = torch.Tensor(list(lengths)).to(self.device).to(torch.int64)
        cols = torch.cat(cols, dim=0).to(torch.int64).to(self.device)
        rows = torch.repeat_interleave(
            torch.arange(len(states)).to(self.device), lengths
        )
        state_policy = torch.zeros(
            (len(states), self.n_alphabet * self.max_seq_length),
            dtype=torch.float32,
            device=self.device,
        )
        state_policy[rows, cols] = 1.0
        return state_policy

    def policytorch2state(self, state_policy: List) -> List:
        """
        Transforms the one-hot encoding version of a sequence (state) given as argument
        into a a sequence of letter indices.

        Example:
          - Sequence: AATGC
          - state_policy: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                 |     A    |      A    |      T    |      G    |      C    |
          - policy2state(state_policy): [0, 0, 1, 3, 2]
                    A, A, T, G, C
        """
        mat_state_policy = torch.reshape(
            state_policy, (self.max_seq_length, self.n_alphabet)
        )
        state = torch.where(mat_state_policy)[1].tolist()
        return state

    def policy2state(self, state_policy: List) -> List:
        """
        Transforms the one-hot encoding version of a sequence (state) given as argument
        into a a sequence of letter indices.

        Example:
          - Sequence: AATGC
          - state_policy: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                 |     A    |      A    |      T    |      G    |      C    |
          - policy2state(state_policy): [0, 0, 1, 3, 2]
                    A, A, T, G, C
        """
        mat_state_policy = np.reshape(
            state_policy, (self.max_seq_length, self.n_alphabet)
        )
        state = np.where(mat_state_policy)[1].tolist()
        return state

    def state2readable(self, state: List) -> str:
        """
        Transforms a sequence given as a list of indices into a sequence of letters
        according to an alphabet.
        """
        return "".join([self.alphabet[el] for el in state])

    def statetorch2readable(self, state: TensorType["batch", "state_dim"]) -> List[str]:
        if state[-1] == self.invalid_state_element:
            state = state[: torch.where(state == self.invalid_state_element)[0][0]]
        state = state.tolist()
        readable = [self.alphabet[el] for el in state]
        return "".join(readable)

    def readable2state(self, readable: str) -> List:
        """
        Transforms a sequence given as a list of letters into a sequence of indices
        according to an alphabet.
        """
        alphabet = {v: k for k, v in self.alphabet.items()}
        return [alphabet[el] for el in readable]

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = []
        self.done = False
        self.id = env_id
        self.n_actions = 0
        return self

    def get_parents(self, state=None, done=None, action=None):
        """
        Determines all parents and actions that lead to sequence state

        Args
        ----
        state : list
            Representation of a sequence (state), as a list of length max_seq_length
            where each element is the index of a letter in the alphabet, from 0 to
            (nalphabet - 1).

        action : int
            Last action performed, only to determine if it was eos.

        Returns
        -------
        parents : list
            List of parents as state2obs(state)

        actions : list
            List of actions that lead to state for each parent in parents
        """
        # TODO: Adapt to tuple form actions
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [(self.eos,)]
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                is_parent = state[-len(a) :] == list(a)
                if not isinstance(is_parent, bool):
                    is_parent = all(is_parent)
                if is_parent:
                    parents.append(state[: -len(a)])
                    actions.append(a)
        return parents, actions

    def step(self, action: Tuple[int]) -> Tuple[List[int], Tuple[int, int], bool]:
        """
        Executes step given an action index

        If action_idx is smaller than eos (no stop), add action to next
        position.

        Args
        ----
        action_idx : int
            Index of action in the action space. a == eos indicates "stop action"

        Returns
        -------
        self.state : list
            The sequence after executing the action

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        assert action in self.action_space
        # If only possible action is eos, then force eos
        if len(self.state) == self.max_seq_length:
            self.done = True
            self.n_actions += 1
            return self.state, (self.eos,), True
        # If action is not eos, then perform action
        if action[0] != self.eos:
            state_next = self.state + list(action)
            if len(state_next) > self.max_seq_length:
                valid = False
            else:
                self.state = state_next
                valid = True
                self.n_actions += 1
            return self.state, action, valid
        # If action is eos, then perform eos
        else:
            if len(self.state) < self.min_seq_length:
                valid = False
            else:
                self.done = True
                valid = True
                self.n_actions += 1
            return self.state, (self.eos,), valid

    def load_dataset(self, split="D1", nfold=5):
        # TODO: rename to make_dataset()?
        source = get_default_data_splits(setting="Target")
        rng = np.random.RandomState()
        # returns a dictionary with two keys 'AMP' and 'nonAMP' and values as lists
        data = source.sample(split, -1)
        if split == "D1":
            groups = np.array(source.d1_pos.group)
        if split == "D2":
            groups = np.array(source.d2_pos.group)
        if split == "D":
            groups = np.concatenate(
                (np.array(source.d1_pos.group), np.array(source.d2_pos.group))
            )

        n_pos, n_neg = len(data["AMP"]), len(data["nonAMP"])
        pos_train, pos_test = next(
            GroupKFold(nfold).split(np.arange(n_pos), groups=groups)
        )
        neg_train, neg_test = next(
            GroupKFold(nfold).split(
                np.arange(n_neg), groups=rng.randint(0, nfold, n_neg)
            )
        )

        pos_train = [data["AMP"][i] for i in pos_train]
        neg_train = [data["nonAMP"][i] for i in neg_train]
        pos_test = [data["AMP"][i] for i in pos_test]
        neg_test = [data["nonAMP"][i] for i in neg_test]
        train = pos_train + neg_train
        test = pos_test + neg_test

        train = [sample for sample in train if len(sample) < self.max_seq_length]
        test = [sample for sample in test if len(sample) < self.max_seq_length]

        return train, test

    def get_pairwise_distance(self, samples):
        dists = []
        for pair in itertools.combinations(samples, 2):
            dists.append(self.get_distance(*pair))
        dists = torch.FloatTensor(dists)
        return dists

    def get_distance_from_D0(self, samples, dataset_obs):
        # TODO: optimize
        # TODO: should this be proxy2state?
        dataset_states = [self.policytorch2state(el) for el in dataset_obs]
        dataset_samples = self.statebatch2oracle(dataset_states)
        min_dists = []
        for sample in samples:
            dists = []
            sample_repeated = itertools.repeat(sample, len(dataset_samples))
            for s_0, x_0 in zip(sample_repeated, dataset_samples):
                dists.append(self.get_distance(s_0, x_0))
            min_dists.append(np.min(np.array(dists)))
        return torch.FloatTensor(min_dists)

    def get_distance(self, seq1, seq2):
        return levenshtein(seq1, seq2) / 1

    def plot_reward_distribution(self, scores, title):
        fig, ax = plt.subplots()
        if isinstance(scores, TensorType):
            scores = scores.cpu().detach().numpy()
        plt.hist(scores)
        ax.set_title(title)
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel("Energy")
        plt.show()
        plt.close()
        return fig

    # TODO: do we need this?
    def make_test_set(
        self,
        path_base_dataset,
        ntest,
        min_length=0,
        max_length=np.inf,
        seed=167,
        output_csv=None,
    ):
        """
        Constructs an approximately uniformly distributed (on the score) set, by
        selecting samples from a larger base set.

        Args
        ----
        path_base_dataset : str
            Path to a CSV file containing the base data set.

        ntest : int
            Number of test samples.

        seed : int
            Random seed.

        dask : bool
            If True, use dask to efficiently read a large base file.

        output_csv: str
            Optional path to store the test set as CSV.
        """
        if path_base_dataset is None:
            return None, None
        times = {
            "all": 0.0,
            "indices": 0.0,
        }
        t0_all = time.time()
        if seed:
            np.random.seed(seed)
        df_base = pd.read_csv(path_base_dataset, index_col=0)
        df_base = df_base.loc[
            (df_base["samples"].map(len) >= min_length)
            & (df_base["samples"].map(len) <= max_length)
        ]
        energies_base = df_base["energies"].values
        min_base = energies_base.min()
        max_base = energies_base.max()
        distr_unif = np.random.uniform(low=min_base, high=max_base, size=ntest)
        # Get minimum distance samples without duplicates
        t0_indices = time.time()
        idx_samples = []
        for idx in tqdm(range(ntest)):
            dist = np.abs(energies_base - distr_unif[idx])
            idx_min = np.argmin(dist)
            if idx_min in idx_samples:
                idx_sort = np.argsort(dist)
                for idx_next in idx_sort:
                    if idx_next not in idx_samples:
                        idx_samples.append(idx_next)
                        break
            else:
                idx_samples.append(idx_min)
        t1_indices = time.time()
        times["indices"] += t1_indices - t0_indices
        # Make test set
        df_test = df_base.iloc[idx_samples]
        if output_csv:
            df_test.to_csv(output_csv)
        t1_all = time.time()
        times["all"] += t1_all - t0_all
        return df_test, times
