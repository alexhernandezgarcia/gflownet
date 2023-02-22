"""
Classes to represent aptamers environments
"""
from typing import List
import itertools
import numpy as np
import numpy.typing as npt
import pandas as pd
from gflownet.envs.base import GFlowNetEnv
import time


class AptamerSeq(GFlowNetEnv):
    """
    Aptamer sequence environment

    Attributes
    ----------
    max_seq_length : int
        Maximum length of the sequences

    min_seq_length : int
        Minimum length of the sequences

    n_alphabet : int
        Number of letters in the alphabet

    state : list
        Representation of a sequence (state), as a list of length max_seq_length where
        each element is the index of a letter in the alphabet, from 0 to (n_alphabet -
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
        max_seq_length=42,
        min_seq_length=1,
        n_alphabet=4,
        min_word_len=1,
        max_word_len=1,
        proxy=None,
        oracle=None,
        reward_beta=1,
        env_id=None,
        energies_stats=None,
        reward_norm=1.0,
        reward_norm_std_mult=0.0,
        reward_func="power",
        denorm_proxy=False,
        **kwargs,
    ):
        super(AptamerSeq, self).__init__(
            env_id,
            reward_beta,
            reward_norm,
            reward_norm_std_mult,
            reward_func,
            energies_stats,
            denorm_proxy,
            proxy,
            oracle,
            **kwargs,
        )
        self.source = []
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.n_alphabet = n_alphabet
        self.min_word_len = min_word_len
        self.max_word_len = max_word_len
        self.action_space = self.get_actions_space()
        self.eos = len(self.action_space)
        self.reset()
        self.fixed_policy_output = self.get_fixed_policy_output()
        self.random_policy_output = self.get_fixed_policy_output()
        self.policy_output_dim = len(self.fixed_policy_output)
        self.policy_input_dim = len(self.state2policy())
        self.max_traj_len = self.get_max_traj_len()
        # Set up proxy
        self.proxy.setup(self.max_seq_length)

    def get_actions_space(self):
        """
        Constructs list with all possible actions
        """
        assert self.max_word_len >= self.min_word_len
        valid_wordlens = np.arange(self.min_word_len, self.max_word_len + 1)
        alphabet = [a for a in range(self.n_alphabet)]
        actions = []
        for r in valid_wordlens:
            actions_r = [el for el in itertools.product(alphabet, repeat=r)]
            actions += actions_r
        return actions

    def get_max_traj_len(
        self,
    ):
        return self.max_seq_length / self.min_word_len + 1

    def reward_arbitrary_i(self, state):
        if len(state) > 0:
            return (state[-1] + 1) * len(state)
        else:
            return 0

    def statebatch2oracle(self, states: List[List]):
        """
        Prepares a batch of sequence states for the oracles.

        Args
        ----
        states : list of lists
            List of sequences.
        """
        queries = [s + [-1] * (self.max_seq_length - len(s)) for s in states]
        queries = np.array(queries, dtype=int)
        if queries.ndim == 1:
            queries = queries[np.newaxis, ...]
        queries += 1
        if queries.shape[1] == 1:
            import ipdb

            ipdb.set_trace()
            queries = np.column_stack((queries, np.zeros(queries.shape[0])))
        return queries

    def state2policy(self, state: List = None) -> List:
        """
        Transforms the sequence (state) given as argument (or self.state if None) into a
        one-hot encoding. The output is a list of length n_alphabet * max_seq_length,
        where each n-th successive block of n_alphabet elements is a one-hot encoding of
        the letter in the n-th position.

        Example:
          - Sequence: AATGC
          - state: [0, 1, 3, 2]
                    A, T, G, C
          - state2policy(state): [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                                 |     A    |      T    |      G    |      C    |

        If max_seq_length > len(s), the last (max_seq_length - len(s)) blocks are all
        0s.
        """
        if state is None:
            state = self.state.copy()
        state_policy = np.zeros(self.n_alphabet * self.max_seq_length, dtype=np.float32)
        if len(state) > 0:
            state_policy[(np.arange(len(state)) * self.n_alphabet + state)] = 1
        return state_policy

    def statebatch2policy(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Transforms a batch of states into the policy model format. The output is a numpy
        array of shape [n_states, n_angles * n_dim + 1].

        See state2policy().
        """
        cols, lengths = zip(
            *[
                (state + np.arange(len(state)) * self.n_alphabet, len(state))
                for state in states
            ]
        )
        rows = np.repeat(np.arange(len(states)), lengths)
        state_policy = np.zeros(
            (len(states), self.n_alphabet * self.max_seq_length), dtype=np.float32
        )
        state_policy[rows, np.concatenate(cols)] = 1.0
        return state_policy

    def policy2state(self, state_policy: List) -> List:
        """
        Transforms the one-hot encoding version of a sequence (state) given as argument
        into a a sequence of letter indices.

        Example:
          - Sequence: AATGC
          - state_policy: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
                          |     A    |      A    |      T    |      G    |      C    |
          - state: [0, 0, 1, 3, 2]
                    A, A, T, G, C
        """
        return np.where(
            np.reshape(state_policy, (self.max_seq_length, self.n_alphabet))
        )[1].tolist()

    def state2readable(self, state, alphabet={0: "A", 1: "T", 2: "C", 3: "G"}):
        """
        Transforms a sequence given as a list of indices into a sequence of letters
        according to an alphabet.
        """
        return [alphabet[el] for el in state]

    def readable2state(self, letters, alphabet={0: "A", 1: "T", 2: "C", 3: "G"}):
        """
        Transforms a sequence given as a list of indices into a sequence of letters
        according to an alphabet.
        """
        alphabet = {v: k for k, v in alphabet.items()}
        return [alphabet[el] for el in letters]

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = []
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

    def get_parents(self, state=None, done=None, action=None):
        """
        Determines all parents and actions that lead to sequence state

        Args
        ----
        state : list
            Representation of a sequence (state), as a list of length max_seq_length
            where each element is the index of a letter in the alphabet, from 0 to
            (n_alphabet - 1).

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]
        else:
            parents = []
            actions = []
            for idx, a in enumerate(self.action_space):
                is_parent = state[-len(a) :] == list(a)
                if not isinstance(is_parent, bool):
                    is_parent = all(is_parent)
                if is_parent:
                    parents.append(state[: -len(a)])
                    actions.append(idx)
        return parents, actions

    def step(self, action_idx):
        """
        Executes step given an action index

        If action_idx is smaller than eos (no stop), add action to next
        position.

        See: step_daug()
        See: step_chain()

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
        # If only possible action is eos, then force eos
        if len(self.state) == self.max_seq_length:
            self.done = True
            self.n_actions += 1
            return self.state, [self.eos], True
        # If action is not eos, then perform action
        if action_idx != self.eos:
            action = self.action_space[action_idx]
            state_next = self.state + list(action)
            if len(state_next) > self.max_seq_length:
                valid = False
            else:
                self.state = state_next
                valid = True
                self.n_actions += 1
            return self.state, action_idx, valid
        # If action is eos, then perform eos
        else:
            if len(self.state) < self.min_seq_length:
                valid = False
            else:
                self.done = True
                valid = True
                self.n_actions += 1
            return self.state, self.eos, valid

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        Returns a vector of length the action space + 1: True if action is invalid
        given the current state, False otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(len(self.action_space) + 1)]
        mask = [False for _ in range(len(self.action_space) + 1)]
        seq_length = len(state)
        if seq_length < self.min_seq_length:
            mask[self.eos] = True
        for idx, a in enumerate(self.action_space):
            if seq_length + len(a) > self.max_seq_length:
                mask[idx] = True
        return mask

    def no_eos_mask(self, state=None):
        """
        Returns True if no eos action is allowed given state
        """
        if state is None:
            state = self.state.copy()
        return len(state) < self.min_seq_length

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

    def make_train_set(
        self,
        ntrain,
        oracle=None,
        seed=168,
        output_csv=None,
    ):
        """
        Constructs a randomly sampled train set.

        Args
        ----
        ntest : int
            Number of test samples.

        seed : int
            Random seed.

        output_csv: str
            Optional path to store the test set as CSV.
        """
        samples_dict = oracle.initializeDataset(
            save=False, returnData=True, customSize=ntrain, custom_seed=seed
        )
        energies = samples_dict["energies"]
        samples_mat = samples_dict["samples"]
        state_letters = oracle.numbers2letters(samples_mat)
        state_ints = [
            "".join([str(el) for el in state if el > 0]) for state in samples_mat
        ]
        if isinstance(energies, dict):
            energies.update({"samples": state_letters, "indices": state_ints})
            df_train = pd.DataFrame(energies)
        else:
            df_train = pd.DataFrame(
                {"samples": state_letters, "indices": state_ints, "energies": energies}
            )
        if output_csv:
            df_train.to_csv(output_csv)
        return df_train

    # TODO: improve approximation of uniform
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

    @staticmethod
    def np2df(test_path, al_init_length, al_queries_per_iter, pct_test, data_seed):
        data_dict = np.load(test_path, allow_pickle=True).item()
        letters = numbers2letters(data_dict["samples"])
        df = pd.DataFrame(
            {
                "samples": letters,
                "energies": data_dict["energies"],
                "train": [False] * len(letters),
                "test": [False] * len(letters),
            }
        )
        # Split train and test section of init data set
        rng = np.random.default_rng(data_seed)
        indices = rng.permutation(al_init_length)
        n_tt = int(pct_test * len(indices))
        indices_tt = indices[:n_tt]
        indices_tr = indices[n_tt:]
        df.loc[indices_tt, "test"] = True
        df.loc[indices_tr, "train"] = True
        # Split train and test the section of each iteration to preserve splits
        idx = al_init_length
        iters_remaining = (len(df) - al_init_length) // al_queries_per_iter
        indices = rng.permutation(al_queries_per_iter)
        n_tt = int(pct_test * len(indices))
        for it in range(iters_remaining):
            indices_tt = indices[:n_tt] + idx
            indices_tr = indices[n_tt:] + idx
            df.loc[indices_tt, "test"] = True
            df.loc[indices_tr, "train"] = True
            idx += al_queries_per_iter
        return df
