"""
parent class to represent sequence-like environments, such as AMP and DNA. Sequences
are constructed by adding tokens from a dictionary. An alternative to this kind of
sequence environment (not-implemented as of July 2023) would be a "mutation-based"
modification of the sequences, or a combination of mutations and additions.
"""
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, set_device, tlong
from torch.distributions import Categorical
from torchtyping import TensorType


class Smiles(GFlowNetEnv):
    """
    SMILES environment: A sequence of SMILES tokens.

    See:
    https://github.com/deepchem/deepchem/blob/master/deepchem/feat/tests/data/vocab.txt

    Attributes
    ----------
    min_length : int
        Minimum length of the sequence.

    max_length : int
        Maximum length of the sequence.
    """

    def __init__(
        self,
        vobac_path: str = None,
        max_length: int = 512,
        **kwargs,
    ):
        # Base class init
        super().__init__(**kwargs)

        self.vocab_path = vocab_path

        # Set device because it is needed in the init
        self.device = set_device(kwargs["device"])

        # Tokenization
        # A BERT sequence has the following format: [CLS] X [SEP]
        self.tokenizer = SmilesTokenizer(self.vocab_path)
        self.vocab_dict = self.read_vocab(
            vocab_path
        )  # {0: '[PAD]', 1: '[unused1]', 2: '[unused2]' ...}

        self.tokens = tuple(
            self.vocab_dict.values()
        )  # ('[PAD]', '[unused1]', '[unused2]', '[unused3]', ...., '[Cn]' )
        self.token_indices = tuple(
            self.vocab_dict.keys()
        )  #  (0, 1, 2, 3, 4, ....., 590)
        self.n_tokens = len(self.tokens)
        self.max_length = max_length
        assert len(set(self.tokens)) == len(self.n_tokens)
        assert max_length > 0

        self.eos_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.eos_idx = tokenizer.convert_tokens_to_ids(
            self.eos_token
        )  # 13 is the idx for pad_token for the vocab.txt from deepchem smiletokenizer
        self.pad_idx = tokenizer.convert_tokens_to_ids(
            self.pad_token
        )  # 0 is the idx for pad_token for the vocab.txt from deepchem smiletokenizer
        self.dtype = type(self.pad_token)

        self.token2idx = {token: idx for idx, token in self.vocab_dict.items()}
        self.idx2token = self.vocab_dict

        # Source state: vector of length max_length filled with pad token
        self.source = tlong(
            torch.full((self.max_length,), self.pad_idx), device=self.device
        )

        # End-of-sequence action
        self.eos = (self.eos_idx,)

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs list with all possible actions, including eos.

        An action is represented by a single-element tuple indicating the index of the
        token to be added to the current sequence (state).

        The action space of this parent class is:
            action_space: [(0,), (1,), .....,  (589,), (590,)]]
        """
        return [(self.token2idx[token],) for token in self.tokens]

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[TensorType["max_length"]] = None,
        done: Optional[bool] = None,
    ) -> List[bool]:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.

        Args
        ----
        state : tensor
            Input state. If None, self.state is used.

        done : bool
            Whether the trajectory is done. If None, self.done is used.

        Returns
        -------
        A list of boolean values.
        """

        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [True for _ in range(self.action_space_dim)]
        # If sequence is not at maximum length, all actions are valid
        if state[-1] == self.pad_idx:
            return [False for _ in range(self.action_space_dim)]
        # Otherwise, only EOS is valid
        mask = [True for _ in range(self.action_space_dim)]
        mask[self.action_space.index(self.eos)] = False
        return mask

    def get_parents(
        self,
        state: Optional[TensorType["max_length"]] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        The GFlowNet graph is a tree and there is only one parent per state.

        Args
        ----
        state : tensor
            Input state. If None, self.state is used.

        done : bool
            Whether the trajectory is done. If None, self.done is used.

        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format. This environment has a single parent per
            state.

        actions : list
            List of actions that lead to state for each parent in parents. This
            environment has a single parent per state.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [state], [self.eos]
        if self.equal(state, self.source):
            return [], []
        pos_last_token = self._get_seq_length(state) - 1
        parent = copy(state)
        parent[pos_last_token] = self.pad_idx
        p_action = (state[pos_last_token],)
        return [parent], [p_action]

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> [TensorType["max_length"], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple int values indicating the
            dimensions to increment by 1.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        # Generic pre-step checks
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, False
        valid = True
        self.n_actions += 1
        # If action is EOS, set done to True and return state as is
        if action == self.eos:
            self.done = True
            return self.state, action, valid
        # Update state
        self.state[self._get_seq_length()] = action[0]
        return self.state, action, valid

    def states2proxy(
        self,
        states: Union[
            List[TensorType["max_length"]], TensorType["batch", "max_length"]
        ],
    ) -> List[List]:
        """
        Prepares a batch of states in "environment format" for a proxy: states
        are represented by the tokens instead of the indices, with padding up to the
        max_length.

        Important: by default, the output of states2proxy() is a list of lists, instead
        of a tensor as in most environments. This is to allow for string tokens.

        Example, with max_length = 16:
          - smiles: 'CC(=O)OC1=C'
          - tokens: tokenizer.encode(smiles, max_length=16, truncation=True, padding='max_length')
                        -> [12, 16, 16, 17, 22, 19, 18, 19, 16, 20, 22, 16, 13, 0, 0, 0]
                        add_special_tokens = False as argument will remove tokens 12 (['CLS']) and 13 ['SEP'] from the tokens
          - Sequence (tokens): [12, 16, 16, 17, 22, 19, 18, 19, 16, 20, 22, 16, 13, 0, 0, 0]
          - state: [12, 16, 16, 17, 22, 19, 18, 19, 16, 20, 22, 16, 13, 0, 0, 0]
          - proxy format: tokenizer(convert_ids_to_tokens(state) -->
                ['[CLS]', 'C', 'C', '(', '=', 'O', ')', 'O', 'C', '1', '=', 'C', '[SEP]', '[PAD]', '[PAD]', '[PAD]']

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A list containing all the states in the batch, represented themselves as lists.
        """
        states = tlong(states, device=self.device).tolist()
        states_proxy = []
        for state in states:
            # states_proxy.append([self.idx2token[idx] for idx in state])
            states_proxy.append(self.tokenizer.convert_ids_to_tokens(state))
        return states_proxy

    def states2policy(
        self,
        states: Union[
            List[TensorType["max_length"]], TensorType["batch", "max_length"]
        ],
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: states
        are one-hot encoded.

        Example, with max_length = 5:
          - Sequence (tokens): 0100
          - state: [1, 2, 1, 1, 0]
          - policy format: [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
                           |   0   |    1   |    0   |    0   |   PAD  |

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tlong(states, device=self.device)
        return (
            F.one_hot(states, self.n_tokens + 1)
            .reshape(states.shape[0], -1)
            .to(self.float)
        )

    def state2readable(self, state: TensorType["max_length"] = None) -> str:
        """
        Converts a state into a human-readable string.

        The output string contains the token corresponding to each index in the state,
        separated by spaces.

        Args
        ----
        states : tensor
            A state in environment format. If None, self.state is used.

        Returns
        -------
        A string of space-separated tokens.

        Example:
        state: [12, 16, 16, 17, 22, 19, 18, 19, 16, 20, 22, 16, 13, 0, 0, 0]
        tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(states)) -->
        [CLS] C C ( = O ) O C 1 = C [SEP] [PAD] [PAD] [PAD]'

        """
        state = self._get_state(state)
        state = state.tolist()
        # return "".join([str(self.idx2token[idx]) + " " for idx in state])[:-1]
        return self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(state)
        )

    def readable2state(self, readable: str) -> TensorType["max_length"]:
        """
        Converts a state in readable format into the "environment format" (tensor)

        Args
        ----
        readable : str
            A state in readable format - space-separated tokens.

        Returns
        -------
        A tensor containing the indices of the tokens.
        """
        return tlong(
            [self.token2idx[self.dtype(token)] for token in readable.split(" ")],
            device=self.device,
        )

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[TensorType["batch", "max_length"]]:
        """
        Constructs a batch of n states uniformly sampled in the sample space of the
        environment.

        Args
        ----
        n_states : int
            The number of states to sample.

        seed : int
            Random seed.
        """
        n_tokens = len(self.tokens)
        n_per_length = tlong(
            [n_tokens**length for length in range(1, self.max_length + 1)],
            device=self.device,
        )
        lengths = Categorical(logits=n_per_length.repeat(n_states, 1)).sample() + 1
        samples = torch.randint(
            low=1, high=n_tokens + 1, size=(n_states, self.max_length)
        )
        for idx, length in enumerate(lengths):
            samples[idx, length:] = 0
        return samples

    def _get_seq_length(self, state: TensorType["max_length"] = None):
        """
        Returns the effective length of a state, that is ignoring the padding.

        Args
        ----
        state : tensor
            The input sequence. If None, self.state is used.

        Returns
        -------
        Single element int tensor.
        """
        state = self._get_state(state)
        if state[-1] == self.pad_idx:
            return torch.where(state == self.pad_idx)[0][0]
        else:
            return len(state)

    def read_vocab(self, vocab_path):
        """
        return a dictionary of the format: {0: '[PAD]', 1: '[unused1]', 2: '[unused2]' ...}
        """
        vocab_dict = {}
        with open(self.vocab_path, "r") as fh:
            for i, line in enumerate(fh):
                token = line.strip()
                vocab_dict[i] = token
        return vocab_dict
