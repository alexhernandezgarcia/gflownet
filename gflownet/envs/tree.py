import pickle
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch_geometric as pyg
from networkx.drawing.nx_pydot import graphviz_layout
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from torch.distributions import Beta, Categorical, MixtureSameFamily, Uniform
from torch_geometric.utils.convert import from_networkx
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv


class NodeType:
    """
    Encodes two types of nodes present in a tree:
    0 - condition node (node other than leaf), that stores the information
        about on which feature to make the decision, and using what threshold.
    1 - classifier node (leaf), that stores the information about class output
        that will be predicted once that node is reached.
    """

    CONDITION = 0
    CLASSIFIER = 1


class Operator:
    """
    Operator based on which the decision is made (< or >=).

    We assume the convention of having left child output label = 0
    and right child label = 1 if operator is <, and the opposite if
    operator is >=. That way, during prediction, we can act as if
    the operator was always the same (and only care about the output
    label).
    """

    LT = 0
    GTE = 1


class Status:
    """
    Status of the node. Every node except the one on which a macro step
    was initiated will be marked as inactive; the node will be marked
    as active iff the process of its splitting is in progress.
    """

    INACTIVE = 0
    ACTIVE = 1


class Stage:
    """
    Current stage of the tree, encoded as part of the state.
    0 - complete, indicates that there is no macro step initiated, and
        the only allowed action is to pick one of the leaves for splitting.
    1 - leaf, indicates that a leaf was picked for splitting, and the only
        allowed action is picking a feature on which it will be split.
    2 - feature, indicates that a feature was picked, and the only allowed
        action is picking a threshold for splitting.
    3 - threshold, indicates that a threshold was picked, and the only
        allowed action is picking an operator.
    4 - operator, indicates that operator was picked. The only allowed
        action from here is finalizing the splitting process and spawning
        two new leaves, which should be done automatically, upon which
        the stage should be changed to complete.
    """

    COMPLETE = 0
    LEAF = 1
    FEATURE = 2
    THRESHOLD = 3
    OPERATOR = 4


class ActionType:
    """
    Type of action that will be passed to Tree.step. Refer to Stage for details.
    """

    PICK_LEAF = 0
    PICK_FEATURE = 1
    PICK_THRESHOLD = 2
    PICK_OPERATOR = 3


class Attribute:
    """
    Contains indices of individual attributes in a state tensor.

    Types of attributes defining each node of the tree:

        0 - node type (condition or classifier),
        1 - index of the feature used for splitting (condition node only, -1 otherwise),
        2 - decision threshold (condition node only, -1 otherwise),
        3 - class output (classifier node only, -1 otherwise), in the case of < operator
            the left child will have class = 0, and the right child will have class = 1;
            the opposite for the >= operator,
        4 - whether the node has active status (1 if node was picked and the macro step
            didn't finish yet, 0 otherwise).
    """

    TYPE = 0
    FEATURE = 1
    THRESHOLD = 2
    CLASS = 3
    ACTIVE = 4
    N = 5  # Total number of attributes.


class Tree(GFlowNetEnv):
    """
    GFlowNet environment representing a decision tree.

    Constructing a tree consists of a combination of macro steps (picking a leaf
    to split using a given feature, threshold and operator), which are divided
    into a series of consecutive micro steps (1 - pick a leaf, 2 - pick a feature,
    3 - pick a threshold, 4 - pick an operator). A consequence of that is, as long
    as a macro step is not in progress, the tree constructed so far is always a
    valid decision tree, which means that forward-looking loss etc. can be used.

    Internally, the tree is represented as a fixed-shape tensor (thus, specifying
    the maximum depth is required), with nodes indexed from k = 0 to 2**max_depth - 2,
    and each node containing a 5-element attribute tensor (see class Attribute for
    details). The nodes are indexed from top left to bottom right, as follows:

                0
        1               2
    3       4       5       6

    States are represented by a tensor with shape [n_nodes + 1, 5], where each k-th row
    corresponds to the attributes of the k-th node of the tree. The last row contains
    the information about the stage of the tree (see class Stage).
    """

    def __init__(
        self,
        X_train: Optional[npt.NDArray] = None,
        y_train: Optional[npt.NDArray] = None,
        X_test: Optional[npt.NDArray] = None,
        y_test: Optional[npt.NDArray] = None,
        data_path: Optional[str] = None,
        scale_data: bool = True,
        max_depth: int = 10,
        threshold_components: int = 1,
        beta_params_min: float = 0.1,
        beta_params_max: float = 2.0,
        fixed_distribution: dict = {
            "beta_alpha": 2.0,
            "beta_beta": 5.0,
        },
        random_distribution: dict = {
            "beta_alpha": 1.0,
            "beta_beta": 1.0,
        },
        **kwargs,
    ):
        """
        Attributes
        ----------
        X_train : np.array
            Train dataset, with dimensionality (n_observations, n_features). It may be
            None if a data set is provided via data_path.

        y_train : np.array
            Train labels, with dimensionality (n_observations,). It may be
            None if a data set is provided via data_path.

        X_test : np.array
            Test dataset, with dimensionality (n_observations, n_features). It may be
            None if a data set is provided via data_path, or if you don't want to perform
            test set evaluation.

        y_train : np.array
            Test labels, with dimensionality (n_observations,). It may be
            None if a data set is provided via data_path, or if you don't want to perform
            test set evaluation.

        data_path : str
            A path to a data set, with the following options:
            - *.pkl: Pickled dict with X_train, y_train, and (optional) X_test and y_test
              variables.
            - *.csv: CSV containing an optional 'Split' column in the last place, containing
              'train' and 'test' values, and M remaining columns, where the first (M - 1)
              columns will be taken to construct the input X, and M-th column will be the
              target y.
            Ignored if X_train and y_train are not None.

        scale_data : bool
            Whether to perform min-max scaling on the provided data (to a [0; 1] range).

        max_depth : int
            Maximum depth of a tree.

        threshold_components : int
            The number of mixture components that will be used for sampling
            the threshold.
        """
        if X_train is not None and y_train is not None:
            self.X_train = X_train
            self.y_train = y_train
            if X_test is not None and y_test is not None:
                self.X_test = X_test
                self.y_test = y_test
            else:
                self.X_test = None
                self.y_test = None
        elif data_path is not None:
            self.X_train, self.y_train, self.X_test, self.y_test = Tree._load_dataset(
                data_path
            )
        else:
            raise ValueError(
                "A Tree must be initialised with a data set. X_train, y_train and data_path cannot "
                "be all None"
            )
        if scale_data:
            self.scaler = MinMaxScaler().fit(self.X_train)
            self.X_train = self.scaler.transform(self.X_train)
            if self.X_test is not None:
                self.X_test = self.scaler.transform(self.X_test)
        self.y_train = self.y_train.astype(int)
        if self.y_test is not None:
            self.y_test = self.y_test.astype(int)
        if not set(self.y_train).issubset({0, 1}):
            raise ValueError(
                f"Expected y_train to have values in {{0, 1}}, received {set(self.y_train)}."
            )
        self.n_features = self.X_train.shape[1]
        self.max_depth = max_depth
        # Parameters of the policy distribution
        self.components = threshold_components
        self.beta_params_min = beta_params_min
        self.beta_params_max = beta_params_max
        # Source will contain information about the current stage (on the last position),
        # and up to 2**max_depth - 1 nodes, each with Attribute.N attributes, for a total of
        # 1 + Attribute.N * (2**max_depth - 1) values. The root (0-th node) of the
        # source is initialized with a classifier.
        self.n_nodes = 2**max_depth - 1
        self.source = torch.full((self.n_nodes + 1, Attribute.N), torch.nan)
        self._set_stage(Stage.COMPLETE, self.source)
        attributes_root = self.source[0]
        attributes_root[Attribute.TYPE] = NodeType.CLASSIFIER
        attributes_root[Attribute.FEATURE] = -1
        attributes_root[Attribute.THRESHOLD] = -1
        attributes_root[Attribute.CLASS] = Counter(self.y_train).most_common()[0][0]
        attributes_root[Attribute.ACTIVE] = Status.INACTIVE

        # End-of-sequence action.
        self.eos = (-1, -1)

        # Conversions
        # TODO: add functionality to select conversion type depending on config
        self.state2policy = self.state2policy_mlp
        self.statetorch2policy = self.statetorch2policy_mlp
        # self.statetorch2proxy = self.statetorch2policy
        self.statetorch2oracle = self.statetorch2policy

        super().__init__(
            fixed_distribution=fixed_distribution,
            random_distribution=random_distribution,
            **kwargs,
        )

    @staticmethod
    def _get_parent(k: int) -> Optional[int]:
        """
        Get node index of a parent of k-th node.
        """
        if k == 0:
            return None
        return (k - 1) // 2

    @staticmethod
    def _get_left_child(k: int) -> int:
        """
        Get node index of a left child of k-th node.
        """
        return 2 * k + 1

    @staticmethod
    def _get_right_child(k: int) -> int:
        """
        Get node index of a right child of k-th node.
        """
        return 2 * k + 2

    def _get_stage(self, state: Optional[torch.Tensor] = None) -> int:
        """
        Returns the stage of the current environment from self.state[-1, 0] or from the
        state passed as an argument.
        """
        if state is None:
            state = self.state
        return state[-1, 0]

    def _set_stage(
        self, stage: int, state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sets the stage of the current environment (self.state) or of the state passed
        as an argument by updating state[-1, 0].
        """
        if state is None:
            state = self.state
        state[-1, 0] = stage
        return state

    def _pick_leaf(self, k: int) -> None:
        """
        Select one of the leaves (classifier nodes) that will be split, and initiate
        macro step.
        """
        attributes = self.state[k]

        assert self._get_stage() == Stage.COMPLETE
        assert attributes[Attribute.TYPE] == NodeType.CLASSIFIER
        assert not torch.any(torch.isnan(attributes))
        assert torch.all(attributes[1:3] == -1)
        assert attributes[Attribute.ACTIVE] == Status.INACTIVE

        attributes[Attribute.TYPE] = NodeType.CONDITION
        attributes[1:4] = -1
        attributes[Attribute.ACTIVE] = Status.ACTIVE

        self._set_stage(Stage.LEAF)

    def _pick_feature(self, k: int, feature: float) -> None:
        """
        Select the feature on which currently selected leaf will be split.
        """
        attributes = self.state[k]

        assert self._get_stage() == Stage.LEAF
        assert attributes[Attribute.TYPE] == NodeType.CONDITION
        assert torch.all(attributes[1:4] == -1)
        assert attributes[Attribute.ACTIVE] == Status.ACTIVE

        attributes[Attribute.FEATURE] = feature

        self._set_stage(Stage.FEATURE)

    def _pick_threshold(self, k: int, threshold: float) -> None:
        """
        Select the threshold for splitting the currently selected leaf ond
        the selected feature.
        """
        attributes = self.state[k]

        assert self._get_stage() == Stage.FEATURE
        assert attributes[Attribute.TYPE] == NodeType.CONDITION
        assert attributes[Attribute.FEATURE] >= 0
        assert torch.all(attributes[2:4] == -1)
        assert attributes[Attribute.ACTIVE] == Status.ACTIVE

        attributes[Attribute.THRESHOLD] = threshold

        self._set_stage(Stage.THRESHOLD)

    def _pick_operator(self, k: int, operator: float) -> None:
        """
        Select the operator (< or >=) for splitting the currently selected
        left, feature and threshold, temporarily encode it in attributes,
        and initiate final splitting.
        """
        attributes = self.state[k]

        assert self._get_stage() == Stage.THRESHOLD
        assert attributes[Attribute.TYPE] == NodeType.CONDITION
        assert torch.all(attributes[1:3] >= 0)
        assert attributes[Attribute.CLASS] == -1
        assert attributes[Attribute.ACTIVE] == Status.ACTIVE

        attributes[Attribute.CLASS] = operator

        self._set_stage(Stage.OPERATOR)

        self._split_leaf(k)

    def _split_leaf(self, k: int) -> None:
        """
        Finalize the splitting of a leaf using set feature, threshold
        and operator: transform the leaf into a condition node and
        spawn two children.

        We assume the convention of having left child output label = 0
        and right child label = 1 if operator is <, and the opposite if
        operator is >=. That way, during prediction, we can act as if
        the operator was always the same (and only care about the output
        label).
        """
        attributes = self.state[k]

        assert self._get_stage() == Stage.OPERATOR
        assert attributes[Attribute.TYPE] == NodeType.CONDITION
        assert torch.all(attributes[1:4] >= 0)
        assert attributes[Attribute.ACTIVE] == Status.ACTIVE

        k_left = Tree._get_left_child(k)
        k_right = Tree._get_right_child(k)

        if attributes[Attribute.CLASS] == Operator.LT:
            self._insert_classifier(k_left, output=0)
            self._insert_classifier(k_right, output=1)
        else:
            self._insert_classifier(k_left, output=1)
            self._insert_classifier(k_right, output=0)

        attributes[Attribute.CLASS] = -1
        attributes[Attribute.ACTIVE] = Status.INACTIVE

        self._set_stage(Stage.COMPLETE)

    def _insert_classifier(self, k: int, output: int) -> None:
        """
        Replace attributes of k-th node with those of a classifier node.
        """
        attributes = self.state[k]

        assert torch.all(torch.isnan(attributes))

        attributes[Attribute.TYPE] = NodeType.CLASSIFIER
        attributes[Attribute.FEATURE] = -1
        attributes[Attribute.THRESHOLD] = -1
        attributes[Attribute.CLASS] = output
        attributes[Attribute.ACTIVE] = Status.INACTIVE

    def get_action_space(self) -> List[Tuple[int, int]]:
        """
        Actions are a tuple containing:
            1) action type:
                0 - pick leaf to split,
                1 - pick feature,
                2 - pick threshold,
                3 - pick operator,
            2) action value, depending on the action type:
                pick leaf: leaf index
                pick feature: feature index
                pick threshold: threshold value
                pick operator: operator index
        """
        actions = []
        # Pick leaf
        self._action_index_pick_leaf = 0
        actions.extend([(ActionType.PICK_LEAF, idx) for idx in range(self.n_nodes)])
        # Pick feature
        self._action_index_pick_feature = len(actions)
        actions.extend(
            [(ActionType.PICK_FEATURE, idx) for idx in range(self.n_features)]
        )
        # Pick threshold
        self._action_index_pick_threshold = len(actions)
        actions.extend([(ActionType.PICK_THRESHOLD, -1)])
        # Pick operator
        self._action_index_pick_operator = len(actions)
        actions.extend(
            [(ActionType.PICK_OPERATOR, idx) for idx in [Operator.LT, Operator.GTE]]
        )
        # EOS
        self._action_index_eos = len(actions)
        actions.append(self.eos)

        return actions

    def step(
        self, action: Tuple[int, Union[int, float]], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int, Union[int, float]], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed.
            See: self.get_action_space()

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
        # Replace the continuous value of threshold by -1 to allow checking it
        action_to_check = self.action2representative(action)
        do_step, self.state, action_to_check = self._pre_step(
            action_to_check, skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, False

        self.n_actions += 1

        if action != self.eos:
            action_type, action_value = action

            if action_type == ActionType.PICK_LEAF:
                self._pick_leaf(action_value)
            else:
                active_node = self._find_active(self.state)
                if action_type == ActionType.PICK_FEATURE:
                    self._pick_feature(active_node, action_value)
                elif action_type == ActionType.PICK_THRESHOLD:
                    self._pick_threshold(active_node, action_value)
                elif action_type == ActionType.PICK_OPERATOR:
                    self._pick_operator(active_node, action_value)
                else:
                    raise NotImplementedError(
                        f"Unrecognized action type: {action_type}."
                    )

            return self.state, action, True
        else:
            self.done = True
            return self.state, action, True

    def set_state(self, state: List, done: Optional[bool] = False):
        """
        Sets the state and done. If done is True but incompatible with state (Stage is
        not COMPLETE), then force done False and print warning.
        """
        if done is True and self._get_stage() != Stage.COMPLETE:
            done = False
            warnings.warn(
                f"""
            Attempted to set state {self.state2readable(state)} with done = True, which
            is not compatible with the environment. Forcing done = False.
            """
            )
        return super().set_state(state, done)

    def sample_actions(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        sampling_method: str = "policy",
        mask_invalid_actions: TensorType["n_states", "action_space_dim"] = None,
        temperature_logits: float = 1.0,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs.
        """
        n_states = policy_outputs.shape[0]
        logprobs = torch.zeros(n_states, device=self.device, dtype=self.float)
        # Discrete actions
        mask_discrete = mask_invalid_actions[:, self._action_index_pick_threshold]
        if torch.any(mask_discrete):
            policy_outputs_discrete = policy_outputs[
                mask_discrete, : self._index_continuous_policy_output
            ]
            actions_discrete, logprobs_discrete = super().sample_actions(
                policy_outputs_discrete,
                sampling_method,
                mask_invalid_actions[
                    mask_discrete, : self._index_continuous_policy_output
                ],
                temperature_logits,
            )
            logprobs[mask_discrete] = logprobs_discrete
        if torch.all(mask_discrete):
            return actions_discrete, logprobs
        # Continuous actions
        mask_cont = torch.logical_not(mask_discrete)
        n_cont = mask_cont.sum()
        policy_outputs_cont = policy_outputs[
            mask_cont, self._index_continuous_policy_output :
        ]
        if sampling_method == "uniform":
            distr_threshold = Uniform(
                torch.zeros(n_cont),
                torch.ones(n_cont),
            )
        elif sampling_method == "policy":
            mix_logits = policy_outputs_cont[:, 0::3]
            mix = Categorical(logits=mix_logits)
            alphas = policy_outputs_cont[:, 1::3]
            alphas = self.beta_params_max * torch.sigmoid(alphas) + self.beta_params_min
            betas = policy_outputs_cont[:, 2::3]
            betas = self.beta_params_max * torch.sigmoid(betas) + self.beta_params_min
            beta_distr = Beta(alphas, betas)
            distr_threshold = MixtureSameFamily(mix, beta_distr)
        thresholds = distr_threshold.sample()
        # Log probs
        logprobs[mask_cont] = distr_threshold.log_prob(thresholds)
        # Build actions
        actions_cont = [(ActionType.PICK_THRESHOLD, th.item()) for th in thresholds]
        actions = []
        for is_discrete in mask_discrete:
            if is_discrete:
                actions.append(actions_discrete.pop(0))
            else:
                actions.append(actions_cont.pop(0))
        return actions, logprobs

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        is_forward: bool,
        actions: TensorType["n_states", "n_dim"],
        states_target: TensorType["n_states", "policy_input_dim"],
        mask_invalid_actions: TensorType["n_states", "1"] = None,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.
        """
        n_states = policy_outputs.shape[0]
        if states_target is None:
            states_target = torch.empty(
                (n_states, self.policy_input_dim), device=self.device
            )
        logprobs = torch.zeros(n_states, device=self.device, dtype=self.float)
        # Discrete actions
        mask_discrete = mask_invalid_actions[:, self._action_index_pick_threshold]
        if torch.any(mask_discrete):
            policy_outputs_discrete = policy_outputs[
                mask_discrete, : self._index_continuous_policy_output
            ]
            logprobs_discrete = super().get_logprobs(
                policy_outputs_discrete,
                is_forward,
                actions[mask_discrete],
                states_target[mask_discrete],
                mask_invalid_actions[
                    mask_discrete, : self._index_continuous_policy_output
                ],
            )
            logprobs[mask_discrete] = logprobs_discrete
        if torch.all(mask_discrete):
            return logprobs
        # Continuous actions
        mask_cont = torch.logical_not(mask_discrete)
        policy_outputs_cont = policy_outputs[
            mask_cont, self._index_continuous_policy_output :
        ]
        mix_logits = policy_outputs_cont[:, 0::3]
        mix = Categorical(logits=mix_logits)
        alphas = policy_outputs_cont[:, 1::3]
        alphas = self.beta_params_max * torch.sigmoid(alphas) + self.beta_params_min
        betas = policy_outputs_cont[:, 2::3]
        betas = self.beta_params_max * torch.sigmoid(betas) + self.beta_params_min
        beta_distr = Beta(alphas, betas)
        distr_threshold = MixtureSameFamily(mix, beta_distr)
        thresholds = actions[mask_cont, -1]
        logprobs[mask_cont] = distr_threshold.log_prob(thresholds)
        return logprobs

    def state2policy_mlp(
        self, state: Optional[TensorType["state_dim"]] = None
    ) -> TensorType["policy_input_dim"]:
        """
        Prepares a state in "GFlowNet format" for the policy model.
        """
        if state is None:
            state = self.state.clone().detach()
        state[state.isnan()] = -1
        return state.flatten()

    def statetorch2policy_mlp(
        self, states: TensorType["batch_size", "state_dim"]
    ) -> TensorType["batch_size", "policy_input_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for an MLP policy model.
        It simply replaces the NaNs by -1s.
        """
        states[states.isnan()] = -1
        return states.flatten(start_dim=1)

    def policy2state(
        self, policy: Optional[TensorType["policy_input_dim"]] = None
    ) -> None:
        """
        Returns None to signal that the conversion is not reversible.
        """
        return None

    def statebatch2proxy(
        self, states: List[TensorType["state_dim"]]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in "GFlowNet format" for the proxy: simply
        stacks the list of tensors and calls self.statetorch2proxy.

        Args
        ----
        state : list
        """
        states = torch.stack(states)
        return self.statetorch2proxy(states)

    def _attributes_to_readable(self, attributes: List) -> str:
        # Node type
        if attributes[Attribute.TYPE] == NodeType.CONDITION:
            node_type = "condition, "
        elif attributes[Attribute.TYPE] == NodeType.CLASSIFIER:
            node_type = "classifier, "
        else:
            return ""
        # Feature
        feature = f"feat. {str(attributes[Attribute.FEATURE])}, "
        # Decision threshold
        if attributes[Attribute.THRESHOLD] != -1:
            assert attributes[Attribute.TYPE] == 0
            threshold = f"th. {str(attributes[Attribute.THRESHOLD])}, "
        else:
            threshold = "th. -1, "
        # Class output
        if attributes[Attribute.CLASS] != -1:
            assert attributes[Attribute.TYPE] == 1
            class_output = f"class {str(attributes[Attribute.CLASS])}, "
        else:
            class_output = "class -1, "
        if attributes[Attribute.ACTIVE] == Status.ACTIVE:
            active = " (active)"
        else:
            active = ""
        return node_type + feature + threshold + class_output + active

    def state2readable(self, state=None):
        """
        Converts a state into human-readable representation.
        """
        if state is None:
            state = self.state.clone().detach()
        state = state.cpu().numpy()
        readable = ""
        for idx in range(self.n_nodes):
            attributes = self._attributes_to_readable(state[idx])
            if len(attributes) == 0:
                continue
            readable += f"{idx}: {attributes} | "
        readable += f"stage: {self._get_stage(state)}"
        return readable

    def _readable_to_attributes(self, readable: str) -> List:
        attributes_list = readable.split(", ")
        # Node type
        if attributes_list[0] == "condition":
            node_type = NodeType.CONDITION
        elif attributes_list[0] == "classifier":
            node_type = NodeType.CLASSIFIER
        else:
            node_type = -1
        # Feature
        feature = float(attributes_list[1].split("feat. ")[-1])
        # Decision threshold
        threshold = float(attributes_list[2].split("th. ")[-1])
        # Class output
        class_output = float(attributes_list[3].split("class ")[-1])
        # Active
        if "(active)" in readable:
            active = Status.ACTIVE
        else:
            active = Status.INACTIVE
        return [node_type, feature, threshold, class_output, active]

    def readable2state(self, readable):
        """
        Converts a human-readable representation of a state into the standard format.
        """
        readable_list = readable.split(" | ")
        state = torch.full((self.n_nodes + 1, Attribute.N), torch.nan)
        for el in readable_list[:-1]:
            node_index, attributes_str = el.split(": ")
            node_index = int(node_index)
            attributes = self._readable_to_attributes(attributes_str)
            for idx, att in enumerate(attributes):
                state[node_index, idx] = att
        stage = float(readable_list[-1].split("stage: ")[-1])
        state = self._set_stage(stage, state)
        return state

    @staticmethod
    def _find_leaves(state: torch.Tensor) -> List[int]:
        """
        Compute indices of leaves from a state.
        """
        return torch.where(state[:-1, Attribute.TYPE] == NodeType.CLASSIFIER)[
            0
        ].tolist()

    @staticmethod
    def _find_active(state: torch.Tensor) -> int:
        """
        Get index of the (only) active node. Assumes that active node exists
        (that we are in the middle of a macro step).
        """
        active = torch.where(state[:-1, Attribute.ACTIVE] == Status.ACTIVE)[0]
        assert len(active) == 1
        return active.item()

    def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled. It initializes the output tensor
        by using the parameters provided in the argument params.

        The output of the policy of a Tree environment consists of a discrete and
        continuous part. The discrete part (first part) corresponds to the discrete
        actions, while the continuous part (second part) corresponds to the single
        continuous action, that is the sampling of the threshold of a node classifier.

        The latter is modelled by a mixture of Beta distributions. Therefore, the
        continuous part of of the policy output is vector of dimensionality c * 3,
        where c is the number of components in the mixture (self.components).
        The three parameters of each component are the following:

          1) the weight of the component in the mixture
          2) the logit(alpha) parameter of the Beta distribution to sample the
             threshold.
          3) the logit(beta) parameter of the Beta distribution to sample the
             threshold.

        Note: contrary to other environments where there is a need to model a mixture
        of discrete and continuous distributions (for example to consider the
        possibility of sampling the EOS action instead of a continuous action), there
        is no such need here because either the continuous action is the only valid
        action or it is not valid.
        """
        policy_output_discrete = torch.ones(
            self.action_space_dim, device=self.device, dtype=self.float
        )
        self._index_continuous_policy_output = len(policy_output_discrete)
        self._len_continuous_policy_output = self.components * 3
        policy_output_continuous = torch.ones(
            self._len_continuous_policy_output,
            device=self.device,
            dtype=self.float,
        )
        policy_output_continuous[1::3] = params["beta_alpha"]
        policy_output_continuous[2::3] = params["beta_beta"]
        return torch.cat([policy_output_discrete, policy_output_continuous])

    def get_mask_invalid_actions_forward(
        self, state: Optional[torch.Tensor] = None, done: Optional[bool] = None
    ) -> List[bool]:
        if state is None:
            state = self.state
        if done is None:
            done = self.done

        if done:
            return [True] * self.policy_output_dim

        leaves = Tree._find_leaves(state)
        stage = self._get_stage(state)
        mask = [True] * self.policy_output_dim

        if stage == Stage.COMPLETE:
            # In the "complete" stage (in which there are no ongoing micro steps)
            # only valid actions are the ones for picking one of the leaves or EOS.
            for k in leaves:
                # Check if splitting the node wouldn't exceed max depth.
                if Tree._get_right_child(k) < self.n_nodes:
                    mask[self._action_index_pick_leaf + k] = False
            mask[self._action_index_eos] = False
        elif stage == Stage.LEAF:
            # Leaf was picked, only picking the feature actions are valid.
            for idx in range(
                self._action_index_pick_feature, self._action_index_pick_threshold
            ):
                mask[idx] = False
        elif stage == Stage.FEATURE:
            # Feature was picked, only picking the threshold action is valid.
            for idx in range(
                self._action_index_pick_threshold, self._action_index_pick_operator
            ):
                mask[idx] = False
        elif stage == Stage.THRESHOLD:
            # Threshold was picked, only picking the operator actions are valid.
            for idx in range(self._action_index_pick_operator, self._action_index_eos):
                mask[idx] = False
        else:
            raise ValueError(f"Unrecognized stage {stage}.")

        return mask

    def get_mask_invalid_actions_backward(
        self,
        state: Optional[torch.Tensor] = None,
        done: Optional[bool] = None,
        parents_a: Optional[List] = None,
    ) -> List:
        """
        Simply appends to the standard "discrete part" of the mask a dummy part
        corresponding to the continuous part of the policy output so as to match the
        dimensionality.
        """
        return (
            super().get_mask_invalid_actions_backward(state, done, parents_a)
            + [True] * self._len_continuous_policy_output
        )

    def get_parents(
        self,
        state: Optional[torch.Tensor] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        if state is None:
            state = self.state
        if done is None:
            done = self.done

        if done:
            return [state], [self.eos]

        leaves = Tree._find_leaves(state)
        stage = self._get_stage(state)
        parents = []
        actions = []

        if stage == Stage.COMPLETE:
            # In the "complete" stage (in which there are no ongoing micro steps),
            # to find parents we first look for the nodes for which both children
            # are leaves, and then undo the last "pick operator" micro step.
            # In other words, reverse self._pick_operator, self._split_leaf and
            # self._insert_classifier (for both children).
            leaves = set(leaves)
            triplets = []
            for k in leaves:
                if k % 2 == 1 and k + 1 in leaves:
                    triplets.append((Tree._get_parent(k), k, k + 1))
            for k_parent, k_left, k_right in triplets:
                parent = state.clone()
                attributes_parent = parent[k_parent]
                attributes_left = parent[k_left]
                attributes_right = parent[k_right]

                # Set action operator as class of left child
                action = (
                    ActionType.PICK_OPERATOR,
                    int(attributes_left[Attribute.CLASS].item()),
                )

                # Revert stage (to "threshold": we skip "operator" because from it,
                # finalizing splitting should be automatically executed).
                parent = self._set_stage(Stage.THRESHOLD, parent)

                # Reset children attributes.
                attributes_left[:] = torch.nan
                attributes_right[:] = torch.nan

                # Revert parent attributes to the previous state.
                attributes_parent[Attribute.CLASS] = -1
                attributes_parent[Attribute.ACTIVE] = Status.ACTIVE

                parents.append(parent)
                actions.append(action)
        else:
            k = Tree._find_active(state)

            if stage == Stage.LEAF:
                # Reverse self._pick_leaf.
                for output in [0, 1]:
                    parent = state.clone()
                    attributes = parent[k]

                    parent = self._set_stage(Stage.COMPLETE, parent)
                    attributes[Attribute.TYPE] = NodeType.CLASSIFIER
                    attributes[Attribute.FEATURE] = -1
                    attributes[Attribute.THRESHOLD] = -1
                    attributes[Attribute.CLASS] = output
                    attributes[Attribute.ACTIVE] = Status.INACTIVE

                    parents.append(parent)
                    actions.append((ActionType.PICK_LEAF, k))
            elif stage == Stage.FEATURE:
                # Reverse self._pick_feature.
                parent = state.clone()
                attributes = parent[k]

                parent = self._set_stage(Stage.LEAF, parent)
                attributes[Attribute.FEATURE] = -1

                parents.append(parent)
                actions.append((ActionType.PICK_FEATURE, self.state[k][1].item()))
            elif stage == Stage.THRESHOLD:
                # Reverse self._pick_threshold.
                parent = state.clone()
                attributes = parent[k]

                parent = self._set_stage(Stage.FEATURE, parent)
                attributes[Attribute.THRESHOLD] = -1

                parents.append(parent)
                actions.append((ActionType.PICK_THRESHOLD, -1))
            else:
                raise ValueError(f"Unrecognized stage {stage}.")

        return parents, actions

    @staticmethod
    def action2representative(action: Tuple) -> Tuple:
        """
        Replaces the continuous value of a PICK_THRESHOLD action by -1 so that it can
        be contrasted with the action space and masks.
        """
        if action[0] == ActionType.PICK_THRESHOLD:
            action = (ActionType.PICK_THRESHOLD, -1)
        return action

    def get_max_traj_length(self) -> int:
        return self.n_nodes * Attribute.N

    def _get_graph(self, graph: Optional[nx.DiGraph] = None, k: int = 0) -> nx.DiGraph:
        """
        Recursively convert self.state into a networkx directional graph.
        """
        if graph is None:
            graph = nx.DiGraph()

        attributes = self.state[k]
        graph.add_node(k, x=attributes)

        if attributes[Attribute.TYPE] != NodeType.CLASSIFIER:
            k_left = Tree._get_left_child(k)
            if not torch.any(torch.isnan(self.state[k_left])):
                self._get_graph(graph, k=k_left)
                graph.add_edge(k, k_left)

            k_right = Tree._get_right_child(k)
            if not torch.any(torch.isnan(self.state[k_right])):
                self._get_graph(graph, k=k_right)
                graph.add_edge(k, k_right)

        return graph

    def _to_pyg(self) -> pyg.data.Data:
        """
        Convert self.state into a PyG graph.
        """
        return from_networkx(self._get_graph())

    @staticmethod
    def _load_dataset(data_path):
        data_path = Path(data_path)
        if data_path.suffix == ".csv":
            df = pd.read_csv(data_path)
            if df.columns[-1].lower() != "split":
                X_train = df.iloc[:, 0:-1].values
                y_train = df.iloc[:, -1].values
                X_test = None
                y_test = None
            else:
                if set(df.iloc[:, -1]) != {"train", "test"}:
                    raise ValueError(
                        f"Expected df['Split'] to have values in {{'train', 'test'}}, "
                        f"received {set(df.iloc[:, -1])}."
                    )
                X_train = df[df.iloc[:, -1] == "train"].iloc[:, 0:-2].values
                y_train = df[df.iloc[:, -1] == "train"].iloc[:, -2].values
                X_test = df[df.iloc[:, -1] == "test"].iloc[:, 0:-2].values
                y_test = df[df.iloc[:, -1] == "test"].iloc[:, -2].values
        elif data_path.suffix == ".pkl":
            with open(data_path, "rb") as f:
                dct = pickle.load(f)
                X_train = dct["X_train"]
                y_train = dct["y_train"]
                X_test = dct.get("X_test")
                y_test = dct.get("y_test")
        else:
            raise ValueError(
                "data_path must be a CSV (*.csv) or a pickled dict (*.pkl)."
            )
        return X_train, y_train, X_test, y_test

    @staticmethod
    def predict(state: torch.Tensor, x: npt.NDArray, k: int = 0) -> int:
        """
        Recursively predict output label given a feature vector x
        of a single observation.
        """
        attributes = state[k]

        if attributes[Attribute.TYPE] == NodeType.CLASSIFIER:
            return attributes[Attribute.CLASS]

        if (
            x[attributes[Attribute.FEATURE].long().item()]
            < attributes[Attribute.THRESHOLD]
        ):
            return Tree.predict(state, x, k=Tree._get_left_child(k))
        else:
            return Tree.predict(state, x, k=Tree._get_right_child(k))

    def _predict(self, x: npt.NDArray, k: int = 0) -> int:
        return Tree.predict(self.state, x, k)

    def plot(self) -> None:
        """
        Plot current state of the tree.
        """
        graph = self._get_graph()

        labels = {}
        node_color = []
        for node in graph:
            x = graph.nodes[node]["x"]
            if x[0] == NodeType.CONDITION:
                labels[node] = rf"$x_{int(x[1].item())}$ < {np.round(x[2].item(), 4)}"
                node_color.append("white")
            else:
                labels[node] = f"C={int(x[3].item())}"
                node_color.append("red")

        nx.draw(
            graph,
            graphviz_layout(graph, prog="dot"),
            labels=labels,
            node_color=node_color,
            with_labels=True,
            node_size=800,
        )
        plt.show()

    @staticmethod
    def _compute_scores(states: torch.Tensor, X: npt.NDArray, y: npt.NDArray) -> dict:
        """
        Computes accuracy and balanced accuracy metrics for a given dataset (X, y)
        based on a tensor of sampled states representing individual trees.

        The metrics are computed in two modes: either as an average of scores calculated
        for individual trees (mean_tree_*), or as a single score calculated on a prediction
        made by the whole ensemble (forest_*), with ensembling done via prediction averaging.

        Args
        ----
        states : Tensor
            Collection of sampled states representing the ensemble.

        X : NDArray
            Feature matrix with dimensionality (n_observations, n_features).

        y : NDArray
            Target vector with dimensionality (n_observations,).

        Returns
        -------
        Dictionary of (metric_name, score) key-value pairs.
        """
        scores = {}
        metrics = {"acc": accuracy_score, "bac": balanced_accuracy_score}

        predictions = np.empty((len(states), len(X)))
        for i, state in enumerate(states):
            for j, x in enumerate(X):
                predictions[i, j] = Tree.predict(state, x)

        for metric_name, metric_function in metrics.items():
            scores[f"mean_tree_{metric_name}"] = np.mean(
                [metric_function(y, y_pred) for y_pred in predictions]
            )
            scores[f"forest_{metric_name}"] = metric_function(
                y, predictions.mean(axis=0).round()
            )

        return scores

    def test(
        self,
        samples: Union[
            TensorType["n_trajectories", "..."], npt.NDArray[np.float32], List
        ],
    ) -> dict:
        """
        Computes a dictionary of metrics, as described in Tree._compute_scores, for
        both training and, if available, test data.

        Args
        ----
        samples : Tensor
            Collection of sampled states representing the ensemble.

        Returns
        -------
        Dictionary of (metric_name, score) key-value pairs.
        """
        result = {}

        for k, v in Tree._compute_scores(samples, self.X_train, self.y_train).items():
            result[f"train_{k}"] = v

        if self.X_test is not None:
            for k, v in Tree._compute_scores(samples, self.X_test, self.y_test).items():
                result[f"test_{k}"] = v

        return result
