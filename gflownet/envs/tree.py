from collections import Counter
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
import torch_geometric as pyg
from networkx.drawing.nx_pydot import graphviz_layout
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


# Number of attributes encoding each node; see Tree._get_attributes.
N_ATTRIBUTES = 5


class Tree(GFlowNetEnv):
    """
    GFlowNet environment representing a decision tree.

    Constructing a tree consists of a combination of macro steps (picking a leaf
    to split using a given feature, threshold and operator), which are divided
    into a series of consecutive micro steps (1 - pick a leaf, 2 - pick a feature,
    3 - pick a threshold, 4 - pick an operator). A consequence of that is, as long
    as a macro step is not in progress, the tree constructed so far is always a
    valid decision tree, which means that forward-looking loss etc. can be used.

    Internally, the tree is represented as a fixed-size tensor (thus, specifying
    the maximum depth is required), with nodes indexed from k = 0 to 2**max_depth - 2,
    and each node containing a 5-element attribute tensor (see _get_attributes for
    details). The nodes are indexed from top left to bottom right, as follows:

                0
        1               2
    3       4       5       6
    """

    def __init__(
        self,
        X: Optional[npt.NDArray] = None,
        y: Optional[npt.NDArray] = None,
        data_path: Optional[str] = None,
        max_depth: int = 10,
        threshold_components: int = 1,
        **kwargs,
    ):
        """
        Attributes
        ----------
        X : np.array
            Input dataset, with dimensionality (n_observations, n_features). It may be
            None if a data set is provided via data_path.

        y : np.array
            Target labels, with dimensionality (n_observations,). It may be
            None if a data set is provided via data_path.

        data_path : str
            A path to a data set, with the following options:
            - *.pkl: Pickled dict with X and y variables.
            - *.csv: CSV with M columns where the first (M - 1) columns will be taken
              to construct the input X, and column M-th will be the target y.
            Ignored if X and y are not None.

        max_depth : int
            Maximum depth of a tree.

        threshold_components : int
            The number of mixture components that will be used for sampling
            the threshold.
        """
        if X is not None and y is not None:
            self.X = X
            self.y = y
        elif data_path is not None:
            self.X, self.y = Tree._load_dataset(data_path)
        else:
            raise ValueError(
                "A Tree must be initialised with a data set. X, y and data_path cannot "
                "be all None"
            )
        self.max_depth = max_depth
        self.components = threshold_components
        self.leaves = set()

        # Source will contain information about the current stage (on the 0-th position),
        # and up to 2**max_depth - 1 nodes, each with N_ATTRIBUTES attributes, for a total of
        # 1 + N_ATTRIBUTES * (2**max_depth - 1) values.
        self.n_nodes = 2**max_depth - 1
        self.source = torch.full((N_ATTRIBUTES * self.n_nodes + 1,), torch.nan)
        self.source[0] = Stage.COMPLETE

        # End-of-sequence action.
        self.eos = (-1, -1)

        super().__init__(**kwargs)

        self._insert_classifier(k=0, output=int(Counter(self.y).most_common()[0][0]))

    @staticmethod
    def _get_start_end(k: int) -> Tuple[int, int]:
        """
        Get start and end index of attribute tensor encoding k-th node in self.state.
        """
        return k * N_ATTRIBUTES + 1, (k + 1) * N_ATTRIBUTES + 1

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

    def _get_attributes(
        self, k: int, state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns a 5-element tensor of attributes for k-th node. The encoded values are:
        0 - node type (condition or classifier),
        1 - index of the feature used for splitting (condition node only, -1 otherwise),
        2 - decision threshold (condition node only, -1 otherwise),
        3 - class output (classifier node only, -1 otherwise), in the case of < operator
            the left child will have class = 0, and the right child will have class = 1;
            the opposite for the >= operator,
        4 - whether the node has active status (1 if node was picked and the macro step
            didn't finish yet, 0 otherwise).
        """
        if state is None:
            state = self.state

        st, en = Tree._get_start_end(k)

        return state[st:en]

    def _pick_leaf(self, k: int) -> None:
        """
        Select one of the leaves (classifier nodes) that will be split, and initiate
        macro step.
        """
        attributes = self._get_attributes(k)

        assert self.state[0] == Stage.COMPLETE
        assert attributes[0] == NodeType.CLASSIFIER
        assert not torch.any(torch.isnan(attributes))
        assert torch.all(attributes[1:3] == -1)
        assert attributes[4] == Status.INACTIVE

        attributes[0] = NodeType.CONDITION
        attributes[1:4] = -1
        attributes[4] = Status.ACTIVE

        self.state[0] = Stage.LEAF

    def _pick_feature(self, k: int, feature: float) -> None:
        """
        Select the feature on which currently selected leaf will be split.
        """
        attributes = self._get_attributes(k)

        assert self.state[0] == Stage.LEAF
        assert attributes[0] == NodeType.CONDITION
        assert torch.all(attributes[1:4] == -1)
        assert attributes[4] == Status.ACTIVE

        attributes[1] = feature

        self.state[0] = Stage.FEATURE

    def _pick_threshold(self, k: int, threshold: float) -> None:
        """
        Select the threshold for splitting the currently selected leaf ond
        the selected feature.
        """
        attributes = self._get_attributes(k)

        assert self.state[0] == Stage.FEATURE
        assert attributes[0] == NodeType.CONDITION
        assert attributes[1] >= 0
        assert torch.all(attributes[2:4] == -1)
        assert attributes[4] == Status.ACTIVE

        attributes[2] = threshold

        self.state[0] = Stage.THRESHOLD

    def _pick_operator(self, k: int, operator: float) -> None:
        """
        Select the operator (< or >=) for splitting the currently selected
        left, feature and threshold, temporarily encode it in attributes,
        and initiate final splitting.
        """
        attributes = self._get_attributes(k)

        assert self.state[0] == Stage.THRESHOLD
        assert attributes[0] == NodeType.CONDITION
        assert torch.all(attributes[1:3] >= 0)
        assert attributes[3] == -1
        assert attributes[4] == Status.ACTIVE

        attributes[3] = operator

        self.state[0] = Stage.OPERATOR

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
        attributes = self._get_attributes(k)

        assert self.state[0] == Stage.OPERATOR
        assert attributes[0] == NodeType.CONDITION
        assert torch.all(attributes[1:4] >= 0)
        assert attributes[4] == Status.ACTIVE

        k_left = Tree._get_left_child(k)
        k_right = Tree._get_right_child(k)

        if attributes[3] == Operator.LT:
            self._insert_classifier(k_left, output=0)
            self._insert_classifier(k_right, output=1)
        else:
            self._insert_classifier(k_left, output=1)
            self._insert_classifier(k_right, output=0)

        attributes[3] = -1
        attributes[4] = Status.INACTIVE

        self.leaves.remove(k)
        self.state[0] = Stage.COMPLETE

    def _insert_classifier(self, k: int, output: int) -> None:
        """
        Replace attributes of k-th node with those of a classifier node.
        """
        attributes = self._get_attributes(k)

        assert torch.all(torch.isnan(attributes))

        attributes[0] = NodeType.CLASSIFIER
        attributes[1:3] = -1
        attributes[3] = output
        attributes[4] = Status.INACTIVE

        self.leaves.add(k)

    def get_action_space(self) -> List[Tuple[int, int]]:
        """
        Actions are a tuple containing:
            1) action type:
                0 - pick leaf to split,
                1 - pick feature,
                2 - pick threshold,
                3 - pick operator,
            2) node index.
            3) action value

        Note: The action space consists of only the discrete (fixed) part of the
        actions, that is the first two elements of tge tuple (action type, node index),
        since action value can vary and be continuous-valued.
        """
        actions = [(t, k) for t in range(4) for k in range(self.n_nodes)]
        actions.append(self.eos)

        return actions

    def step(
        self, action: Tuple[int, int, float], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int, int, float], bool]:
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
        do_step, self.state, _ = self._pre_step(
            action[:2], skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, False

        self.n_actions += 1

        if action != self.eos:
            action_type, k, action_value = action

            if action_type == ActionType.PICK_LEAF:
                self._pick_leaf(k)
            elif action_type == ActionType.PICK_FEATURE:
                self._pick_feature(k, action_value)
            elif action_type == ActionType.PICK_THRESHOLD:
                self._pick_threshold(k, action_value)
            elif action_type == ActionType.PICK_OPERATOR:
                self._pick_operator(k, action_value)
            else:
                raise NotImplementedError(f"Unrecognized action type: {action_type}.")

            return self.state, action, True
        else:
            self.done = True
            return self.state, action, True

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
        import ipdb; ipdb.set_trace()
        device = policy_outputs.device
        mask_states_sample = ~mask_invalid_actions.flatten()
        n_states = policy_outputs.shape[0]
        # Sample angle increments
        angles = torch.zeros(n_states, self.n_dim).to(device)
        logprobs = torch.zeros(n_states, self.n_dim).to(device)
        if torch.any(mask_states_sample):
            if sampling_method == "uniform":
                distr_angles = Uniform(
                    torch.zeros(len(ns_range_noeos)),
                    2 * torch.pi * torch.ones(len(ns_range_noeos)),
                )
            elif sampling_method == "policy":
                mix_logits = policy_outputs[mask_states_sample, 0::3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                mix = Categorical(logits=mix_logits)
                locations = policy_outputs[mask_states_sample, 1::3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                concentrations = policy_outputs[mask_states_sample, 2::3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                vonmises = VonMises(
                    locations,
                    torch.exp(concentrations) + self.vonmises_min_concentration,
                )
                distr_angles = MixtureSameFamily(mix, vonmises)
            angles[mask_states_sample] = distr_angles.sample()
            logprobs[mask_states_sample] = distr_angles.log_prob(
                angles[mask_states_sample]
            )
        logprobs = torch.sum(logprobs, axis=1)
        # Build actions
        actions_tensor = torch.inf * torch.ones(
            angles.shape, dtype=self.float, device=device
        )
        actions_tensor[mask_states_sample, :] = angles[mask_states_sample]
        actions = [tuple(a.tolist()) for a in actions_tensor]
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
        device = policy_outputs.device
        mask_states_sample = ~mask_invalid_actions.flatten()
        n_states = policy_outputs.shape[0]
        logprobs = torch.zeros(n_states, self.n_dim).to(device)
        if torch.any(mask_states_sample):
            mix_logits = policy_outputs[mask_states_sample, 0::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            mix = Categorical(logits=mix_logits)
            locations = policy_outputs[mask_states_sample, 1::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            concentrations = policy_outputs[mask_states_sample, 2::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            vonmises = VonMises(
                locations,
                torch.exp(concentrations) + self.vonmises_min_concentration,
            )
            distr_angles = MixtureSameFamily(mix, vonmises)
            logprobs[mask_states_sample] = distr_angles.log_prob(
                actions[mask_states_sample]
            )
        logprobs = torch.sum(logprobs, axis=1)
        return logprobs

    @staticmethod
    def _find_leaves(state: torch.Tensor) -> List[int]:
        """
        Compute indices of leaves from a state.
        """
        leaves = [x.item() for x in torch.where(state[1::5] == NodeType.CLASSIFIER)[0]]

        return leaves

    @staticmethod
    def _find_active(state: torch.Tensor) -> int:
        """
        Compute index of the (only) active node. Assumes that active node exists
        (that we are in the middle of a macro step).
        """
        k = torch.where(state[N_ATTRIBUTES::N_ATTRIBUTES] == Status.ACTIVE)[0].item()

        return k

    def get_mask_invalid_actions_forward(
        self, state: Optional[torch.Tensor] = None, done: Optional[bool] = None
    ) -> List[bool]:
        if state is None:
            state = self.state
            leaves = self.leaves
        else:
            leaves = Tree._find_leaves(state)
        if done is None:
            done = self.done

        if done:
            return [True] * self.action_space_dim

        stage = state[0]
        mask = [True] * self.action_space_dim

        if stage == Stage.COMPLETE:
            # In the "complete" stage (in which there are no ongoing micro steps)
            # only valid actions are the ones for picking one of the leaves or EOS.
            for k in leaves:
                # Check if splitting the node wouldn't exceed max depth.
                if Tree._get_right_child(k) < self.n_nodes:
                    mask[k] = False
            mask[-1] = False
        else:
            k = Tree._find_active(state)

            if stage == Stage.LEAF:
                # Leaf was picked, only picking the feature is valid.
                mask[k + self.n_nodes] = False
            elif stage == Stage.FEATURE:
                # Feature was picked, only picking the threshold is valid.
                mask[k + 2 * self.n_nodes] = False
            elif stage == Stage.THRESHOLD:
                # Threshold was picked, only picking the operator is valid.
                mask[k + 3 * self.n_nodes] = False
            else:
                raise ValueError(f"Unrecognized stage {stage}.")

        return mask

    def get_parents(
        self,
        state: Optional[torch.Tensor] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        if state is None:
            state = self.state
            leaves = self.leaves
        else:
            leaves = Tree._find_leaves(state)
        if done is None:
            done = self.done

        if done:
            return [state], [self.eos]

        stage = state[0]
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

                # Revert stage (to "threshold": we skip "operator" because from it,
                # finalizing splitting should be automatically executed).
                parent[0] = Stage.THRESHOLD

                # Reset children attributes.
                attributes_left = self._get_attributes(k_left, parent)
                attributes_left[:] = torch.nan
                attributes_right = self._get_attributes(k_right, parent)
                attributes_right[:] = torch.nan

                # Revert parent attributes to the previous state.
                attributes_parent = self._get_attributes(k_parent, parent)
                action = (
                    ActionType.PICK_OPERATOR,
                    k_parent,
                    attributes_parent[3].item(),
                )
                attributes_parent[3] = -1
                attributes_parent[4] = Status.ACTIVE

                parents.append(parent)
                actions.append(action)
        else:
            k = Tree._find_active(state)

            if stage == Stage.LEAF:
                # Reverse self._pick_leaf.
                for output in [0, 1]:
                    parent = state.clone()
                    attributes = self._get_attributes(k, parent)

                    parent[0] = Stage.COMPLETE
                    attributes[0] = NodeType.CLASSIFIER
                    attributes[1:3] = -1
                    attributes[3] = output
                    attributes[4] = Status.INACTIVE

                    parents.append(parent)
                    actions.append(action)
            elif stage == Stage.FEATURE:
                # Reverse self._pick_feature.
                parent = state.clone()
                attributes = self._get_attributes(k, parent)

                parent[0] = Stage.LEAF
                attributes[1] = -1

                parents.append(parent)
                actions.append(action)
            elif stage == Stage.THRESHOLD:
                # Reverse self._pick_threshold.
                parent = state.clone()
                attributes = self._get_attributes(k, parent)

                parent[0] = Stage.FEATURE
                attributes[2] = -1

                parents.append(parent)
                actions.append(action)
            else:
                raise ValueError(f"Unrecognized stage {stage}.")

        return parents, actions

    def get_max_traj_length(self) -> int:
        return self.n_nodes * N_ATTRIBUTES

    def _get_graph(self, graph: Optional[nx.DiGraph] = None, k: int = 0) -> nx.DiGraph:
        """
        Recursively convert self.state into a networkx directional graph.
        """
        if graph is None:
            graph = nx.DiGraph()

        attributes = self._get_attributes(k)
        graph.add_node(k, x=attributes)

        if attributes[0] != NodeType.CLASSIFIER:
            k_left = Tree._get_left_child(k)
            if not torch.any(torch.isnan(self._get_attributes(k_left))):
                self._get_graph(graph, k=k_left)
                graph.add_edge(k, k_left)

            k_right = Tree._get_right_child(k)
            if not torch.any(torch.isnan(self._get_attributes(k_right))):
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
        from pathlib import Path

        data_path = Path(data_path)
        if data_path.suffix == ".csv":
            import pandas as pd

            df = pd.read_csv(data_path)
            X = df.iloc[:, 0:-1].values
            y = df.iloc[:, -1].values
        elif data_path.suffix == ".pkl":
            import pickle

            with open(data_path, "rb") as f:
                dct = pickle.load(f)
                X = dct["X"]
                y = dct["y"]
        else:
            raise ValueError(
                "data_path must be a CSV (*.csv) or a pickled dict (*.pkl)."
            )
        return X, y

    def predict(self, x: npt.NDArray, k: int = 0) -> int:
        """
        Recursively predict output label given a feature vector x
        of a single observation.
        """
        attributes = self._get_attributes(k)

        if attributes[0] == NodeType.CLASSIFIER:
            return attributes[3]

        if x[attributes[1].long().item()] < attributes[2]:
            return self.predict(x, k=Tree._get_left_child(k))
        else:
            return self.predict(x, k=Tree._get_right_child(k))

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
