from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import torch
import torch_geometric as pyg
from networkx.drawing.nx_pydot import graphviz_layout
from torch_geometric.utils.convert import from_networkx

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
        the only allowed action is to pick one of the leafs for splitting.
    1 - leaf, indicates that a leaf was picked for splitting, and the only
        allowed action is picking a feature on which it will be split.
    2 - feature, indicates that a feature was picked, and the only allowed
        action is picking a threshold for splitting.
    3 - threshold, indicates that a threshold was picked, and the only
        allowed action is picking an operator.
    4 - operator, indicates that operator was picked. The only allowed
        action from here is finalizing the splitting process and spawning
        two new leafs, which should be done automatically, upon which
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


# Number of attributes encoding each node; see Tree._get_attributes
N_ATTRIBUTES = 5


class Tree(GFlowNetEnv):
    def __init__(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        max_depth: int = 10,
        threshold_components: int = 1,
        **kwargs,
    ):
        self.X = X
        self.y = y
        self.max_depth = max_depth
        self.components = threshold_components
        self.leafs = set()

        # Source will contain information about the current stage (on the 0-th position),
        # and up to 2**max_depth - 1 nodes, each with N_ATTRIBUTES attributes, for a total of
        # 1 + N_ATTRIBUTES * (2**max_depth - 1) values.
        self.n_nodes = 2**max_depth - 1
        self.source = torch.full((N_ATTRIBUTES * self.n_nodes + 1,), torch.nan)
        self.source[0] = Stage.COMPLETE

        # End-of-sequence action
        self.eos = (-1, -1)

        super().__init__(**kwargs)

        # TODO: use most frequent class as the output
        self._insert_classifier(k=0, output=1)

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

    def _get_attributes(self, k: int) -> torch.Tensor:
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
        st, en = Tree._get_start_end(k)

        return self.state[st:en]

    def _pick_leaf(self, k: int) -> None:
        """
        Select one of the leafs (classifier nodes) that will be split, and initiate
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
        and operator: transformer the leaf into a condition node and
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

        self.leafs.remove(k)
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

        self.leafs.add(k)

    def get_action_space(self) -> List[Tuple[int, int]]:
        """
        Actions are a tuple containing:
            1) action type:
                0 - pick leaf to split,
                1 - pick feature,
                2 - pick threshold,
                3 - pick operator,
            2) node index.
        """
        actions = [(t, k) for t in range(4) for k in range(self.n_nodes)]
        actions.append(self.eos)

        return actions

    def step(
        self, action: Tuple[int, int, float]
    ) -> Tuple[List[int], Tuple[int, int, float], bool]:
        # If action not found in action space raise an error
        if action[:2] not in self.action_space:
            raise ValueError(
                f"Tried to execute action {action} not present in action space."
            )
        else:
            action_idx = self.action_space.index(action[:2])

        # If action is in invalid mask, exit immediately
        if self.get_mask_invalid_actions_forward()[action_idx]:
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

    def get_mask_invalid_actions_forward(
        self, state: Optional[torch.Tensor] = None, done: Optional[bool] = None
    ) -> List[bool]:
        if state is None:
            state = self.state
            leafs = self.leafs
        else:
            leafs = [
                x.item() for x in torch.where(state[1::5] == NodeType.CLASSIFIER)[0]
            ]
        if done is None:
            done = self.done

        if done:
            return [True] * self.action_space_dim

        stage = state[0]
        mask = [True] * self.action_space_dim

        if stage == Stage.COMPLETE:
            # In the "complete" stage (in which there are no ongoing micro steps)
            # only valid actions are the ones for picking one of the leafs or EOS.
            for k in leafs:
                # Check if splitting the node wouldn't exceed max depth
                if Tree._get_right_child(k) < self.n_nodes:
                    mask[k] = False
            mask[-1] = False
        else:
            # Find index of the (only) active node.
            k = torch.where(state[N_ATTRIBUTES::N_ATTRIBUTES] == Status.ACTIVE)[
                0
            ].item()

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
