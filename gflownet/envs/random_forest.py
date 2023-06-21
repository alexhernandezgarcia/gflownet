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
    CONDITION = 0
    CLASSIFIER = 1


class Operator:
    LT = 0
    GTE = 1


class Stage:
    COMPLETE = 0
    LEAF = 1
    FEATURE = 2
    THRESHOLD = 3
    OPERATOR = 4


class ActionType:
    PICK_LEAF = 0
    PICK_FEATURE = 1
    PICK_THRESHOLD = 2
    PICK_OPERATOR = 3


N_ATTRIBUTES = 4


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
        return k * N_ATTRIBUTES + 1, (k + 1) * N_ATTRIBUTES + 1

    @staticmethod
    def _get_parent(k: int) -> int:
        return (k - 1) // 2

    @staticmethod
    def _get_left_child(k: int) -> int:
        return 2 * k + 1

    @staticmethod
    def _get_right_child(k: int) -> int:
        return 2 * k + 2

    def _get_attributes(self, k: int) -> torch.Tensor:
        st, en = Tree._get_start_end(k)

        return self.state[st:en]

    def _pick_leaf(self, k: int) -> None:
        attributes = self._get_attributes(k)

        assert self.state[0] == Stage.COMPLETE
        assert attributes[0] == NodeType.CLASSIFIER
        assert not torch.any(torch.isnan(attributes))
        assert torch.all(attributes[1:3] == -1)

        attributes[0] = NodeType.CONDITION
        attributes[1:4] = -1

        self.state[0] = Stage.LEAF

    def _pick_feature(self, k: int, feature: float) -> None:
        attributes = self._get_attributes(k)

        assert self.state[0] == Stage.LEAF
        assert attributes[0] == NodeType.CONDITION
        assert torch.all(attributes[1:4] == -1)

        attributes[1] = feature

        self.state[0] = Stage.FEATURE

    def _pick_threshold(self, k: int, threshold: float) -> None:
        attributes = self._get_attributes(k)

        assert self.state[0] == Stage.FEATURE
        assert attributes[0] == NodeType.CONDITION
        assert attributes[1] >= 0
        assert torch.all(attributes[2:4] == -1)

        attributes[2] = threshold

        self.state[0] = Stage.THRESHOLD

    def _pick_operator(self, k: int, operator: float) -> None:
        attributes = self._get_attributes(k)

        assert self.state[0] == Stage.THRESHOLD
        assert attributes[0] == NodeType.CONDITION
        assert torch.all(attributes[1:3] >= 0)
        assert attributes[3] == -1

        attributes[3] = operator

        self.state[0] = Stage.OPERATOR

        self._split_leaf(k)

    def _split_leaf(self, k: int) -> None:
        attributes = self._get_attributes(k)

        assert self.state[0] == Stage.OPERATOR
        assert attributes[0] == NodeType.CONDITION
        assert torch.all(attributes[1:4] >= 0)

        k_left = Tree._get_left_child(k)
        k_right = Tree._get_right_child(k)

        if attributes[3] == Operator.LT:
            self._insert_classifier(k_left, output=0)
            self._insert_classifier(k_right, output=1)
        else:
            self._insert_classifier(k_left, output=1)
            self._insert_classifier(k_right, output=0)

        attributes[3] = -1

        self.leafs.remove(k)
        self.state[0] = Stage.COMPLETE

    def _insert_classifier(self, k: int, output: int) -> None:
        attributes = self._get_attributes(k)

        assert torch.all(torch.isnan(attributes))

        attributes[0] = NodeType.CLASSIFIER
        attributes[1:3] = -1
        attributes[3] = output

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
        self, state: Optional[List[int]] = None, done: Optional[bool] = None
    ) -> List[bool]:
        if state is None:
            state = self.state
        if done is None:
            done = self.done

        if done:
            return [True] * self.action_space_dim

        stage = state[0]
        mask = [True] * self.action_space_dim

        if stage == Stage.COMPLETE:
            # In the "complete" stage (in which there are no ongoing micro steps)
            # only valid actions are the ones for picking one of the leafs or EOS.
            for k in self.leafs:
                # Check if splitting the node wouldn't exceed max depth
                if Tree._get_right_child(k) < self.n_nodes:
                    mask[k] = True
            mask[-1] = False
        elif stage == Stage.LEAF:
            # Leaf was picked, only picking the feature is valid.
            pass
        elif stage == Stage.FEATURE:
            # Feature was picked, only picking threshold is valid.
            pass
        elif stage == Stage.THRESHOLD:
            # Threshold was picked, only picking operator is valid.
            pass
        else:
            raise ValueError(f"Unrecognized stage {stage}.")

        return mask

    def get_max_traj_length(self) -> int:
        return self.n_nodes * N_ATTRIBUTES

    def _get_graph(self, graph: Optional[nx.DiGraph] = None, k: int = 0) -> nx.DiGraph:
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
        return from_networkx(self._get_graph())

    def predict(self, x: npt.NDArray, k: int = 0) -> int:
        attributes = self._get_attributes(k)

        if attributes[0] == NodeType.CLASSIFIER:
            return attributes[3]

        if x[attributes[1].long().item()] < attributes[2]:
            return self.predict(x, k=Tree._get_left_child(k))
        else:
            return self.predict(x, k=Tree._get_right_child(k))

    def plot(self) -> None:
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
