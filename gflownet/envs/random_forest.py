from enum import Enum
from typing import Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
import torch_geometric as pyg
from networkx.drawing.nx_pydot import graphviz_layout
from torch_geometric.utils.convert import from_networkx

from gflownet.envs.base import GFlowNetEnv


class Operator(Enum):
    LT = -1
    GTE = 1


class Node:
    def __init__(self, output: int, parent: Optional["Node"] = None):
        self.content: Union[Classifier, Condition] = Classifier(output)
        self.parent: Optional[Node] = parent
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.index: Optional[int] = None

    def split(self, feature: int, threshold: float, operator: Operator) -> None:
        assert isinstance(self.content, Classifier)

        self.content = Condition(feature, threshold, operator)

        if self.content.operator == Operator.LT:
            self.left = Node(output=0, parent=self)
            self.right = Node(output=1, parent=self)
        elif self.content.operator == Operator.GTE:
            self.left = Node(output=1, parent=self)
            self.right = Node(output=0, parent=self)
        else:
            raise NotImplementedError(f"Unrecognized operator {self.content.operator}.")

    def predict(self, x: npt.NDArray) -> int:
        if isinstance(self.content, Classifier):
            return self.content.output

        if x[self.content.feature] < self.content.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def attributes(self) -> npt.NDArray:
        if isinstance(self.content, Classifier):
            node_type = 0
            feature = -1
            threshold = -1
            output = self.content.output
        else:
            node_type = 1
            feature = self.content.feature
            threshold = self.content.threshold
            output = -1

        return np.array([node_type, feature, threshold, output])


class Classifier:
    def __init__(self, output: int):
        self.output = output


class Condition:
    def __init__(self, feature: int, threshold: float, operator: Operator):
        self.feature = feature
        self.threshold = threshold
        self.operator = operator


class Tree(GFlowNetEnv):
    def __init__(self, X, y, **kwargs):
        self.X = X
        self.y = y
        self.state = nx.DiGraph()

        self.root = Node(output=1)  # TODO: use most frequent class
        self.leafs = {}
        self._insert_node(self.root)

        # super().__init__(**kwargs)

    def _split_node(
        self, node: Node, feature: int, threshold: float, operator: Operator
    ) -> None:
        node.split(feature, threshold, operator)

        self.state.nodes[node.index]["x"] = node.attributes()

        self._insert_node(node.left)
        self._insert_node(node.right)

        self.leafs.pop(node.index)

    def _insert_node(self, node: Node) -> None:
        node.index = len(self.state)

        self.state.add_node(node.index, x=node.attributes())
        if node.parent is not None:
            self.state.add_edge(node.parent.index, node.index)

        self.leafs[node.index] = node

    def _to_pyg(self) -> pyg.data.Data:
        return from_networkx(self.state)

    def plot(self) -> None:
        labels = {}
        node_color = []
        for node in self.state:
            x = self.state.nodes[node]["x"]
            if x[0] == 1:
                labels[node] = fr"$x_{int(x[1])}$ < {x[2]}"
                node_color.append("white")
            else:
                labels[node] = f"C={x[3]}"
                node_color.append("red")

        nx.draw(
            self.state,
            graphviz_layout(self.state, prog="dot"),
            labels=labels,
            node_color=node_color,
            with_labels=True,
            node_size=800,
        )
        plt.show()
