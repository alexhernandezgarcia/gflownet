from enum import Enum
from typing import Optional, Union

import networkx as nx
import numpy as np
import numpy.typing as npt

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


class RandomForest(GFlowNetEnv):
    def __init__(self, X, y, **kwargs):
        self.X = X
        self.y = y
        self.state = nx.DiGraph()

        super().__init__(**kwargs)
