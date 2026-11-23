import torch
from gflownet.proxy.base import Proxy

class SequenceScrabble_score(Proxy):
    def __init__(self, values=None, pattern=None, pattern_value=10, default_value =0.01, **kwargs):
        super().__init__(**kwargs)
        self.values = dict(values or {})
        self.default_value = default_value
        self.pattern = pattern
        self.pattern_value = pattern_value

    def __call__(self, states):
        if self.pattern is None:
            scores = [
                self.values.get(state, self.default_value)
                for state in states
            ]
        else:
            scores = [
                self.pattern_value if self.pattern in state else self.default_value
                for state in states
            ]

        return torch.tensor(scores, device=self.device, dtype=self.float)
