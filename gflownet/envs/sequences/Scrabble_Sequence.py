from gflownet.envs.sequences.bidirectional import BidirectionalSequence
from string import ascii_uppercase
class ScrabbleSequence(BidirectionalSequence):
    def __init__(self, right_only=False, left_only=False, sequence_length=4, min_length=4, **kwargs):
        super().__init__(
                tokens=ascii_uppercase[:sequence_length],
                right_only=right_only,
                left_only=left_only,
                min_length=min_length,
                max_length=sequence_length,
                pad_token="_",
                **kwargs
                )

    def states2proxy(self, states):
        proxy_states = []
        for state in states:
            proxy_states.append(self.state2readable(state))

        return proxy_states

                
