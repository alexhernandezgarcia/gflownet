'''import statements'''


'''
This script selects sequences to be sent to the oracle for scoring

> Inputs: model extrema (sequences in 0123 format)
> Outputs: sequences to be scored (0123 format)

To-Do:
==> test heruistic logic
==> implement RL model
'''

class learner():
    def __init__(self,params):
        self.params = params

    def identifySequences(self, sampleSequences, sampleScores, sampleUncertainty):
        # resample up to 100 lowest energy sequences
        # sequences arrive in sorted minimum energy order so we can just take up to the last 100
        oracleSequences = sampleSequences[-100:]

        return oracleSequences