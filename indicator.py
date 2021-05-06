'''import statements'''
import numpy as np

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

    def identifySequences(self, sampleDict):
        '''
        select the samples which will be sent to the oracle for scoring
        :param sampleDict:
        :return:
        '''
        filteredDict = self.filterDuplicates(sampleDict) # cut any samples already in the dataset or duplicated from sampling
        oracleSamples = self.selectSamples(filteredDict) # we want samples which are a balance of 1) maximally different, 2) maximally uncertain, 3) minimize energy

        return oracleSamples


    def filterDuplicates(self,sampleDict):
        '''
        make sure there are no duplicates in the filtered samples OR in the existing dataset
        :param samples:
        :return:
        '''
        samples = sampleDict['samples']
        scores = sampleDict['scores']
        energy = sampleDict['energy']
        variance = sampleDict['variance']

        dataset = np.load('datasets/' + self.params['dataset'] + '.npy', allow_pickle=True).item()
        dataset = dataset['samples']
        checkAgainst = np.concatenate((dataset,samples)) # combined dataset to check against

        filteredSamples = []
        filteredScores = []
        filteredEnergy = []
        filteredVariance = []
        for i in range(len(samples)):
            duplicates = 0
            for j in range(len(checkAgainst)):
                if all(samples[i] == checkAgainst[j]):
                    duplicates += 1

            if duplicates == 1:
                filteredSamples.append(samples[i]) # keep sequences that appear exactly once
                filteredScores.append(scores[i])
                filteredEnergy.append(energy[i])
                filteredVariance.append(variance[i])

        filteredDict = {}
        filteredDict['samples'] = np.concatenate(np.expand_dims(filteredSamples,0))
        filteredDict['scores'] = np.asarray(filteredScores)
        filteredDict['energy'] = np.asarray(filteredEnergy)
        filteredDict['variance'] = np.asarray(filteredVariance)

        return filteredDict


    def selectSamples(self, sampleDict):
        '''
        identify the samples which are
        1) maximally different
        2) maximally uncertain
        3) minimize the energy
        and pick the top N to ask the oracle about

        if we have used a good score function in sampling, we can do 2 and 3 at the same time, with learned coefficients, combined in the 'scores'
        we may also optimize variance and energy separately, if we feel the score function isn't doing a good job on its own
        :param sampleDict:
        :return:
        '''
        # over time, we will adjust coefficients (can be learned) to shift from exploration to optimization
        scalingFactor = 1 - self.params['iteration']/self.params['pipeline iterations']
        a = scalingFactor # diversity will get less important
        b = 1-scalingFactor # energy will get more important
        c = scalingFactor # uncertainty will get less important

        samples = sampleDict['samples']
        # for binary input we can use a trivial difference function
        distances = self.binaryDistance(samples)

        scores = sampleDict['scores']
        energy = sampleDict['energy']
        variance = sampleDict['variance']

        # for now, we can scale these three metrics from 0-1, and take their sum to get a new score out of 3
        scaledDistance = (distances - np.amin(distances))/np.ptp(distances)
        scaledEnergy = 1-(energy - np.amin(energy))/np.ptp(energy) # smaller is better here, so we invert scaling
        scaledVariance = (variance - np.amin(variance))/np.ptp(variance)
        scaledScore = a * scaledDistance +  b * scaledEnergy + c * scaledVariance
        sortedScore = scaledScore[np.argsort(scaledScore)] # sort
        sortedSamples = samples[np.argsort(scaledScore)]

        bestSamples = sortedSamples[-self.params['queries per iter']:]
        distance2 = self.binaryDistance(bestSamples) # in the future we can use this to force a reshuffle if the samples are too similar

        #if len(bestSamples) < self.params['queries per iter']: # if we didn't get enough samples, suggest random ones

        return bestSamples


    def binaryDistance(self,samples):
        '''
        compute simple sum of distances between sample vectors
        :param samples:
        :return:
        '''
        distances = np.zeros(len(samples))
        for i in range(len(samples)):
            distances[i] = np.sum(samples[i] != samples)

        return distances/samples.shape[0]/samples.shape[1]