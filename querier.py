"""import statements"""
import numpy as np
from sampler import *
import multiprocessing as mp

'''
This script selects sequences to be sent to the oracle for scoring

> Inputs: model extrema (sequences in 0123 format)
> Outputs: sequences to be scored (0123 format)

To-Do:
==> test heruistic logic
==> implement RL model
'''


class querier():
    def __init__(self, params):
        self.params = params

    def buildQuery(self, model, parallel=True):
        """
        select the samples which will be sent to the oracle for scoring
        :param sampleDict:
        :return:
        """
        nQueries = self.params['queries per iter']
        if self.params['query mode'] == 'random':
            '''
            generate query randomly
            '''

            if self.params['variable sample size']:
                samples = []
                while len(samples) < self.params['init dataset length']:
                    for i in range(self.params['min sample length'], self.params['max sample length'] + 1):
                        # samples are integer sequeces of varying lengths
                        samples.extend(np.random.randint(0 + 1, self.params['dict size'] + 1, size=(self.params['queries per iter'] * i, i)))

                    samples = numpy_fillna(np.asarray(samples)).astype(int)  # pad sequences up to maximum length
                    samples = self.filterDuplicates(samples, scores=False)

                samples = samples[:nQueries]  # after shuffle, reduce dataset to desired size, with properly weighted samples
                oracleSamples = samples  # samples are a binary set
            else:  # fixed sample size
                oracleSamples = []
                oracleSamples.extend(np.random.randint(1, self.params['dict size'] + 1, size=(self.params['queries per iter'], self.params['max sample length'])))  # samples are a binary set
                oracleSamples = self.filterDuplicates(oracleSamples, scores=False)
                while len(oracleSamples) < nQueries:
                    oracleSamples.extend(np.random.randint(1, self.params['dict size'] + 1, size=(self.params['queries per iter'], self.params['max sample length'])))  # samples are a binary set
                    oracleSamples = self.filterDuplicates(oracleSamples, scores=False)

            seedInd = 0


        elif (self.params['query mode'] == 'score') or (self.params['query mode'] == 'uncertainty') or (self.params['query mode'] == 'heuristic'):
            '''
            query samples with best good scores, according to our model
            '''
            oracleSamples = []
            if self.params['query mode'] == 'score':
                scoreFunction = [1, 0]  # weighting between score and uncertainty
            elif self.params['query mode'] == 'uncertainty':
                scoreFunction = [0, 1]
            elif self.params['query mode'] == 'heuristic':
                scoreFunction = [0.5, 0.5]  # this can be modulated over time, in principle

            seedInd = 0
            samples = []
            scores = []
            while (len(oracleSamples) < self.params['queries per iter']) and (seedInd < 2):  # until we fill up the query size threshold - flag for if it can't add more for some reason
                self.samplingOutputs = self.runSampling(model, scoreFunction, seedInd, parallel)
                if seedInd == 0:
                    samples = self.samplingOutputs['samples']
                    scores = self.samplingOutputs['scores']
                else:
                    samples = np.concatenate((samples, self.samplingOutputs['samples']), axis=0)
                    scores = np.concatenate((scores, self.samplingOutputs['scores']), axis=0)
                samples, scores = self.filterDuplicates([samples, scores])
                oracleSamples = samples  # don't prune for now
                seedInd += 1


        elif self.params['query mode'] == 'learned':
            raise RuntimeError

        return oracleSamples[:nQueries], seedInd

    def runSampling(self, model, scoreFunction, seedInd, parallel=True, useOracle=False):
        """
        run MCMC sampling
        :param parallel:
        :return:
        """
        gammas = np.logspace(-4, 2, self.params['sampler gammas'])
        if not parallel:
            sampleOutputs = []
            for i in range(len(gammas)):
                mcmcSampler = sampler(self.params, seedInd, scoreFunction)
                sampleOutputs.append(mcmcSampler.sample(model, gammas[i]))
        else:
            mcmcSampler = sampler2(self.params, seedInd, scoreFunction, gammas)
            sampleOutputs = mcmcSampler.sample(model)

        samples = []
        scores = []
        energies = []
        uncertainties = []
        for i in range(len(gammas)):
            samples.extend(sampleOutputs['optimalSamples'][i])
            scores.extend(sampleOutputs['optima'][i])
            energies.extend(sampleOutputs['enAtOptima'][i])
            uncertainties.extend(sampleOutputs['varAtOptima'][i])

        outputs = {
            'samples': np.asarray(samples),
            'scores': np.asarray(scores),
            'energies': np.asarray(energies),
            'uncertainties': np.asarray(uncertainties)
        }

        return outputs

    def filterDuplicates(self, sampleDict, scores=True):
        """
        make sure there are no duplicates in the filtered samples OR in the existing dataset
        if scores == True - we sort all of these as well, otherwise we just look at the samples
        :param samples:
        :return:
        """

        if not scores:  # input is just a list of samples
            samples = sampleDict
            dataset = np.load('datasets/' + self.params['dataset'] + '.npy', allow_pickle=True).item()
            dataset = dataset['samples']
            checkAgainst = np.concatenate((dataset, samples))  # combined dataset to check against

            filteredSamples = []
            for i in range(len(samples)):
                duplicates = 0
                for j in range(len(checkAgainst)):
                    if all(samples[i] == checkAgainst[j]):
                        duplicates += 1

                if duplicates == 1:
                    filteredSamples.append(samples[i])  # keep sequences that appear exactly once
            return np.asarray(filteredSamples)

        else:
            samples = np.asarray(sampleDict[0])
            scores = np.asarray(sampleDict[1])

            dataset = np.load('datasets/' + self.params['dataset'] + '.npy', allow_pickle=True).item()
            dataset = dataset['samples']
            checkAgainst = np.concatenate((dataset, samples))  # combined dataset to check against

            filteredSamples = []
            filteredScores = []

            for i in range(len(samples)):
                duplicates = 0
                for j in range(len(checkAgainst)):
                    if all(samples[i][0:len(checkAgainst[j])] == checkAgainst[j][0:len(samples[i])]) and (len(checkAgainst[j] == len(samples[i]))):
                        duplicates += 1

                if duplicates == 1:
                    filteredSamples.append(samples[i])  # keep sequences that appear exactly once
                    filteredScores.append(scores[i])

            filteredSamples = np.concatenate(np.expand_dims(filteredSamples, 0))
            filteredScores = np.asarray(filteredScores)

            return filteredSamples, filteredScores

    def selectSamples(self, sampleDict):
        """
        identify the samples which are
        1) maximally different
        2) maximally uncertain
        3) minimize the energy
        and pick the top N to ask the oracle about

        if we have used a good score function in sampling, we can do 2 and 3 at the same time, with learned coefficients, combined in the 'scores'
        we may also optimize variance and energy separately, if we feel the score function isn't doing a good job on its own
        :param sampleDict:
        :return:
        """
        # over time, we will adjust coefficients (can be learned) to shift from exploration to optimization
        scalingFactor = 1 - self.params['iteration'] / self.params['pipeline iterations']
        a = scalingFactor  # diversity will get less important
        b = 1 - scalingFactor  # energy will get more important
        c = scalingFactor  # uncertainty will get less important

        samples = sampleDict['samples']
        # for binary input we can use a trivial difference function
        distances = binaryDistance(samples)

        scores = sampleDict['scores']
        energy = sampleDict['energy']
        variance = sampleDict['variance']

        # for now, we can scale these three metrics from 0-1, and take their sum to get a new score out of 3
        scaledDistance = (distances - np.amin(distances)) / np.ptp(distances)
        scaledEnergy = 1 - (energy - np.amin(energy)) / np.ptp(energy)  # smaller is better here, so we invert scaling
        scaledVariance = (variance - np.amin(variance)) / np.ptp(variance)
        scaledScore = a * scaledDistance + b * scaledEnergy + c * scaledVariance
        sortedScore = scaledScore[np.argsort(scaledScore)]  # sort
        sortedSamples = samples[np.argsort(scaledScore)]

        bestSamples = sortedSamples[-self.params['queries per iter']:]
        distance2 = self.binaryDistance(bestSamples)  # in the future we can use this to force a reshuffle if the samples are too similar

        # if len(bestSamples) < self.params['queries per iter']: # if we didn't get enough samples, suggest random ones

        return bestSamples


def askSampler(model, params, seedInd, gamma, scoreFunction, useOracle = False):
    """
    rewritten for sampling in a parallelized fashion using multiprocessing
    needs to be outside the class method for multiprocessing to work    :param params:
    :param i:
    :param scoreFunction:
    :return:
    """
    t0 = time.time()
    mcmcSampler = sampler(params, seedInd, scoreFunction=scoreFunction)
    sampleOutputs = mcmcSampler.sample(model, gamma, useOracle = useOracle)
    tf = time.time()
    #print('Sampler {} finished after {} seconds'.format(gamma, int(tf-t0)))
    return sampleOutputs
