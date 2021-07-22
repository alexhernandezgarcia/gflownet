'''import statements'''
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
    def __init__(self,params):
        self.params = params


    def buildQuery(self, model):
        '''
        select the samples which will be sent to the oracle for scoring
        :param sampleDict:
        :return:
        '''
        nQueries = self.params['queries per iter']
        if self.params['query mode'] == 'random':
            '''
            generate query randomly
            '''
            oracleSamples = []
            while len(oracleSamples) < nQueries:
                randomSamples = np.random.randint(0,2,size=(self.params['queries per iter'], self.params['sample length']))
                oracleSamples.extend(randomSamples)
                oracleSamples = self.filterDuplicates(oracleSamples,scores=False)


        elif (self.params['query mode'] == 'score') or (self.params['query mode'] == 'uncertainty') or (self.params['query mode'] == 'heuristic'):
            '''
            query samples with best good scores, according to our model
            '''
            oracleSamples = []
            if self.params['query mode'] == 'score':
                scoreFunction = [1 ,0] # weighting between score and uncertainty
            elif self.params['query mode'] == 'uncertainty':
                scoreFunction = [0, 1]
            elif self.params['query mode'] == 'heuristic':
                scoreFunction = [0.5, 0.5] # this can be modulated over time, in principle

            seedInd = 0
            samples = []
            scores = []
            while (len(oracleSamples) < self.params['queries per iter']) and (seedInd < 100): # until we fill up the query size threshold - flag for if it can't add more for some reason
                samplingOutputs = self.runSampling(model, scoreFunction, seedInd)
                if seedInd == 0:
                    samples = samplingOutputs['samples']
                    scores = samplingOutputs['scores']
                else:
                    samples = np.concatenate((samples,samplingOutputs['samples']),axis=0)
                    scores = np.concatenate((scores, samplingOutputs['scores']), axis=0)
                samples, scores = self.filterDuplicates([samples, scores])
                oracleSamples = samples # don't prune for now
                seedInd += 1

            print('Finished {} iterations of sampling'.format(seedInd))

        elif self.params['query mode'] == 'learned':
            raise RuntimeError

        return oracleSamples[:nQueries]


    def runSampling(self, model, scoreFunction, seedInd, parallel=True):
        '''
        run MCMC sampling
        :param parallel:
        :return:
        '''
        if not parallel:
            gammas = np.logspace(-3, 1, self.params['sampler gammas'])
            sampleOutputs = []
            for i in range(len(gammas)):
                mcmcSampler = sampler(self.params, seedInd, scoreFunction)
                sampleOutputs.append(mcmcSampler.sample(model, gammas[i]))
        else:
            if self.params['device'] == 'local':
                nHold = 4 # keep some CPUS if running locally
            else:
                nHold = 1
            cpus = os.cpu_count() - nHold  # np.min((os.cpu_count()-2,params['runs']))
            if cpus > self.params['sampler gammas']:
                nGammas = cpus
            pool = mp.Pool(cpus)
            gammas = np.logspace(-4, 1, nGammas)
            for i in range(int(np.ceil(nGammas / cpus))):
                output = [pool.apply_async(askSampler, args=[model, self.params, seedInd, gammas[j], scoreFunction]) for j in range(nGammas)]
                if i > 0:
                    for j in range(cpus):
                        sampleOutputs.append(output[j].get())
                else:
                    sampleOutputs = [output[i].get() for i in range(cpus)]

        samples = []
        scores = []
        for i in range(len(sampleOutputs)):
            samples.extend(sampleOutputs[i]['optimalSamples'])
            scores.extend(sampleOutputs[i]['optima'])

        outputs = {
            'samples' : np.asarray(samples),
            'scores' :  np.asarray(scores)
        }

        return outputs


    def filterDuplicates(self, sampleDict, scores = True):
        '''
        make sure there are no duplicates in the filtered samples OR in the existing dataset
        if scores == True - we sort all of these as well, otherwise we just look at the samples
        :param samples:
        :return:
        '''

        if not scores: # input is just a list of samples
            samples = sampleDict
            dataset = np.load('datasets/' + self.params['dataset'] + '.npy', allow_pickle=True).item()
            dataset = dataset['samples']
            checkAgainst = np.concatenate((dataset,samples)) # combined dataset to check against

            filteredSamples = []
            for i in range(len(samples)):
                duplicates = 0
                for j in range(len(checkAgainst)):
                    if all(samples[i] == checkAgainst[j]):
                        duplicates += 1

                if duplicates == 1:
                    filteredSamples.append(samples[i]) # keep sequences that appear exactly once
            return filteredSamples

        else:
            samples = np.asarray(sampleDict[0])
            scores = np.asarray(sampleDict[1])


            dataset = np.load('datasets/' + self.params['dataset'] + '.npy', allow_pickle=True).item()
            dataset = dataset['samples']
            checkAgainst = np.concatenate((dataset,samples)) # combined dataset to check against

            filteredSamples = []
            filteredScores = []

            for i in range(len(samples)):
                duplicates = 0
                for j in range(len(checkAgainst)):
                    if all(samples[i] == checkAgainst[j]):
                        duplicates += 1

                if duplicates == 1:
                    filteredSamples.append(samples[i]) # keep sequences that appear exactly once
                    filteredScores.append(scores[i])

            filteredSamples = np.concatenate(np.expand_dims(filteredSamples,0))
            filteredScores = np.asarray(filteredScores)

            return filteredSamples, filteredScores


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

        return distances


def askSampler(model, params, seedInd, gamma, scoreFunction):
    '''
    rewritten for sampling in a parallelized fashion
    needs to be outside the class method for multiprocessing to work    :param params:
    :param i:
    :param scoreFunction:
    :return:
    '''
    mcmcSampler = sampler(params, seedInd, scoreFunction=scoreFunction)
    sampleOutputs = mcmcSampler.sample(model, gamma)
    return sampleOutputs