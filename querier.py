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

    def buildQuery(self, model, statusDict, energySampleDict):
        """
        select the samples which will be sent to the oracle for scoring
        :param sampleDict:
        :return:
        """
        nQueries = self.params.queries_per_iter
        if self.params.query_mode == 'random':
            '''
            generate query randomly
            '''
            samples = generateRandomSamples(self.params.queries_per_iter, [self.params.min_sample_length,self.params.max_sample_length], self.params.dict_size, variableLength = self.params.variable_sample_length, oldDatasetPath = 'datasets/' + self.params.dataset + '.npy')

        else:
            if self.params.query_mode == 'learned':
                raise RuntimeError("No learned models have been implemented!")
                # TODO implement learned model
                # TODO upgrade sampler

            else:
                '''
                query samples with best good scores, according to our model and a scoring function
                
                '''

                # generate candidates
                if self.params.query_mode == 'energy':
                    self.sampleDict = energySampleDict
                else:
                    self.sampleForQuery(model, statusDict['iter'])

                samples = self.sampleDict['samples']
                scores = self.sampleDict['scores']
                uncertainties = self.sampleDict['uncertainties']
                samples, inds = filterDuplicateSamples(samples, oldDatasetPath='datasets/' + self.params.dataset + '.npy', returnInds=True)
                scores = scores[inds]

            # create batch from candidates
            # agglomerative clustering
            clusters, clusterScores, clusterVars = doAgglomerativeClustering(samples, scores, uncertainties, cutoff=self.params.minima_dist_cutoff)
            clusterSizes, avgClusterScores, minCluster, avgClusterVars, minClusterVars, minClusterSamples = clusterAnalysis(clusters, clusterScores, clusterVars)
            samples = minClusterSamples

            # alternatively, simple exclusion
            #bestInds = sortTopXSamples(samples[np.argsort(scores)], nSamples=len(samples), distCutoff=self.params.minima_dist_cutoff)  # sort out the best, and at least minimally distinctive samples
            #samples = samples[bestInds]

            # alternatively, just take whatever is given
            # samples = samples[np.argsort(scores)]

            while len(samples) < nQueries: # if we don't have enough samples from mcmc, add random ones to pad out the
                randomSamples = generateRandomSamples(1000, [self.params.min_sample_length,self.params.max_sample_length], self.params.dict_size, variableLength = self.params.variable_sample_length, oldDatasetPath = 'datasets/' + self.params.dataset + '.npy')
                samples = filterDuplicateSamples(np.concatenate((samples,randomSamples),axis=0))


        return samples[:nQueries]


    def sampleForQuery(self, model, iterNum):
        '''
        generate query candidates via MCMC sampling
        automatically filter any duplicates within the sample and the existing dataset
        :return:
        '''
        if self.params.query_mode == 'energy':
            scoreFunction = [1, 0]  # weighting between score and uncertainty - look for minimum score
        elif self.params.query_mode == 'uncertainty':
            scoreFunction = [0, 1]  # look for maximum uncertainty
        elif self.params.query_mode == 'heuristic':
            scoreFunction = [0.5, 0.5]  # put in user specified values (or functions) here

        # do a single sampling run
        self.sampleDict = self.runSampling(model, scoreFunction, iterNum)


    def runSampling(self, model, scoreFunction, seedInd, useOracle=False):
        """
        run MCMC sampling
        :return:
        """
        gammas = np.logspace(self.params.stun_min_gamma, self.params.stun_max_gamma, self.params.mcmc_num_samplers)
        self.mcmcSampler = sampler(self.params, seedInd, scoreFunction, gammas)
        outputs = runSampling(self.params, self.mcmcSampler, model, useOracle=useOracle)

        return outputs
