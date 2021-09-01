"""import statements"""
import numpy as np
from sampler import *
from Agent import DQN
from gflownet import GFlowNetAgent
import multiprocessing as mp

'''
This script selects sequences to be sent to the oracle for scoring

> Inputs: model extrema (sequences in 0123 format)
> Outputs: sequences to be scored (0123 format)

To-Do:
==> implement RL model
==> implement gFlowNet sampler
'''


class Querier():
    def __init__(self, params):
        self.params = params
        self.method = params.sample_method
        if self.params.query_mode == 'learned':
            self.qModel = DQN(self.params) # initialize q-network

    def buildQuery(self, model, statusDict, energySampleDict):
        """
        select the samples which will be sent to the oracle for scoring
        :param sampleDict:
        :return:
        """
        # TODO upgrade sampler

        nQueries = self.params.queries_per_iter
        if self.params.query_mode == 'random':
            '''
            generate query randomly
            '''
            samples = generateRandomSamples(self.params.queries_per_iter, [self.params.min_sample_length,self.params.max_sample_length], self.params.dict_size, variableLength = self.params.variable_sample_length, oldDatasetPath = 'datasets/' + self.params.dataset + '.npy')

        else:
            if self.params.query_mode == 'learned':
                self.qModel.updateModelState(statusDict, model)
                self.sampleForQuery(self.qModel, statusDict['iter'])

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

            samples = self.constructQuery(samples, scores, uncertainties, nQueries)

        return samples


    def constructQuery(self, samples, scores, uncertainties, nQueries):
        # create batch from candidates
        if self.params.query_selection == 'clustering':
            # agglomerative clustering
            clusters, clusterScores, clusterVars = doAgglomerativeClustering(samples, scores, uncertainties, cutoff=self.params.minima_dist_cutoff)
            clusterSizes, avgClusterScores, minCluster, avgClusterVars, minClusterVars, minClusterSamples = clusterAnalysis(clusters, clusterScores, clusterVars)
            samples = minClusterSamples
        elif self.params.query_selection == 'cutoff':
            # build up sufficiently different examples in order of best scores
            bestInds = sortTopXSamples(samples[np.argsort(scores)], nSamples=len(samples), distCutoff=self.params.minima_dist_cutoff)  # sort out the best, and at least minimally distinctive samples
            samples = samples[bestInds]
        elif self.params.query_selection == 'argmin':
            # just take the bottom x scores
            samples = samples[np.argsort(scores)]

        while len(samples) < nQueries:  # if we don't have enough samples from samplers, add random ones to pad out the query
            randomSamples = generateRandomSamples(1000, [self.params.min_sample_length, self.params.max_sample_length], self.params.dict_size, variableLength=self.params.variable_sample_length,
                                                  oldDatasetPath='datasets/' + self.params.dataset + '.npy')
            samples = filterDuplicateSamples(np.concatenate((samples, randomSamples), axis=0))

        return samples[:nQueries]


    def sampleForQuery(self, model, iterNum):
        '''
        generate query candidates via MCMC or GFlowNet sampling
        automatically filter any duplicates within the sample and the existing dataset
        :return:
        '''
        if self.params.query_mode == 'energy':
            scoreFunction = [1, 0]  # weighting between score and uncertainty - look for minimum score
        elif self.params.query_mode == 'uncertainty':
            scoreFunction = [0, 1]  # look for maximum uncertainty
        elif self.params.query_mode == 'heuristic':
            scoreFunction = [0.5, 0.5]  # put in user specified values (or functions) here
        elif self.params.query_mode == 'learned':
            scoreFunction = None
        else:
            raise ValueError(self.params.query_mode + 'is not a valid query function!')

        # do a single sampling run
        self.sampleDict = self.runSampling(model, scoreFunction, iterNum)


    def runSampling(self, model, scoreFunction, seedInd, useOracle=False):
        """
        run MCMC or GFlowNet sampling
        :return:
        """
        if self.method.lower() == "mcmc":
            # TODO add optional post-sample annealing
            gammas = np.logspace(self.params.stun_min_gamma, self.params.stun_max_gamma, self.params.mcmc_num_samplers)
            self.mcmcSampler = Sampler(self.params, seedInd, scoreFunction, gammas)
            samples = self.mcmcSampler.sample(model, useOracle=useOracle)
            outputs = samples2dict(samples)
        elif self.method.lower() == "gflownet":
            # TODO: instead of initializing gflownet from scratch, we could retrain it?
            gflownet = GFlowNetAgent(self.params, proxy=model.evaluate)
            gflownet.train()
            outputs = gflownet.sample(
                    self.params.gflownet_n_samples, self.params.horizon,
                    self.params.nalphabet, model.evaluate
            )
        else:
            raise NotImplemented("method can be either mcmc or gflownet")

        return outputs
