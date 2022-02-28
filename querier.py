"""import statements"""
import numpy as np
from sampler import *
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
    def __init__(self, config):
        self.config = config
        self.method = config.al.sample_method
        if self.config.al.query_mode == 'learned':
            pass

    def buildQuery(self, model, statusDict, energySampleDict, action = None, comet=None,):
        """
        select the samples which will be sent to the oracle for scoring
        if we are dynamically updating hyperparameters, take an action
        :param sampleDict:
        :return:
        """
        self.comet = comet
        if action is not None:
            self.updateHyperparams(action)

        nQueries = self.config.al.queries_per_iter
        if self.config.al.query_mode == 'random':
            '''
            generate query randomly
            '''
            query = generateRandomSamples(nQueries, [self.config.dataset.min_length,self.config.dataset.max_length], self.config.dataset.dict_size, variableLength = self.config.dataset.variable_length, oldDatasetPath = 'datasets/' + self.config.dataset.oracle + '.npy')

        else:
            #if self.config.al.query_mode == 'learned': # we aren't doing this anymore
            #    self.qModel.updateModelState(statusDict, model)
            #    self.sampleDict = self.sampleForQuery(self.qModel, statusDict['iter'])

            #else:
            if True:
                '''
                query samples with best good scores, according to our model and a scoring function
                '''

                # generate candidates
                #MK when up update model state calculation, it should be updated here as well
                if (self.config.al.query_mode == 'energy') and (self.config.al.sample_method == 'mcmc'): # we already do energy based sampling with mcmc to generate the model state
                    self.sampleDict = energySampleDict
                else:
                    self.sampleDict = self.sampleForQuery(model, statusDict['iter'])

            samples = self.sampleDict['samples']
            scores = self.sampleDict['scores']
            uncertainties = self.sampleDict['uncertainties']
            samples, inds = filterDuplicateSamples(samples, oldDatasetPath='datasets/' + self.config.dataset.oracle + '.npy', returnInds=True)
            scores = scores[inds]

            query = self.constructQuery(samples, scores, uncertainties, nQueries)

            if self.comet:
                self.comet.log_histogram_3d(self.sampleDict['scores'], name='sampler output scores', step=statusDict['iter'])
                self.comet.log_histogram_3d(np.sqrt(uncertainties), name='sampler output std deviations', step=statusDict['iter'])
                self.comet.log_histogram_3d(self.sampleDict['energies'], name='sampler output energies', step=statusDict['iter'])

        return query


    def updateHyperparams(self,action):
        '''
        take an 'action' to adjust hyperparameters
        action space has a size of 9, and is the  product space of
        [increase, stay the same, decrease] for the two parameters
        minima_dist_cutoff and [c1 - c2] where c1 is the 'energy'
        weight and c2 is the 'uncertainty' weight in the sampler scoring function
        and c1 + c2 = 1
        '''
        binary_to_policy = np.array(((1,1,1,0,0,0,-1,-1,-1),(1,0,-1,1,0,-1,1,0,-1)))
        actions = binary_to_policy @ np.asarray(action) # action 1 is for dist cutoff modulation, action 2 is for c1-c2 tradeoff
        self.config.al.minima_dist_cutoff = self.config.al.minima_dist_cutoff + actions[0] * 0.1 # modulate by 0.1
        self.config.al.energy_uncertainty_tradeoff = self.config.al.energy_uncertainty_tradeoff + actions[1] * 0.1 # modulate by 0.1


    def constructQuery(self, samples, scores, uncertainties, nQueries):
        # create batch from candidates
        if self.config.al.query_selection == 'clustering':
            # agglomerative clustering
            clusters, clusterScores, clusterVars = doAgglomerativeClustering(samples, scores, uncertainties, self.config.dataset.dict_size, cutoff=normalizeDistCutoff(self.config.al.minima_dist_cutoff))

            clusterSizes, avgClusterScores, minCluster, avgClusterVars, minClusterVars, minClusterSamples = clusterAnalysis(clusters, clusterScores, clusterVars)
            samples = minClusterSamples
        elif self.config.al.query_selection == 'cutoff':
            # build up sufficiently different examples in order of best scores
            bestInds = sortTopXSamples(samples[np.argsort(scores)], nSamples=len(samples), distCutoff=normalizeDistCutoff(self.config.al.minima_dist_cutoff))  # sort out the best, and at least minimally distinctive samples
            samples = samples[bestInds]
        elif self.config.al.query_selection == 'argmin':
            # just take the bottom x scores
            if any([s in self.config.dataset.oracle for s in ["pins", "pairs"]]):
                # TODO: a similar operation should be done for the other query
                # selection options
                samples = samples[np.argsort(scores)[::-1]]
            else:
                samples = samples[np.argsort(scores)]

        while len(samples) < nQueries:  # if we don't have enough samples from samplers, add random ones to pad out the query
            randomSamples = generateRandomSamples(1000, [self.config.dataset.min_length, self.config.dataset.max_length], self.config.dataset.dict_size, variableLength=self.config.dataset.variable_length,
                                                  oldDatasetPath='datasets/' + self.config.dataset.oracle + '.npy')
            samples = filterDuplicateSamples(np.concatenate((samples, randomSamples), axis=0))

        return samples[:nQueries]


    def sampleForQuery(self, model, iterNum):
        '''
        generate query candidates via MCMC or GFlowNet sampling
        automatically filter any duplicates within the sample and the existing dataset
        :return:
        '''
        if self.config.al.query_mode == 'energy':
            scoreFunction = [1, 0]  # weighting between score and uncertainty - look for minimum score
        elif self.config.al.query_mode == 'uncertainty':
            scoreFunction = [0, 1]  # look for maximum uncertainty
        elif self.config.al.query_mode == 'heuristic':
            c1 = 0.5 - self.config.al.energy_uncertainty_tradeoff / 2
            c2 = 0.5 + self.config.al.energy_uncertainty_tradeoff / 2
            scoreFunction = [c1, c2]  # put in user specified values (or functions) here
        elif self.config.al.query_mode == 'learned':
            c1 = 0.5 - self.config.al.energy_uncertainty_tradeoff / 2
            c2 = 0.5 + self.config.al.energy_uncertainty_tradeoff / 2
            scoreFunction = [c1, c2]  # put in user specified values (or functions) here
        elif self.config.al.query_mode == 'fancy_acquisition':
            c1 = 1
            c2 = 1
            scoreFunction = [c1, c2]
        else:
            raise ValueError(self.config.al.query_mode + 'is not a valid query function!')

        # do a single sampling run
        sampleDict = self.runSampling(model, scoreFunction, iterNum)

        return sampleDict

    def runSampling(self, model, scoreFunction, seedInd, useOracle=False, method_overwrite = False):
        """
        run MCMC or GFlowNet sampling
        :return:
        """
        if not method_overwrite:
            method = self.method
        else:
            method = method_overwrite

        if method.lower() == "mcmc":
            gammas = np.logspace(self.config.mcmc.stun_min_gamma, self.config.mcmc.stun_max_gamma, self.config.mcmc.num_samplers)
            self.mcmcSampler = Sampler(self.config, seedInd, scoreFunction, gammas)
            samples = self.mcmcSampler.sample(model, useOracle=useOracle)
            outputs = samples2dict(samples)

        elif method.lower() == "random":
            samples = generateRandomSamples(10000, [self.config.dataset.min_length,self.config.dataset.max_length], self.config.dataset.dict_size, variableLength = self.config.dataset.variable_length, oldDatasetPath = 'datasets/' + self.config.dataset.oracle + '.npy')
            energies, std_dev = model.evaluate(samples,output="Both")
            scores = energies * scoreFunction[0] - scoreFunction[1] * np.asarray(std_dev)
            outputs = {
                'samples': samples,
                'energies': energies,
                'uncertainties': std_dev,
                'scores':scores
            }
            outputs = self.doAnnealing(scoreFunction, model, outputs)

        elif method.lower() == "gflownet":
            gflownet = GFlowNetAgent(self.config, comet = self.comet, proxy=model.raw,
                    al_iter=seedInd, data_path='datasets/' + self.config.dataset.oracle + '.npy')

            t0 = time.time()
            gflownet.train()
            tf = time.time()
            printRecord('Training GFlowNet took {} seconds'.format(int(tf-t0)))
            t0 = time.time()
            outputs, times = gflownet.sample(
                    self.config.gflownet.n_samples, self.config.dataset.max_length,
                    self.config.dataset.min_length, self.config.dataset.dict_size, 
                    self.config.gflownet.min_word_len, 
                    self.config.gflownet.max_word_len, model.evaluate
            )
            tf = time.time()
            printRecord('Sampling {} samples from GFlowNet took {} seconds'.format(self.config.gflownet.n_samples, int(tf-t0)))
            outputs = filterOutputs(outputs)

            if self.config.gflownet.annealing:
                outputs = self.doAnnealing(scoreFunction, model, outputs)

        else:
            raise NotImplemented("method can be either mcmc or gflownet or random")

        return outputs


    def doAnnealing(self, scoreFunction, model, outputs):
        t0 = time.time()
        initConfigs = outputs['samples'][np.argsort(outputs['scores'])]
        initConfigs = initConfigs[0:self.config.gflownet.post_annealing_samples]

        annealer = Sampler(self.config, 1, scoreFunction, gammas=np.arange(len(initConfigs)))  # the gamma is a dummy
        annealedOutputs = annealer.postSampleAnnealing(initConfigs, model)

        filteredOutputs = filterOutputs(outputs, additionalEntries = annealedOutputs)
        tf = time.time()

        nAddedSamples = int(len(filteredOutputs['samples']) - len(outputs['samples']))

        printRecord('Post-sample annealing added {} samples in {} seconds'.format(nAddedSamples, int(tf-t0)))

        return filteredOutputs
