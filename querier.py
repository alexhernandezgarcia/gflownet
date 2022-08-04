"""import statements"""
import numpy as np
from sampler import *
from gflownet import GFlowNetAgent, batch2dict
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

    def buildQuery(self, model, statusDict, action = None, comet=None,):
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
            query = generateRandomSamples(nQueries, [self.config.dataset.min_length,self.config.dataset.max_length],
                                          self.config.dataset.dict_size, variableLength = self.config.dataset.variable_length,
                                          oldDatasetPath = 'datasets/' + self.config.dataset.oracle + '.npy')

        else:

            '''
            query samples with best good scores, according to our model and a scoring function
            '''

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

        # for memory purposes, we have to cap the total number of samples considered
        maxLen = 10000
        if len(samples) > maxLen:
            bestInds = np.argsort(scores)[:maxLen]
            scores = scores[bestInds]
            samples = samples[bestInds]
            uncertainties = uncertainties[bestInds]

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
            samples = samples[np.argsort(scores)]

        if len(samples) < nQueries:
            missing_samples = nQueries - len(samples)
            printRecord("Querier did not produce enough samples, adding {} random entries, consider adjusting query size and/or cutoff".format(missing_samples))

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
        # get score function by which we identify 'good' samples to send to oracle
        scoreFunction = self.getScoreFunction()

        # do a single sampling run
        sampleDict = self.runSampling(model, scoreFunction = scoreFunction, al_iter = iterNum)

        return sampleDict # return information on samples collected during on the run


    def getScoreFunction(self):
        if self.config.al.query_mode == 'energy':
            scoreFunction = [1, 0]  # weighting between score and uncertainty - look for minimum score
        elif self.config.al.query_mode == 'uncertainty':
            scoreFunction = [0, 1]  # look for maximum uncertainty
        elif (self.config.al.query_mode == 'heuristic') or (self.config.al.query_mode == 'learned'):
            c1 = 0.5 - self.config.al.energy_uncertainty_tradeoff / 2
            c2 = 0.5 + self.config.al.energy_uncertainty_tradeoff / 2
            scoreFunction = [c1, c2]
        elif self.config.al.query_mode == 'fancy_acquisition':
            c1 = 1
            c2 = 1
            scoreFunction = [c1, c2]
        else:
            raise ValueError(self.config.al.query_mode + 'is not a valid query function!')

        return scoreFunction

    def runSampling(self, model, scoreFunction = (1, 0), al_iter = 0, useOracle=False, method_overwrite = False):
        """
        run MCMC or GFlowNet sampling
        :return:
        """
        if not method_overwrite:
            method = self.method
        else:
            method = method_overwrite

        if method.lower() == "mcmc":
            t0 = time.time()
            gammas = np.logspace(self.config.mcmc.stun_min_gamma, self.config.mcmc.stun_max_gamma, self.config.mcmc.num_samplers)
            self.mcmcSampler = Sampler(self.config, al_iter, scoreFunction, gammas)
            outputs = self.mcmcSampler.sample(model, useOracle=useOracle)
            outputs = filterOutputs(outputs)
            printRecord('MCMC sampling took {} seconds'.format(int(time.time()-t0)))

        elif method.lower() == "random":
            t0 = time.time()
            samples = generateRandomSamples(self.config.al.num_random_samples, [self.config.dataset.min_length,self.config.dataset.max_length], self.config.dataset.dict_size,
                                            variableLength = self.config.dataset.variable_length,
                                            oldDatasetPath = 'datasets/' + self.config.dataset.oracle + '.npy',
                                            seed = self.config.seeds.sampler + al_iter)
            if self.config.al.query_mode == 'fancy_acquisition':
                scores, energies, std_dev = model.evaluate(samples,output="fancy_acquisition")
            else:
                energies, std_dev = model.evaluate(samples,output="Both")
                scores = energies * scoreFunction[0] - scoreFunction[1] * np.asarray(std_dev)
            outputs = {
                'samples': samples,
                'energies': energies,
                'uncertainties': std_dev,
                'scores':scores
            }
            outputs = self.doAnnealing(scoreFunction, model, outputs, seed = al_iter)
            printRecord('Random sampling and annealing took {} seconds'.format(int(time.time()-t0)))

        elif method.lower() == "gflownet":
            gflownet = GFlowNetAgent(self.config, comet = self.comet, proxy=model.raw,
                                     al_iter=al_iter, data_path='datasets/' + self.config.dataset.oracle + '.npy')

            t0 = time.time()
            gflownet.train()
            printRecord('Training GFlowNet took {} seconds'.format(int(time.time()-t0)))
            outputs, times = gflownet.sample_batch(gflownet.env, 
                self.config.gflownet.n_samples, train=False)
            outputs, times_batch = batch2dict(outputs, gflownet.env,
                    get_uncertainties=True, query_function=self.config.al.query_mode)
            printRecord('Sampling {} samples from GFlowNet took {} seconds'.format(self.config.gflownet.n_samples, int(time.time()-t0)))
            outputs = filterOutputs(outputs)

            if self.config.gflownet.annealing:
                outputs = self.doAnnealing(scoreFunction, model, outputs, seed=al_iter)

        else:
            raise NotImplemented("method can be either mcmc or gflownet or random")

        return outputs


    def doAnnealing(self, scoreFunction, model, outputs, seed = 0, useOracle=False):
        t0 = time.time()
        initConfigs = outputs['samples'][np.argsort(outputs['scores'])]
        initConfigs = initConfigs[0:self.config.al.annealing_samples]

        annealer = Sampler(self.config, 1, scoreFunction, gammas=np.arange(len(initConfigs)))  # the gamma is a dummy, and will not be used (this is not STUN MC)
        annealedOutputs = annealer.postSampleAnnealing(initConfigs, model, useOracle=useOracle, seed = self.config.seeds.sampler + seed)

        filteredOutputs = filterOutputs(outputs, additionalEntries = annealedOutputs)

        nAddedSamples = int(len(filteredOutputs['samples']) - len(outputs['samples']))

        printRecord('Post-sample annealing added {} samples in {} seconds'.format(nAddedSamples, int(time.time()-t0)))

        return filteredOutputs
