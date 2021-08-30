from models import modelNet
from querier import *
from sampler import *
from torch.utils import data
import torch.nn.functional as F
import torch

import os
import glob
import multiprocessing as mp


class ActiveLearning():
    def __init__(self, params):
        self.pipeIter = None
        self.params = params
        self.runNum = self.params.run_num
        self.setup()
        self.getModelSize()


    def setup(self):
        '''
        setup working directory
        move to relevant directory
        :return:
        '''
        self.oracle = Oracle(self.params) # oracle needs to be initialized to initialize toy datasets

        if (self.params.run_num == 0) or (self.params.explicit_run_enumeration == True): # if making a new workdir
            if self.params.run_num == 0:
                self.makeNewWorkingDirectory()
            else:
                self.workDir = self.params.workdir + '/run%d'%self.params.run_num # explicitly enumerate the new run directory
                os.mkdir(self.workDir)

            os.mkdir(self.workDir + '/ckpts')
            os.mkdir(self.workDir + '/datasets')
            os.chdir(self.workDir) # move to working dir
            printRecord('Starting Fresh Run %d' %self.runNum)
            self.oracle.initializeDataset() # generate toy model dataset
        else:
            # move to working dir
            self.workDir = self.params.workdir + '/' + 'run%d' %self.params.run_num
            os.chdir(self.workDir)
            printRecord('Resuming run %d' % self.params.run_num)


        self.querier = Querier(self.params) # might as well initialize the querier here


    def makeNewWorkingDirectory(self):    # make working directory
        '''
        make a new working directory
        non-overlapping previous entries
        :return:
        '''
        workdirs = glob.glob(self.params.workdir + '/' + 'run*') # check for prior working directories
        if len(workdirs) > 0:
            prev_runs = []
            for i in range(len(workdirs)):
                prev_runs.append(int(workdirs[i].split('run')[-1]))

            prev_max = max(prev_runs)
            self.workDir = self.params.workdir + '/' + 'run%d' %(prev_max + 1)
            os.mkdir(self.workDir)
            self.runNum = int(prev_max + 1)
        else:
            self.workDir = self.params.workdir + '/' + 'run1'
            os.mkdir(self.workDir)


    def runPipeline(self):
        '''
        run  the active learning pipeline for a number of iterations
        :return:
        '''
        self.testMinima = [] # best test loss of models, for each iteration of the pipeline
        self.bestScores = [] # best optima found by the sampler, for each iteration of the pipeline

        if self.params.dataset_type == 'toy':
            self.sampleOracle() # use the oracle to pre-solve the problem for future benchmarking
            printRecord(f"The true global minimum is {bcolors.OKGREEN}%.3f{bcolors.ENDC}" % self.trueMinimum)

        self.params.dataset_size = self.params.init_dataset_length
        for self.pipeIter in range(self.params.pipeline_iterations):
            printRecord(f'Starting pipeline iteration #{bcolors.FAIL}%d{bcolors.ENDC}' % int(self.pipeIter+1))
            self.iterate() # run the pipeline
            self.saveOutputs() # save pipeline outputs
            if (self.pipeIter > 0) and (self.params.dataset_type == 'toy'):
                self.reportCumulativeResult()


    def iterate(self):
        '''
        run one iteration of the pipeline - train model, sample sequences, select sequences, consult oracle
        :return:
        '''

        t0 = time.time()
        self.retrainModels(parallel=self.params.training_parallelism)
        tf = time.time()
        printRecord('Retraining took {} seconds'.format(int(tf-t0)))

        t0 = time.time()
        self.getModelState() # run energy-only sampling and create model state dict
        query = self.querier.buildQuery(self.model, self.stateDict, self.sampleDict)  # pick Samples to be scored
        tf = time.time()
        printRecord('Query generation took {} seconds'.format(int(tf-t0)))

        t0 = time.time()
        scores = self.oracle.score(query) # score Samples
        tf = time.time()
        printRecord('Oracle scored' + bcolors.OKBLUE + ' {} '.format(len(scores)) + bcolors.ENDC + 'queries with average score of' + bcolors.OKGREEN + ' {:.3f}'.format(np.average(scores)) + bcolors.ENDC)
        if not self.params.dataset_type == 'toy':
            printRecord('Oracle scoring took {} seconds'.format(int(tf-t0)))

        self.updateDataset(query, scores) # add scored Samples to dataset


    def getModelState(self):
        '''
        sample the model
        report on the status of dataset
        report on best scores according to models
        report on model confidence
        :manualRerun: reruns the sampler even if we already have priorsampler data)
        :return:
        '''
        '''
        key outputs (not toy):
            - test loss
            - energy and uncertainty of best X distinct samples
        key outputs (toy):
            - large sample loss & bottom x% loss
            - distance to known minimum
            -? number of true minima
        '''

        # run the sampler
        self.loadEstimatorEnsemble()
        self.sampleDict = self.querier.runSampling(self.model, [1, 0], 1) # sample existing optima
        samples = self.sampleDict['samples']
        energies = self.sampleDict['energies']
        uncertainties = self.sampleDict['uncertainties']

        # agglomerative clustering
        clusters, clusterEns, clusterVars = doAgglomerativeClustering(samples,energies,uncertainties,cutoff=self.params.minima_dist_cutoff)
        clusterSizes, avgClusterEns, minClusterEns, avgClusterVars, minClusterVars, minClusterSamples = clusterAnalysis(clusters, clusterEns, clusterVars)

        #clutering alternative - just include sample-by-sample
        #bestInds = sortTopXSamples(samples[np.argsort(scores)], nSamples=len(samples), distCutoff=0.1)  # sort out the best, and at least minimally distinctive samples

        if len(clusters) < self.params.model_state_size: # if we don't have enough clusters for the model, pad with random samples from the sampling run
            minClusterSamples, minClusterEns, minClusterVars = self.addRandomSamples(samples, energies, uncertainties, minClusterSamples, minClusterEns, minClusterVars)

        # get distances to relevant datasets
        internalDist, datasetDist, randomDist = self.getDataDiffs(minClusterSamples[:self.params.model_state_size])

        self.stateDict = {
            'test loss': np.average(self.testMinima), # losses are evaluated on standardized data, so we do not need to re-standardize here
            'test std': np.sqrt(np.var(self.testMinima)),
            'all test losses': self.testMinima,
            'best cluster energies': (minClusterEns[:self.params.model_state_size] - self.model.mean) / self.model.std, # standardize according to dataset statistics
            'best cluster deviations': np.sqrt(minClusterVars[:self.params.model_state_size]) / self.model.std,
            'best cluster samples': minClusterSamples[:self.params.model_state_size],
            'best clusters internal diff': internalDist,
            'best clusters dataset diff': datasetDist,
            'best clusters random set diff': randomDist,
            'clustering cutoff': self.params.minima_dist_cutoff, # could be a learned parameter
            'n proxy models': self.params.proxy_model_ensemble_size,
            'iter': self.pipeIter,
            'budget': self.params.pipeline_iterations
        }

        printRecord('%d '%self.params.proxy_model_ensemble_size + f'Model ensemble training converged with average test loss of {bcolors.OKCYAN}%.5f{bcolors.ENDC}' % np.average(np.asarray(self.testMinima[-self.params.proxy_model_ensemble_size:])) + f' and std of {bcolors.OKCYAN}%.3f{bcolors.ENDC}'%(np.sqrt(np.var(self.testMinima))))
        printRecord('Model state contains {} samples'.format(self.params.model_state_size) +
                    ' with minimum energy' + bcolors.OKGREEN + ' {:.2f},'.format(np.amin(minClusterEns)) + bcolors.ENDC +
                    ' average energy' + bcolors.OKGREEN +' {:.2f},'.format(np.average(minClusterEns[:self.params.model_state_size])) + bcolors.ENDC +
                    ' and average std dev' + bcolors.OKCYAN + ' {:.2f}'.format(np.average(np.sqrt(minClusterVars[:self.params.model_state_size]))) + bcolors.ENDC)
        printRecord('Sample average mutual distance is ' + bcolors.WARNING +'{:.2f} '.format(np.average(internalDist)) + bcolors.ENDC +
                    'dataset distance is ' + bcolors.WARNING + '{:.2f} '.format(np.average(datasetDist)) + bcolors.ENDC +
                    'and overall distance estimated at ' + bcolors.WARNING + '{:.2f}'.format(np.average(randomDist)) + bcolors.ENDC)

        if self.params.dataset_type == 'toy': # we can check the test error against a huge random dataset
            self.largeModelEvaluation()
            self.printOverallPerformance(minClusterEns, minClusterVars)


        if self.pipeIter == 0: # if it's the first round, initialize, else, append
            self.stateDictRecord = [self.stateDict]
        else:
            self.stateDictRecord.append(self.stateDict)


    def printOverallPerformance(self,bestEns,bestVars):
        '''
        print the performance of the learner against a known best answer
        :param bestEns:
        :param bestVars:
        :return:
        '''
        bestSoFar = np.amin(bestEns)
        bestSoFarVar = bestVars[np.argmin(bestEns)]
        stdTrueMin = (self.trueMinimum - self.model.mean) / self.model.std
        stdSampleMin = (bestSoFar - self.model.mean)/ self.model.std
        disagreement = np.abs(stdTrueMin-stdSampleMin)/np.abs(stdTrueMin)
        printRecord('Active learner best sample, {:.2f} is off by'.format(bestSoFar) + bcolors.WARNING + ' {}%'.format(int(disagreement * 100)) + bcolors.ENDC + ' from true minimum with std of' + bcolors.WARNING + ' {:.2f}'.format(np.sqrt(bestSoFarVar)) + bcolors.ENDC)


    def retrainModels(self, parallel=True):
        if not parallel:
            testMins = []
            for i in range(self.params.proxy_model_ensemble_size):
                self.resetModel(i)  # reset between ensemble estimators EVERY ITERATION of the pipeline
                self.model.converge()  # converge model
                testMins.append(np.amin(self.model.err_te_hist))
            self.testMinima.append(testMins)
        else:
            del self.model
            if self.params.machine == 'local':
                nHold = 4
            else:
                nHold = 1
            cpus = int(os.cpu_count() - nHold)
            cpus = min(cpus,self.params.proxy_model_ensemble_size) # only as many CPUs as we need
            with mp.Pool(processes=cpus) as pool:
                output = [pool.apply_async(trainModel, args=[self.params, j]) for j in range(self.params.proxy_model_ensemble_size)]
                outputList = [output[i].get() for i in range(self.params.proxy_model_ensemble_size)]
                self.testMinima.append([np.amin(outputList[i]) for i in range(self.params.proxy_model_ensemble_size)])
                pool.close()
                pool.join()


    def loadEstimatorEnsemble(self):
        '''
        load all the trained models at their best checkpoints
        and initialize them in an ensemble model where they can all be queried at once
        :return:
        '''
        ensemble = []
        for i in range(self.params.proxy_model_ensemble_size):
            self.resetModel(i)
            self.model.load(i)
            ensemble.append(self.model.model)

        del self.model
        self.model = modelNet(self.params,0)
        self.model.loadEnsemble(ensemble)

        #print('Loaded {} estimators'.format(int(self.params.proxy_model_ensemble_size)))


    def resetModel(self,ensembleIndex, returnModel = False):
        '''
        load a new instance of the model with reset parameters
        :return:
        '''
        try: # if we have a model already, delete it
            del self.model
        except:
            pass
        self.model = modelNet(self.params,ensembleIndex)
        #printRecord(f'{bcolors.HEADER} New model: {bcolors.ENDC}', getModelName(ensembleIndex))
        if returnModel:
            return self.model


    def getModelSize(self):
        self.model = modelNet(self.params, 0)
        nParams = get_n_params(self.model.model)
        printRecord('Proxy model has {} parameters'.format(int(nParams)))
        del(self.model)


    def largeModelEvaluation(self):
        '''
        if we are using a toy oracle, we should be able to easily get the test loss on a huge sample of the dataset
        :return:
        '''
        self.loadEstimatorEnsemble()

        numSamples = min(int(1e5), self.params.dict_size ** self.params.max_sample_length // 100) # either 1e5, or 1% of the sample space, whichever is smaller
        randomData = self.oracle.initializeDataset(save=False, returnData=True, customSize=numSamples) # get large random dataset
        randomSamples = randomData['samples']
        randomScores = randomData['scores']

        sortInds = np.argsort(randomScores) # sort randoms
        randomSamples = randomSamples[sortInds]
        randomScores = randomScores[sortInds]

        modelScores, modelVars = [[],[]]
        sampleLoader = data.DataLoader(randomSamples, batch_size = self.params.proxy_training_batch_size, shuffle=False, num_workers = 0, pin_memory=False)
        for i, testData in enumerate(sampleLoader):
            score, variance = self.model.evaluate(testData.float(), output='Both')
            modelScores.extend(score)
            modelVars.extend(variance)

        bestTenInd = numSamples // 10
        totalLoss = F.smooth_l1_loss((torch.Tensor(modelScores).float() - self.model.mean) / self.model.std, (torch.Tensor(randomScores).float() - self.model.mean) / self.model.std) # full dataset loss (standardized basis)
        bottomTenLoss = F.smooth_l1_loss((torch.Tensor(modelScores[:bestTenInd]).float() - self.model.mean) / self.model.std, (torch.Tensor(randomScores[:bestTenInd]).float() - self.model.mean) / self.model.std) # bottom 10% loss (standardized basis)

        if self.pipeIter == 0: # if it's the first round, initialize, else, append
            self.totalLoss = [totalLoss]
            self.bottomTenLoss = [bottomTenLoss]
        else:
            self.totalLoss.append(totalLoss)
            self.bottomTenLoss.append(bottomTenLoss)

        printRecord("Model has overall loss of" + bcolors.OKCYAN + ' {:.5f}, '.format(totalLoss) + bcolors.ENDC + 'best 10% loss of' + bcolors.OKCYAN + ' {:.5f} '.format(bottomTenLoss) + bcolors.ENDC +  'on {} toy dataset samples'.format(numSamples))


    def sampleOracle(self):
        '''
        for toy models
        do global optimization directly on the oracle to find the true minimum
        :return:
        '''
        printRecord("Asking toy oracle for the true minimum")

        self.model = 'abc'
        gammas = np.logspace(self.params.stun_min_gamma,self.params.stun_max_gamma,self.params.mcmc_num_samplers)
        mcmcSampler = sampler(self.params, 0, [1,0], gammas)
        sampleDict = runSampling(self.params, mcmcSampler, self.model, useOracle=True)
        if self.params.dataset == 'wmodel': # w model minimum is always zero - even if we don't find it
            bestMin = 0
        else:
            bestMin = np.amin(sampleDict['energies'])

        printRecord(f"Sampling Complete! Lowest Energy Found = {bcolors.FAIL}%.3f{bcolors.ENDC}" % bestMin + " from %d" % self.params.mcmc_num_samplers + " sampling runs.")

        self.oracleRecord = sampleDict
        self.trueMinimum = np.amin(self.oracleRecord['scores'])


    def saveOutputs(self):
        '''
        save params and outputs in a dict
        :return:
        '''
        outputDict = {}
        outputDict['params'] = self.params
        outputDict['state dict record'] = self.stateDictRecord
        if self.params.dataset_type == 'toy':
            outputDict['oracle outputs'] = self.oracleRecord
            outputDict['big dataset loss'] = self.totalLoss
            outputDict['bottom 10% loss'] = self.bottomTenLoss
            if self.pipeIter > 1:
                outputDict['cumulative performance'] = self.cumulativeResult
        np.save('outputsDict',outputDict)


    def updateDataset(self, oracleSequences, oracleScores):
        '''
        loads dataset, appends new datapoints from oracle, and saves dataset
        :param params: model parameters
        :param oracleSequences: sequences which were sent to oracle
        :param oracleScores: scores of sequences sent to oracle
        :return: n/a
        '''
        dataset = np.load('datasets/' + self.params.dataset + '.npy', allow_pickle=True).item()
        # TODO separate between scores and q-scores
        dataset['samples'] = np.concatenate((dataset['samples'], oracleSequences))
        dataset['scores'] = np.concatenate((dataset['scores'], oracleScores))

        self.params.dataset_size = len(dataset['samples'])

        printRecord(f"Added{bcolors.OKBLUE}{bcolors.BOLD} %d{bcolors.ENDC}" % int(len(oracleSequences)) + " to the dataset, total dataset size is" + bcolors.OKBLUE + " {}".format(int(len(dataset['samples']))) + bcolors.ENDC)
        printRecord(bcolors.UNDERLINE + "=====================================================================" + bcolors.ENDC)
        np.save('datasets/' + self.params.dataset, dataset)


    def getScalingFactor(self):
        '''
        since regression is not normalized, we identify a scaling factor against which we normalize our results
        :return:
        '''
        truncationFactor = 0.1 # cut off x% of the furthest outliers
        dataset = np.load('datasets/' + self.params.dataset + '.npy', allow_pickle=True).item()

        scores = dataset['scores']
        d1 = [np.sum(np.abs(scores[i] - scores)) for i in range(len(scores))]
        scores = scores[np.argsort(d1)] # sort according to mutual distance
        margin = int(len(scores) * truncationFactor)
        scores = scores[:-margin] # cut 'margin' of furthest points
        self.scalingFactor = np.ptp(scores)


    def addRandomSamples(self, samples, energies, uncertainties, minClusterSamples, minClusterEns, minClusterVars):
        rands = np.random.randint(0, len(samples), size=self.params.model_state_size - len(minClusterSamples))
        randomSamples = samples[rands]
        randomEnergies = energies[rands]
        randomUncertainties = uncertainties[rands]
        minClusterSamples = np.concatenate((minClusterSamples, randomSamples))
        minClusterEns = np.concatenate((minClusterEns, randomEnergies))
        minClusterVars = np.concatenate((minClusterVars, randomUncertainties))
        printRecord('Padded model state with {} random samples from sampler run'.format(len(rands)))

        return minClusterSamples, minClusterEns, minClusterVars


    def getDataDiffs(self, samples):
        '''
        compute average binary distances between a set of samples and
        1 - itself
        2 - the training dataset
        3 - a large random sample
        :param samples:
        :return:
        '''
        # training dataset
        dataset = np.load('datasets/' + self.params.dataset + '.npy', allow_pickle=True).item()
        dataset = dataset['samples']

        # large, random sample
        numSamples = min(int(1e4), self.params.dict_size ** self.params.max_sample_length // 100) # either 1e5, or 1% of the sample space, whichever is smaller
        randomData = self.oracle.initializeDataset(save=False, returnData=True, customSize=numSamples) # get large random dataset
        randomSamples = randomData['samples']

        internalDist = binaryDistance(samples,pairwise=False,extractInds=len(samples))
        datasetDist = binaryDistance(np.concatenate((samples, dataset)), pairwise=False, extractInds = len(samples))
        randomDist = binaryDistance(np.concatenate((samples,randomSamples)), pairwise=False, extractInds=len(samples))

        return internalDist, datasetDist, randomDist


    def reportCumulativeResult(self):
        '''
        integrate the performance curve over all iterations so far
        :return:
        '''
        directory = os.getcwd()
        plotter = resultsPlotter()
        plotter.process(directory)
        iterAxis = (plotter.xrange - 1) * self.params.queries_per_iter + self.params.init_dataset_length
        bestEns = plotter.normedEns[:,0]
        cumulativeScore = np.trapz(bestEns, x = iterAxis)
        normedCumScore = cumulativeScore / (self.params.dataset_size - self.params.queries_per_iter) # we added to the dataset before this

        printRecord('Cumulative score is {:.2f} gross and {:.5f} per-sample after {} samples'.format(cumulativeScore, normedCumScore, self.params.dataset_size - self.params.queries_per_iter))

        results = {
            'cumulative performance': cumulativeScore,
            'per-sample cumulative performance': normedCumScore,
            'dataset size': (self.params.dataset_size - self.params.queries_per_iter)
        }

        if self.pipeIter == 1:
            self.cumulativeResult = [results]
        else:
            self.cumulativeResult.append(results)


def trainModel(params, i):
    '''
    rewritten for training in a parallelized fashion
    needs to be outside the class method for multiprocessing to work
    :param i:
    :return:
    '''

    model = modelNet(params, i)
    err_te_hist = model.converge(returnHist = True)  # converge model

    return err_te_hist

