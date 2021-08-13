from models import modelNet
from querier import *
from sampler import *
from torch.utils import data
import torch.nn.functional as F
import torch

import os
import glob
import multiprocessing as mp


class activeLearning():
    def __init__(self, params):
        self.pipeIter = None
        self.params = params
        self.runNum = self.params['run num']
        self.setup()


    def setup(self):
        '''
        setup working directory
        move to relevant directory
        :return:
        '''
        self.oracle = oracle(self.params) # oracle needs to be initialized to initialize toy datasets

        if (self.params['run num'] == 0) or (self.params['explicit run enumeration'] == True): # if making a new workdir
            if self.params['run num'] == 0:
                self.makeNewWorkingDirectory()
            else:
                self.workDir = self.params['workdir'] + '/run%d'%self.params['run num'] # explicitly enumerate the new run directory
                os.mkdir(self.workDir)

            os.mkdir(self.workDir + '/ckpts')
            os.mkdir(self.workDir + '/datasets')
            os.chdir(self.workDir) # move to working dir
            printRecord('Starting Fresh Run %d' %self.runNum)
            #copyfile(self.params['dataset directory'] + '/' + self.params['dataset'],self.workDir + '/datasets/' + self.params['dataset'] + '.npy') # if using a real initial dataset (not toy) copy it to the workdir
            self.oracle.initializeDataset() # generate toy model dataset
        else:
            # move to working dir
            self.workDir = self.params['workdir'] + '/' + 'run%d' %self.params['run num']
            os.chdir(self.workDir)
            printRecord('Resuming run %d' % self.params['run num'])


        self.querier = querier(self.params) # might as well initialize the querier here


    def makeNewWorkingDirectory(self):    # make working directory
        '''
        make a new working directory
        non-overlapping previous entries
        :return:
        '''
        workdirs = glob.glob(self.params['workdir'] + '/' + 'run*') # check for prior working directories
        if len(workdirs) > 0:
            prev_runs = []
            for i in range(len(workdirs)):
                prev_runs.append(int(workdirs[i].split('run')[-1]))

            prev_max = max(prev_runs)
            self.workDir = self.params['workdir'] + '/' + 'run%d' %(prev_max + 1)
            os.mkdir(self.workDir)
            self.runNum = int(prev_max + 1)
        else:
            self.workDir = self.params['workdir'] + '/' + 'run1'
            os.mkdir(self.workDir)


    def runPipeline(self):
        '''
        run  the active learning pipeline for a number of iterations
        :return:
        '''
        self.testMinima = [] # best test loss of models, for each iteration of the pipeline
        self.bestScores = [] # best optima found by the sampler, for each iteration of the pipeline

        if self.params['dataset type'] == 'toy':
            self.sampleOracle() # use the oracle to pre-solve the problem for future benchmarking
            printRecord(f"The true global minimum is {bcolors.OKGREEN}%.3f{bcolors.ENDC}" % self.trueMinimum)

        for self.pipeIter in range(self.params['pipeline iterations']):
            printRecord(f'Starting pipeline iteration #{bcolors.FAIL}%d{bcolors.ENDC}' % int(self.pipeIter+1))
            self.params['iteration'] = self.pipeIter + 1
            self.iterate() # run the pipeline

            self.saveOutputs() # save pipeline outputs


    def iterate(self):
        '''
        run one iteration of the pipeline - train model, sample sequences, select sequences, consult oracle
        :return:
        '''
        t0 = time.time()
        self.retrainModels(parallel=self.params['training parallelism'])
        tf = time.time()
        printRecord('Retraining took {} seconds'.format(int(tf-t0)))

        t0 = time.time()
        query = self.getQuery()
        tf = time.time()
        printRecord('Query generation took {} seconds'.format(int(tf-t0)))

        t0 = time.time()
        scores = self.oracle.score(query) # score Samples
        tf = time.time()
        print('Oracle scored' + bcolors.OKBLUE + ' {} '.format(len(scores)) + bcolors.ENDC + 'queries with average score of' + bcolors.OKGREEN + ' {:.3f}'.format(np.average(scores)) + bcolors.ENDC)
        if not self.params['dataset type'] == 'toy':
            print('Oracle scoring took {} seconds'.format(int(tf-t0)))

        self.reportStatus() # compute and record the current status of the active learner w.r.t. the dataset

        self.updateDataset(query, scores) # add scored Samples to dataset


    def reportStatus(self, manualRerun = False):
        '''
        sample the model
        report on the status of dataset
        report on best scores according to models
        report on model confidence
        :manualRerun: reruns the sampler even if we already have priorsampler data)
        :return:
        '''
        # if we need to rerun the sampler, do so, otherwise use data from sampling run immediately preceding
        if (not hasattr(self.querier, 'samplingOutputs')) or manualRerun:
            self.loadEstimatorEnsemble()
            sampleDict = self.querier.runSampling(self.model, [1, 0], 1) # faster this way
        else:
            sampleDict = self.querier.samplingOutputs

        # top X distinct samples - energies and uncertainties - we want to know how many minima it's found, how low they are, and how confident we are about them
        enArgSort = np.argsort(sampleDict['energies'])
        sortedSamples = sampleDict['samples'][enArgSort]
        sortedEns = sampleDict['energies'][enArgSort]
        sortedVars = sampleDict['uncertainties']

        bestInds = sortTopXSamples(sortedSamples, nSamples = self.params['distinct minima'], distCutoff = self.params['minima dist cutoff'])
        bestSamples = sortedSamples[bestInds]
        bestEns = sortedEns[bestInds]
        bestVars = sortedVars[bestInds]

        printRecord('{} Samplers found top {:} distinct samples with minimum energy'.format(int(self.params['num samplers']),len(bestInds)) + bcolors.OKGREEN + ' {:.2f},'.format(np.amin(bestEns)) + bcolors.ENDC + ' average energy' + bcolors.OKGREEN + ' {:.2f},'.format(np.average(bestEns)) + bcolors.ENDC + ' and average std dev' + bcolors.OKCYAN + ' {:.2f}'.format(np.average(np.sqrt(bestVars))) + bcolors.ENDC)

        if self.params['dataset type'] == 'toy': # we can check the test error against a huge random dataset
            self.largeModelEvaluation()
            self.printOverallPerformance(bestEns, bestVars)

        # model beliefs about total dataset as well - similar but not identical to test loss
        # we can also look at the best X scores and uncertainties in the dataset and/or confidence on overall dataset - test loss is a decent proxy for this

        if self.pipeIter == 0: # if it's the first round, initialize, else, append
            self.bestSamples = [bestSamples]
            self.bestEns = [bestEns]
            self.bestVars = [bestVars]
        else:
            self.bestSamples.append(bestSamples)
            self.bestEns.append(bestEns)
            self.bestVars.append(bestVars)


    def printOverallPerformance(self,bestEns,bestVars):
        '''
        print the performance of the learner against a known best answer
        :param bestEns:
        :param bestVars:
        :return:
        '''
        bestSoFar = np.amin(bestEns)
        bestSoFarVar = bestVars[np.argmin(np.amin(bestEns))]
        disagreement = np.abs(self.trueMinimum - bestSoFar) / self.trueMinimum
        printRecord('Active Learner is off by' + bcolors.WARNING + ' {}%'.format(int(disagreement * 100)) + bcolors.ENDC + ' from true minimum with an uncertainty of' + bcolors.WARNING + ' {:.2f}'.format(bestSoFarVar) + bcolors.ENDC)

    def retrainModels(self, parallel=True):
        if not parallel:
            for i in range(self.params['ensemble size']):
                self.resetModel(i)  # reset between ensemble estimators EVERY ITERATION of the pipeline
                self.model.converge()  # converge model
                self.testMinima.append(np.amin(self.model.err_te_hist))
        else:
            del self.model
            if self.params['device'] == 'local':
                nHold = 4
            else:
                nHold = 1
            cpus = int(os.cpu_count() - nHold)
            cpus = min(cpus,self.params['ensemble size']) # only as many CPUs as we need
            with mp.Pool(processes=cpus) as pool:
                output = [pool.apply_async(trainModel, args=[self.params, j]) for j in range(self.params['ensemble size'])]
                outputList = [output[i].get() for i in range(self.params['ensemble size'])]
                self.testMinima.append([np.amin(outputList[i]) for i in range(self.params['ensemble size'])])
                pool.close()
                pool.join()

        printRecord('%d '%self.params['ensemble size'] + f'Model ensemble training converged with average test loss of {bcolors.OKCYAN}%.5f{bcolors.ENDC}' % np.average(np.asarray(self.testMinima[-self.params['ensemble size']:])))


    def getQuery(self):
        self.loadEstimatorEnsemble()
        query = self.querier.buildQuery(self.model, self.pipeIter)  # pick Samples to be scored

        return query


    def loadEstimatorEnsemble(self):
        '''
        load all the trained models at their best checkpoints
        and initialize them in an ensemble model where they can all be queried at once
        :return:
        '''
        ensemble = []
        for i in range(self.params['ensemble size']):
            self.resetModel(i)
            self.model.load(i)
            ensemble.append(self.model.model)

        del self.model
        self.model = modelNet(self.params,0)
        self.model.loadEnsemble(ensemble)

        #print('Loaded {} estimators'.format(int(self.params['ensemble size'])))


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


    def largeModelEvaluation(self):
        '''
        if we are using a toy oracle, we should be able to easily get the test loss on a huge sample of the dataset
        :return:
        '''
        self.loadEstimatorEnsemble()

        numSamples = min(int(1e5), self.params['dict size'] ** self.params['max sample length'] // 100) # either 1e5, or 1% of the sample space, whichever is smaller
        randomData = self.oracle.initializeDataset(save=False, returnData=True, customSize=numSamples) # get large random dataset
        randomSamples = randomData['samples']
        randomScores = randomData['scores']

        sortInds = np.argsort(randomScores) # sort randoms
        randomSamples = randomSamples[sortInds]
        randomScores = randomScores[sortInds]

        modelScores, modelVars = [[],[]]
        sampleLoader = data.DataLoader(randomSamples, batch_size = self.params['batch size'], shuffle=False, num_workers = 0, pin_memory=False)
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

        print("Model has overall loss of" + bcolors.OKCYAN + ' {:.5f}, '.format(totalLoss) + bcolors.ENDC + 'best 10% loss of' + bcolors.OKCYAN + ' {:.5f} '.format(bottomTenLoss) + bcolors.ENDC +  'on {} toy dataset samples'.format(numSamples))

    def sampleOracle(self):
        '''
        for toy models
        do global optimization directly on the oracle to find the true minimum
        :return:
        '''
        printRecord("Asking toy oracle for the true minimum")
        self.model = 'abc'
        gammas = np.logspace(-4,1,self.params['num samplers'])
        mcmcSampler = sampler(self.params, 0, [1,0], gammas)
        sampleDict = runSampling(self.params, mcmcSampler, self.model, useOracle=True)

        bestMin = np.amin(sampleDict['energies'])
        printRecord(f"Sampling Complete! Lowest Energy Found = {bcolors.FAIL}%.3f{bcolors.ENDC}" % bestMin + " from %d" % self.params['num samplers'] + " sampling runs.")

        self.oracleOptima = sampleDict
        self.trueMinimum = np.amin(self.oracleOptima['scores'])


    def saveOutputs(self):
        '''
        save params and outputs in a dict
        :return:
        '''
        outputDict = {}
        outputDict['params'] = self.params
        if self.params['dataset type'] == 'toy':
            outputDict['oracle outputs'] = self.oracleOptima
            outputDict['big dataset loss'] = self.totalLoss
            outputDict['bottom 10% loss'] = self.bottomTenLoss
        outputDict['best samples'] = self.bestSamples
        outputDict['best energies'] = self.bestEns
        outputDict['best vars'] = self.bestVars
        outputDict['model test minima'] = self.testMinima
        np.save('outputsDict',outputDict)


    def updateDataset(self, oracleSequences, oracleScores):
        '''
        loads dataset, appends new datapoints from oracle, and saves dataset
        :param params: model parameters
        :param oracleSequences: sequences which were sent to oracle
        :param oracleScores: scores of sequences sent to oracle
        :return: n/a
        '''
        dataset = np.load('datasets/' + self.params['dataset'] + '.npy', allow_pickle=True).item()

        dataset['samples'] = np.concatenate((dataset['samples'], oracleSequences))
        dataset['scores'] = np.concatenate((dataset['scores'], oracleScores))

        printRecord(f"Added{bcolors.OKBLUE}{bcolors.BOLD} %d{bcolors.ENDC}" % int(len(oracleSequences)) + " to the dataset, total dataset size is" + bcolors.OKBLUE + " {}".format(int(len(dataset['samples']))) + bcolors.ENDC)
        printRecord(bcolors.UNDERLINE + "=====================================================================" + bcolors.ENDC)
        np.save('datasets/' + self.params['dataset'], dataset)


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

