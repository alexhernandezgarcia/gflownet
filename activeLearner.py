from utils import *
from models import *
from querier import *
from oracle import *
from sampler import *

import os
import glob
import tqdm
from shutil import copyfile
import multiprocessing as mp


class activeLearning():
    def __init__(self, params):
        self.params = params
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
            #copyfile(self.params['dataset directory'] + '/' + self.params['dataset'],self.workDir + '/datasets/' + self.params['dataset'] + '.npy') # if using a real initial dataset (not toy) copy it to the workdir
            self.oracle.initializeDataset(self.params['sample length'], self.params['init dataset length']) # generate toy model dataset
        else:
            # move to working dir
            self.workDir = self.params['workdir'] + '/' + 'run%d' %self.params['run num']
            os.chdir(self.workDir)

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
            print('Starting Fresh Run %d' %(prev_max + 1))
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

        if self.params['dataset'] == 'toy':
            pass
            self.sampleOracle() # use the oracle to pre-solve the problem for future benchmarking
            print(f"The true global minimum is {bcolors.OKGREEN}%.3f{bcolors.ENDC}" % np.amin(self.oracleOptima['scores']))

        for i in range(self.params['pipeline iterations']):
            print(f'Starting pipeline iteration #{bcolors.OKGREEN}%d{bcolors.ENDC}' % int(i+1))
            self.params['iteration'] = i + 1
            self.iterate() # run the pipeline

        self.saveOutputs() # save final outputs


    def iterate(self):
        '''
        run one iteration of the pipeline - train model, sample sequences, select sequences, consult oracle
        :return:
        '''
        t0 = time.time()
        self.retrainModels()
        tf = time.time()
        print('Retraining took {} seconds'.format(int(tf-t0)))

        t0 = time.time()
        query = self.getQuery()
        tf = time.time()
        print('Query generation took {} seconds'.format(int(tf-t0)))
        scores = self.oracle.score(query) # score Samples

        updateDataset(self.params, query, scores) # add scored Samples to dataset
        #self.bestScores.append(np.amin(self.sampleDict['energy'])) # find the lowest score of all the sampling runs


    def retrainModels(self, parallel=True):
        if not parallel:
            for i in range(self.params['ensemble size']):
                self.resetModel(i)  # reset between ensemble estimators EVERY ITERATION of the pipeline
                self.model.converge()  # converge model
                self.testMinima.append(np.amin(self.model.err_te_hist))
        else:
            cpus = int(np.amin((self.params['ensemble size'], os.cpu_count() - 1)))  # np.min((os.cpu_count()-2,params['runs']))
            pool = mp.Pool(cpus)
            for i in range(int(np.ceil(self.params['ensemble size'] / cpus))):
                output = [pool.apply_async(trainModel, args=[self.params, j + i]) for j in range(cpus)]
                if i > 0:
                    for j in range(cpus):
                        outputList.append(output[j].get())
                        self.testMinima.append([np.amin(outputList[i]) for i in range(cpus)])
                else:
                    outputList = [output[i].get() for i in range(cpus)]
                    self.testMinima.append([np.amin(outputList[i]) for i in range(cpus)])

        print(f'Model ensemble training converged with average test loss of {bcolors.OKGREEN}%.5f{bcolors.ENDC}' % np.average(np.asarray(self.testMinima[-self.params['ensemble size']:])))


    def getQuery(self):
        self.loadEstimatorEnsemble()
        query = self.querier.buildQuery(self.model)  # pick Samples to be scored

        return query


    def evaluateSampler(self, gamma, useOracle):
        '''
        load the best model and run the sampler
        :return:
        '''
        self.sampler = sampler(self.params, 0, [1, 0])
        samplerDict = self.sampler.sample(self.model, gamma, useOracle)  # identify interesting sequences
        samples, scores, energies, variances = [samplerDict['optimalSamples'],samplerDict['optima'],samplerDict['enAtOptima'],samplerDict['varAtOptima']]
        return samples, scores, energies, variances


    def sampleOracle(self):
        '''
        for toy models
        do global optimization directly on the oracle to find the true minimum
        :return:
        '''
        print("Asking toy oracle for the true minimum")
        self.model = 'abc'
        self.oracleOptima = self.sampleEnsemble(useOracle=True)


    def sampleEnsemble(self, useOracle=False):
        '''
        run an ensemble of samplers with different hyperparameters (gammas)
        and identify the most interesting datapoints
        :return:
        '''
        sampleList = []
        scoreList = []
        energiesList = []
        variancesList = []

        indRec = []
        scoreRec = []
        stunRec = []
        accRec = []
        tempRec = []

        gammas = np.logspace(-3,1,self.params['sampler gammas'])
        for i in range(self.params['sampler gammas']):
            samples, scores, energies, variances = self.evaluateSampler(gammas[i],useOracle)
            sampleList.append(samples)
            scoreList.append(scores)
            energiesList.append(energies)
            variancesList.append(variances)
            if self.params['debug'] == True: # if we are debugging, save the sampling stats
                indRec.append(self.sampler.recInds) # time index
                scoreRec.append(self.sampler.scoreRec) # raw score
                stunRec.append(self.sampler.stunRec) # stun function
                accRec.append(self.sampler.accRec) # acceptance ratio
                tempRec.append(self.sampler.tempRec) # temperature


        sampleDict = {}
        sampleDict['samples'] = np.concatenate(sampleList)
        sampleDict['scores'] = np.concatenate(scoreList)
        sampleDict['energy'] = np.concatenate(energiesList)
        sampleDict['variance'] = np.concatenate(variancesList)

        bestMin = np.amin(sampleDict['energy'])
        print(f"Sampling Complete! Lowest Energy Found = {bcolors.FAIL}%.3f{bcolors.ENDC}" % bestMin + " from %d" %self.params['sampler gammas'] + " sampling runs.")

        return sampleDict


    def getModel(self):
        '''
        initialize model and check for prior checkpoints
        :return:
        '''
        self.model = model(self.params)
        #print(f'{bcolors.HEADER} New model: {bcolors.ENDC}', getDirName(self.params))


    def loadEstimatorEnsemble(self):
        '''
        load all the trained models at their best checkpoints
        and initialize them in an ensemble model where they can all be queried at once
        :return:
        '''
        ensemble = []
        for i in range(1,self.params['ensemble size'] + 1):
            self.resetModel(i)
            self.model.load(i)
            ensemble.append(self.model.model)

        del self.model
        self.model = model(self.params,0)
        self.model.loadEnsemble(ensemble)


    def loadModelCheckpoint(self):
        '''
        load most recent converged model checkpoint
        :return:
        '''
        self.model.load()


    def resetModel(self,ensembleIndex, returnModel = False):
        '''
        load a new instance of the model with reset parameters
        :return:
        '''
        try: # if we have a model already, delete it
            del self.model
        except:
            pass
        self.model = model(self.params,ensembleIndex)
        #print(f'{bcolors.HEADER} New model: {bcolors.ENDC}', getModelName(ensembleIndex))
        if returnModel:
            return self.model


    def saveOutputs(self):
        '''
        save params and outputs in a dict
        :return:
        '''
        outputDict = {}
        outputDict['params'] = self.params
        if self.params['dataset'] == 'toy':
            outputDict['oracle outputs'] = self.oracleOptima
        # outputDict['sample outputs'] = self.sampleDict
        #outputDict['best optima found'] = self.bestScores
        outputDict['model test minima'] = self.testMinima
        np.save('outputsDict',outputDict)
        # outputDict = np.load('outputDict.npy',allow_pickle=True)
        # outputDict = outputDict.item()
        #print('Pipeline Complete: Best optimum found %.3f '%np.amin(self.bestScores) + 'after %d' %int(self.params['pipeline iterations'] * self.params['queries per iter']) + ' queries')


    def plotIterations(self):
        '''
        plot high-level results of each iteration
        1) test loss as function of iteration
        2) minimum discovered energy as function of iteration
        :return:
        '''

        plt.figure(0)
        plt.subplot(1,2,1)
        plt.cla()
        plt.plot(self.testMinima,'o-')
        plt.xlabel('Iteration')
        plt.ylabel('Minimum Test Loss')
        plt.title('Model Performance')
        plt.subplot(1,2,2)
        plt.cla()
        plt.plot(self.bestScores,'o-')
        if self.params['dataset'] == 'toy':
            oracleMinimum = np.amin(self.oracleOptima['scores'])
            plt.plot(np.arange(len(self.bestScores)), np.ones(len(self.bestScores)) * oracleMinimum)

        plt.xlabel('Iteration')
        plt.ylabel('Minimum Sampled State')
        plt.title('Best Discovered Optima')



def trainModel(params, i):
    '''
    rewritten for training in a parallelized fashion
    needs to be outside the class method for multiprocessing to work
    :param i:
    :return:
    '''

    seqModel = model(params, i)
    err_te_hist = seqModel.converge(returnHist = True)  # converge model

    return err_te_hist