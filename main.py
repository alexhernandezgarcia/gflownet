'''import statements'''
from utils import *
from models import *
from sampler import *
from indicator import *
from oracle import *
import os
import glob
import tqdm
from shutil import copyfile

'''
This code implements an active learning protocol, intended to optimize the binding score of aptamers to certain analytes

Modules:
> Controller
--> This script
--> Contains parameters and controls which modules are used

> Model
--> Inputs: dataset
--> Outputs: scores and confidence factors

> Sampler
--> Inputs: Model (as energy function)
--> Outputs: Uncertainty and/or score maxima

> Indicator
--> Inputs: Samples
--> Outputs: Sequences to be scored

> Oracle
--> Inputs: sequences
--> Outputs: binding scores
'''
'''
To-Do
==> Indicator
    ?-> upgrade to RL
==> model
    ?-> upgrade to pair-regression
?=> rebuild debug / plotting modes (partially done / partially unnecessary)
?=> get learner and sampler using the same (dynamic) scoring function
==> check that relevant params (ensemble size) are properly overwritten when picking up old jobs
?=> when we add back real datasets - eventually we'll need to initialize them
?=> parallelize training and sampling runs
?-> if we never see duplicates again, remove the report code in utils
==> write automated test and analysis routines
'''

# initialize control parameters
params = {}
params['device'] = 'local' # 'local' or 'cluster'

# get command line input
if params['device'] == 'cluster':
    params['run num'] = get_input()
elif params['device'] == 'local':
    params['run num'] = 0 # manual setting, for 0, do a fresh run, for != 0, pickup on a previous run.

# Pipeline parameters
params['pipeline iterations'] = 10 # number of cycles with the oracle
params['queries per iter'] = 20 # maximum number of questions we can ask the oracle per cycle
params['mode'] = 'training' # 'training'  'evaluation' 'initialize'
params['debug'] = False
params['plot results'] = False

if params['device'] == 'cluster':
    params['workdir'] = '/home/kilgourm/scratch/learnerruns'
elif params['device'] == 'local':
    params['workdir'] = 'C:/Users\mikem\Desktop/activeLearningRuns'

# Misc parameters
params['random seed'] = 1

# toy data parameters
params['dataset'] = 'toy'
params['init dataset length'] = 1000 # number of items in the initial (toy) dataset
params['sample length'] = 5 # number of input dimensions

# model parameters
params['ensemble size'] = 2 # number of models in the ensemble
params['model filters'] = 12
params['model layers'] = 2
params['max training epochs'] = 20
params['GPU'] = 0 # run model on GPU - not yet tested, may not work at all
params['batch_size'] = 10 # model training batch size

# sampler parameters
params['sampling time'] = 4e4
params['sampler runs'] = 5

if params['mode'] == 'evaluation':
    params['pipeline iterations'] = 1


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
        self.oracle = oracle(self.params)

        if self.params['run num'] == 0:
            self.makeNewWorkingDirectory()
            os.mkdir(self.workDir + '/ckpts')
            os.mkdir(self.workDir + '/datasets')
            # move to working dir
            os.chdir(self.workDir)
            #copyfile(self.params['dataset directory'] + '/' + self.params['dataset'],self.workDir + '/datasets/' + self.params['dataset'] + '.npy')
            self.oracle.initializeDataset(self.params['sample length'], self.params['init dataset length']) # generate toy model dataset
        else:
            # move to working dir
            self.workDir = self.params['workdir'] + '/' + 'run%d' %self.params['run num']
            os.chdir(self.workDir)

            self.workDir = self.params['workdir'] + '/' + 'run%d' %self.params['run num']

        self.learner = learner(self.params)


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
        self.testMinima = []
        self.bestScores = []

        if self.params['dataset'] == 'toy':
            self.sampleOracle() # use the oracle to pre-solve the problem
            print("The true global minimum is %.3f" % np.amin(self.oracleOptima['scores']))

        for i in range(params['pipeline iterations']):
            print(f'Starting pipeline iteration #{bcolors.OKGREEN}%d{bcolors.ENDC}' % int(i+1))
            self.params['iteration'] = i + 1
            self.iterate()

        self.saveOutputs()


    def iterate(self):
        '''
        run one iteration of the pipeline - train model, sample sequences, select sequences, consult oracle
        :return:
        '''
        for i in range(self.params['ensemble size']):
            self.resetModel(i) # reset between ensemble estimators EVERY ITERATION of the pipeline
            self.model.converge() # converge model
            self.testMinima.append(np.amin(self.model.err_te_hist))

        self.sampleDict = self.sampleEnsemble(False) # identify interesting sequences

        oracleSamples = self.learner.identifySequences(self.sampleDict) # pick sequences to be scored

        oracleScores = self.oracle.score(oracleSamples) # score sequences
        #oracleSequences = numbers2letters(oracleSequences)

        updateDataset(self.params, oracleSamples, oracleScores) # add scored sequences to dataset
        self.bestScores.append(np.amin(self.sampler.optima))

        if self.params['plot results'] == True:
            self.plotIterations()


    def evaluateSampler(self, gamma, useOracle):
        '''
        load the best model and run the sampler
        :return:
        '''
        self.sampler = sampler(self.params)
        self.loadEstimatorEnsemble()
        sampleSequences, sampleScores, sampleEnergy, sampleVariance = self.sampler.sample(self.model, gamma, useOracle)  # identify interesting sequences
        return sampleSequences, sampleScores, sampleEnergy, sampleVariance


    def sampleOracle(self):
        '''
        for toy models
        do global optimization directly on the oracle to find the true minimum
        :return:
        '''
        print("Asking oracle for the true minimum")
        self.oracleOptima = self.sampleEnsemble(True)


    def sampleEnsemble(self, useOracle):
        '''
        run an ensemble of samplers with different hyperparameters (gammas)
        and identify the most interesting datapoints
        :return:
        '''
        samples = []
        scores = []
        energy = []
        variance = []

        indRec = []
        scoreRec = []
        stunRec = []
        accRec = []
        tempRec = []

        gammas = np.logspace(-2,1,self.params['sampler runs'])
        for i in range(self.params['sampler runs']):
            sampleSequences, sampleScores, sampleEnergy, sampleVariance = self.evaluateSampler(gammas[i],useOracle)
            samples.append(sampleSequences)
            scores.append(sampleScores)
            energy.append(sampleEnergy)
            variance.append(sampleVariance)
            if self.params['debug'] == True: # if we are debugging, save the sampling stats
                indRec.append(self.sampler.recInds) # time index
                scoreRec.append(self.sampler.scoreRec) # raw score
                stunRec.append(self.sampler.stunRec) # stun function
                accRec.append(self.sampler.accRec) # acceptance ratio
                tempRec.append(self.sampler.tempRec) # temperature

        if self.params['debug'] == True:
            for i in range(5):
                plt.plot(indRec[i], np.clip(stunRec[i], 0, 1))

        sampleDict = {}
        sampleDict['samples'] = np.concatenate(samples)
        sampleDict['scores'] = np.concatenate(scores)
        sampleDict['energy'] = np.concatenate(energy)
        sampleDict['variance'] = np.concatenate(variance)

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


    def resetModel(self,ensembleIndex):
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


    def saveOutputs(self):
        '''
        save params and outputs in a dict
        :return:
        '''
        outputDict = {}
        outputDict['params'] = self.params
        if self.params['dataset'] == 'toy':
            outputDict['oracle outputs'] = self.oracleOptima
        outputDict['sample outputs'] = self.sampleDict
        outputDict['best optima found'] = self.bestScores
        outputDict['model test minima'] = self.testMinima
        np.save('outputsDict',outputDict)
        # outputDict = np.load('outputDict.npy',allow_pickle=True)
        # outputDict = outputDict.item()

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


if __name__ == '__main__':
    activeLearning = activeLearning(params)
    if params['mode'] == 'initalize':
        print("Initialized!")
    elif params['mode'] == 'training':
        activeLearning.runPipeline()
    elif params['mode'] == 'evaluation':
        sampleDict = activeLearning.sampleEnsemble(False)



'''
little script to build new dataset
oracle = oracle(params)
sequences = np.random.randint(0,4,size=(100000,40))
scores = oracle.score(sequences)
dict = {}
dict['samples'] = numbers2letters(sequences)
dict['scores'] = scores
np.save('datasets/dna_2',dict)
'''