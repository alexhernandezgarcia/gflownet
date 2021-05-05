'''import statements'''
from utils import *
from models import *
from sampler import *
from indicator import *
from oracle import *
import os
import glob
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
==> check for dataset duplicates BEFORE going to oracle
==> more sophisticated oracle option selection routine
==> parallel sampling with multiple gammas
==> implement records saving as a dict
==> test production mode (vs debug mode)
==> change seed if we can't suggest new samples for oracle
==> incorporate pipeline convergence stats
==> update training plots for ensemble of models
==> upgrade uncertainty calculation from simple variance to isolate epistemic uncertainty
==> check that relevant params (ensemble size) are properly overwritten when picking up jobs
==> add oracle budget including batch size to params
==> when we add back real datasets - eventually we'll need to initialize them
'''

# initialize control parameters
params = {}
params['device'] = 'cluster' # 'local' or 'cluster'

# get command line input
if params['device'] == 'cluster':
    params['run num'] = get_input()
elif params['device'] == 'local':
    params['run num'] = 0 # manual setting, for 0, do a fresh run, for != 0, pickup on a previous run.

# Pipeline parameters
params['pipeline iterations'] = 15
params['mode'] = 'evaluate' # 'training'  'evaluate' 'initialize'
params['debug'] = 1
params['plot results'] = 1
if params['device'] == 'cluster':
    params['workdir'] = '/home/kilgourm/scratch/learnerruns'
elif params['device'] == 'local':
    params['workdir'] = 'C:/Users\mikem\Desktop/activeLearningRuns'

# Misc parameters
params['random seed'] = 1

# toy data parameters
params['dataset'] = 'toy1'
params['init dataset length'] = 1000 # number of items in the initial dataset
params['sample length'] = 10 # number of input dimensions

# model parameters
params['ensemble size'] = 5 # number of models in the ensemble
params['model filters'] = 12
params['model layers'] = 2
params['max training epochs'] = 10
params['GPU'] = 0 # run model on GPU - not yet tested, may not work at all
params['batch_size'] = 10 # model training batch size

# sampler parameters
params['sampling time'] = 4e4
params['sampler runs'] = 1

if params['mode'] == 'evaluate':
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
        for i in range(params['pipeline iterations']):
            print(f'Starting pipeline iteration #{bcolors.OKGREEN}%d{bcolors.ENDC}' % int(i+1))
            self.params['iteration'] = i + 1
            self.iterate()


    def iterate(self):
        '''
        run one iteration of the pipeline - train model, sample sequences, select sequences, consult oracle
        :return:
        '''
        for i in range(self.params['ensemble size']):
            self.resetModel(i) # reset between ensemble estimators EVERY ITERATION of the pipeline
            self.model.converge() # converge model

        sampleSequences, sampleScores, sampleUncertainty = self.evaluateSampler() # identify interesting sequences

        oracleSequences = self.learner.identifySequences(sampleSequences, sampleScores, sampleUncertainty) # pick sequences to be scored

        oracleScores = self.oracle.score(oracleSequences) # score sequences
        oracleSequences = numbers2letters(oracleSequences)

        updateDataset(self.params, oracleSequences, oracleScores) # add scored sequences to dataset

        if self.params['plot results'] == 1:
            self.plotIterations()


    def evaluateSampler(self):
        '''
        load the best model and run the sampler
        :return:
        '''
        self.sampler = sampler(self.params)
        self.loadEstimatorEnsemble()
        sampleSequences, sampleScores, sampleUncertainty = self.sampler.sample(self.model)  # identify interesting sequences
        return sampleSequences, sampleScores, sampleUncertainty


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
        print(f'{bcolors.HEADER} New model: {bcolors.ENDC}', getModelName(ensembleIndex))


    def plotIterations(self):
        '''
        plot high-level results of each iteration
        1) test loss as function of iteration
        2) minimum discovered energy as function of iteration
        :return:
        '''
        self.testMinima.append(np.amin(self.model.err_te_hist))
        self.bestScores.append(np.amin(self.sampler.emins))

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
        plt.xlabel('Iteration')
        plt.ylabel('Minimum Sampled State')
        plt.title('Best Discovered Optima')


if __name__ == '__main__':
    activeLearning = activeLearning(params)
    if params['mode'] == 'initalize':
        print("Iintialized!")
    elif params['mode'] == 'training':
        activeLearning.runPipeline()
    elif params['mode'] == 'evaluate':
        sampleSequences, sampleScores, sampleUncertainty = activeLearning.evaluateSampler()



'''
little script to build new dataset
oracle = oracle(params)
sequences = np.random.randint(0,4,size=(100000,40))
scores = oracle.score(sequences)
dict = {}
dict['sequences'] = numbers2letters(sequences)
dict['scores'] = scores
np.save('datasets/dna_2',dict)
'''