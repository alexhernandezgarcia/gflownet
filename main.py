'''import statements'''
from utils import *
from models import model
from sampler import sampler
from indicator import learner
from oracle import oracle

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

To-Do
==> evaluate best sequences against oracle
==> implement records saving as a dict
==> test production mode (vs debug mode)
==> test GPU functionality
==> change seed if no new sequences are added

To-Do later
==> profile & optimize
'''

# initialize control parameters
params = {}

# Pipeline parameters
params['pipeline iterations'] = 3
params['mode'] = 'evaluate' # 'training'  'evaluate' 'initialize'
params['debug'] = 1

params['plot results'] = 1

# Misc parameters
params['random seed'] = 1

# model parameters
params['dataset'] = 'dna_simple_2'
params['model filters'] = 2
params['model layers'] = 1
params['max training epochs'] = 5
params['GPU'] = 0 # run model on GPU - not yet tested, may not work at all
params['batch_size'] = 1000 # model training batch size

# sampler parameters
params['sampling time'] = 1e3
params['sampler runs'] = 2

if params['mode'] == 'evaluate':
    params['pipeline iterations'] = 1


class activeLearning():
    def __init__(self, params):
        self.params = params
        self.getModel()


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
        self.resetModel() # reset it to avoid rapid overfit if new data is too small - better to just retrain the whole thing
        self.model.converge() # converge model

        self.model.load() # reload to best checkpoint
        self.sampler = sampler(self.params)
        sampleSequences, sampleScores, sampleUncertainty = self.sampler.sample(self.model) # identify interesting sequences

        self.learner = learner(self.params)
        oracleSequences = self.learner.identifySequences(sampleSequences, sampleScores, sampleUncertainty) # pick sequences to be scored

        self.oracle = oracle(self.params)
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
        self.model.load()
        self.sampler = sampler(self.params)
        sampleSequences, sampleScores, sampleUncertainty = self.sampler.sample(self.model)  # identify interesting sequences
        return sampleSequences, sampleScores, sampleUncertainty


    def getModel(self):
        '''
        initialize model and check for prior checkpoints
        :return:
        '''
        self.model = model(self.params)
        #print(f'{bcolors.HEADER} New model: {bcolors.ENDC}', getDirName(self.params))


    def loadModelCheckpoint(self):
        '''
        load most recent converged model checkpoint
        :return:
        '''
        self.model.load()


    def resetModel(self):
        '''
        load a new instance of the model with reset parameters
        :return:
        '''
        del self.model
        self.model = model(self.params)
        print(f'{bcolors.HEADER} New model: {bcolors.ENDC}', getDirName(self.params))


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