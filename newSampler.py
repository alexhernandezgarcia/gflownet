"""import statements"""
from utils import *
from oracle import *
import numpy as np
from querier import querier
from sampler import sampler
import tqdm

class sampler2:
    """
    finds optimum values of the function defined by the model
    """

    def __init__(self, params, seedInd, scoreFunction, gammas):
        self.params = params
        self.params['STUN'] = 1
        self.params['Target Acceptance Rate'] = 0.234 # found this in a paper
        self.chainLength = self.params['max sample length']
        self.deltaIter = int(100)  # get outputs every this many of iterations with one iteration meaning one move proposed for each "particle" on average
        self.randintsResampleAt = int(1e4)  # larger takes up more memory but increases speed
        self.scoreFunction = scoreFunction
        self.seedInd = seedInd
        self.recordMargin = 0.1  # how close does a state have to be to the best found minimum to be recorded
        self.gammas = gammas
        self.nruns = len(gammas)
        self.temperature = [1 for _ in range(self.nruns)]


        if self.params['dataset type'] == 'toy':
            self.oracle = oracle(self.params)  # if we are using a toy model, initialize the oracle so we can optimize it directly for comparison

        self.getInitConfig()

        if self.params['debug']:
            self.initRecs()


    def __call__(self, model):
        return self.converge(model)


    def getInitConfig(self):
        """
        get initial condition
        :return:
        """
        np.random.seed(self.params['sampler seed'])
        self.config = np.random.randint(1, self.params['dict size'] + 1, size=(self.nruns,self.chainLength))


    def resetConfig(self,ind):
        """
        re-randomize a particular configuration
        :return:
        """
        self.config[ind,:] = np.random.randint(1, self.params['dict size'] + 1, size = (1, self.chainLength))


    def resampleRandints(self):
        """
        periodically resample our relevant random numbers
        :return:
        """
        self.spinRandints = np.random.randint(1, self.params['dict size'] + 1, size=(self.nruns,self.randintsResampleAt)).astype('uint8')
        self.pickSpinRandint = np.random.randint(0, self.chainLength, size=(self.nruns,self.randintsResampleAt)).astype('uint32')
        self.alphaRandoms = np.random.random((self.nruns,self.randintsResampleAt)).astype(float)
        self.changeLengthRandints = np.random.randint(-1, 2, size=(self.nruns,self.randintsResampleAt)).astype('int8')
        self.seqExtensionRandints = np.random.randint(1, self.params['dict size'] + 1, size=(self.nruns,self.randintsResampleAt)).astype('uint8')


    def initOptima(self, scores, energy, variance):
        """
        initialize the minimum energies
        :return:
        """
        self.optima = [[] for i in range(self.nruns)] # record optima of the score function
        self.enAtOptima = [[] for i in range(self.nruns)]  # record energies at the optima
        self.varAtOptima = [[] for i in range(self.nruns)]  # record of uncertainty at the optima
        self.optimalSamples = [[] for i in range(self.nruns)]  # record the optimal samples
        self.optimalInds = [[] for i in range(self.nruns)]
        self.recInds = [[] for i in range(self.nruns)]


        # set initial values
        self.E0 = scores[1]  # initialize the 'best score' value
        for i in range(self.nruns):
            self.optima[i].append(scores[1][i])
            self.enAtOptima[i].append(energy[1][i])
            self.varAtOptima[i].append(variance[1][i])
            self.optimalSamples[i].append(self.config[i])
            self.optimalInds[i].append(0)


    def initRecs(self):
        '''
        step-by-step records for debugging purposes
        :return:
        '''
        self.temprec = [[] for i in range(self.nruns)]
        self.accrec = [[] for i in range(self.nruns)]
        self.stunrec = [[] for i in range(self.nruns)]
        self.scorerec = [[] for i in range(self.nruns)]


    def initConvergenceStats(self):
        # convergence stats
        self.resetInd = [0 for i in range(self.nruns)]  # flag
        self.acceptanceRate = np.zeros(self.nruns) # rolling MCMC acceptance rate


    def computeSTUN(self, scores):
        """
        compute the STUN function for the given energies
        :return:
        """
        return 1 - np.exp(-self.gammas * (scores - self.E0))  # compute STUN function


    def sample(self, model, useOracle=False):
        """
        converge the sampling process
        :param model:
        :return:
        """
        self.converge(model, useOracle)
        return self.__dict__


    def converge(self, model, useOracle=False):
        """
        run the sampler until we converge to an optimum
        :return:
        """

        np.random.seed(int(self.params['sampler seed'] + self.seedInd))  # randomize each gamma ru

        self.initConvergenceStats()
        self.resampleRandints()
        for self.iter in tqdm.tqdm(range(self.params['sampling time'])):  # sample for a certain number of iterations
            self.iterate(model, useOracle)  # try a monte-carlo step!

            if (self.iter % self.deltaIter == 0) and (self.iter > 0):  # every N iterations do some reporting / updating
                self.updateAnnealing()  # change temperature or other conditions

            if self.iter % self.randintsResampleAt == 0: # periodically resample random numbers
                self.resampleRandints()


    def propConfigs(self,ind):
        """
        propose a new ensemble of configurations
        :param ind:
        :return:
        """
        propConfig = np.copy(self.config)
        for i in range(self.nruns):
            propConfig[i, self.pickSpinRandint[i,ind]] = self.spinRandints[i,ind]

            # propose changing sequence length
            if self.params['variable sample length']:
                if self.changeLengthRandints[i,ind] == 0:  # do nothing
                    pass
                else:
                    nnz = np.count_nonzero(propConfig[i])
                    if self.changeLengthRandints[i,ind] == 1:  # extend sequence by adding a new spin (nonzero element)
                        if nnz < self.params['max sample length']:
                            propConfig[i, nnz] = self.seqExtensionRandints[i, ind]
                    elif nnz == -1:  # shorten sequence by trimming the end (set last element to zero)
                        if nnz > self.params['min sample length']:
                            propConfig[i, nnz - 1] = 0

        return propConfig


    def iterate(self, model, useOracle):
        """
        run chainLength cycles of the sampler
        process: 1) propose state, 2) compute acceptance ratio, 3) sample against this ratio and accept/reject move
        :return: config, energy, and stun function will update
        """
        self.ind = self.iter % self.randintsResampleAt # random number index

        # propose a new state
        propConfig = self.propConfigs(self.ind)

        # even if it didn't change, just run it anyway (big parallel - to hard to disentangle)
        # compute acceptance ratio
        scores, energy, variance = self.getScores(propConfig, self.config, model, useOracle)

        try:
            self.E0
        except:
            self.initOptima(scores, energy, variance)  # if we haven't already assigned E0, initialize everything

        F, DE = self.getDE(scores)

        acceptanceRatio = np.minimum(1, np.exp(-DE / self.temperature))

        # accept or reject
        for i in range(self.nruns):
            if self.alphaRandoms[i, self.ind] < acceptanceRatio[i]:  # accept
                self.config[i] = np.copy(propConfig[i])
                self.recInds[i].append(self.iter)

                if (scores[0][i] < self.E0[i]):  # if we have a new minimum, record it ((self.E0[i] - scores[0][i]) / self.E0[i] < self.recordMargin) # of if near a minimum
                    self.saveOptima(scores, energy, variance, propConfig, i)

        if self.params['debug']: # record a bunch of detailed outputs
            self.recordStats(scores, F)

    def getDE(self, scores):
        if self.params['STUN'] == 1:  # compute score difference using STUN
            F = self.computeSTUN(scores)
            DE = F[0] - F[1]
        else:  # compute raw score difference
            F = [0, 0]
            DE = scores[0] - scores[1]

        return F, DE


    def recordStats(self, scores, stunF):
        for i in range(self.nruns):
            self.temprec[i].append(self.temperature[i])
            self.accrec[i].append(self.acceptanceRate[i])
            self.stunrec[i].append(stunF[0][i])
            self.scorerec[i].append(scores[0][i])


    def getScores(self, propConfig, config, model, useOracle):
        """
        compute score against which we're optimizing
        :param propConfig:
        :param config:
        :return:
        """
        if useOracle:
            energy = [self.oracle.score(propConfig),self.oracle.score(config)]
            variance = [[0 for _ in range(len(energy[0]))], [0 for _ in range(len(energy[1]))]]
        else:
            # energy = model.evaluate(np.asarray([propConfig,config]),output="Average")
            # variance = model.evaluate(np.asarray([propConfig,config]),output="Variance")
            r1, r2 = [model.evaluate(np.asarray(config), output='Both'),model.evaluate(np.asarray(propConfig), output='Both')]
            energy = [r1[0], r1[1]]
            variance = [r2[0], r2[1]]

        score = self.scoreFunction[0] * np.asarray(energy) - self.scoreFunction[1] * np.asarray(variance)  # vary the relative importance of these two factors

        return score, energy, variance


    def saveOptima(self, scores, energy, variance, propConfig, ind):
        if scores[0][ind] < self.E0[ind]:
            self.E0[ind] = scores[0][ind]
        self.optima[ind].append(scores[0][ind])
        self.enAtOptima[ind].append(energy[0][ind])
        self.varAtOptima[ind].append(variance[0][ind])
        self.optimalInds[ind].append(self.iter)
        self.optimalSamples[ind].append(propConfig[ind])


    def updateAnnealing(self):
        """
        Following "Adaptation in stochatic tunneling global optimization of complex potential energy landscapes"
        1) updates temperature according to STUN threshold to separate "Local search" and "tunneling" phases
        2) determines when the algorithm is no longer efficiently searching and adapts by resetting the config to a random value
        """
        # 1) if rejection rate is too high, switch to tunneling mode, if it is too low, switch to local search mode
        # acceptanceRate = len(self.stunRec)/self.iter # global acceptance rate

        history = 100
        for i in range(self.nruns):
            acceptedRecently = np.sum((self.iter - np.asarray(self.recInds[i][-history:])) < history)  # rolling acceptance rate - how many accepted out of the last hundred iters
            self.acceptanceRate[i] = acceptedRecently / history

            if self.acceptanceRate[i] < self.params['Target Acceptance Rate']:
                self.temperature[i] = self.temperature[i] * (1 + np.random.random(1)[0] / 5) # modulate temperature semi-stochastically
            else:
                self.temperature[i] = self.temperature[i] * (1 - np.random.random(1)[0] / 5)

            # if we haven't found a new minimum in a long time, randomize input and do a temperature boost
            if (self.iter - self.resetInd[i]) > 1e4:  # within xx of the last reset
                if (self.iter - self.optimalInds[i][-1]) > 1e3: # haven't seen a new near-minimum in xx steps
                    self.resetInd[i] = self.iter
                    self.resetConfig(i)  # re-randomize
                    self.temperature[i] = self.temperature[i] * 2 # boost temperature


params = {}
params['model seed'] = 0  # seed used for model ensemble (each model gets a slightly different seed)
params['sampler seed'] = 0  # seed for MCMC modelling (each set of gammas gets a slightly different seed)
params['dataset seed'] = 0  # if we are using a toy dataset, it may take a specific seed
params['query mode'] = 'score'  # 'random', 'score', 'uncertainty', 'heuristic', 'learned' # different modes for query construction

# toy data parameters
params['debug'] = False
params['dataset'] = 'linear' # 'linear', 'inner product', 'potts', 'seqfold', 'nupack' in order of increasing complexity. Note distributions particulary for seqfold and nupack may not be natively well-behaved.
params['dataset type'] = 'toy' # oracle is very fast to sample
params['init dataset length'] = 1000 # number of items in the initial (toy) dataset
params['dict size'] = 4 # number of possible choices per-state, e.g., [0,1] would be two, [1,2,3,4] (representing ATGC) would be 4
params['variable sample length'] = True #if true, 'max sample length' should be a list with the smallest and largest size of input sequences [min, max]
params['min sample length'], params['max sample length'] = [10, 20] # minimum input sequence length and # maximum input sequence length (inclusive) - or fixed sample size if 'variable sample length' is false

# sampler parameters
params['sampling time'] = int(1e3)
params['sampler gammas'] = 3 # minimum number of gammas over which to search for each sampler (if doing in parallel, we may do more if we have more CPUs than this)

params['debug'] = False # records extra stats for debugging purposes (slow and heavy)
params['device'] = 'local'
gammas = np.logspace(-3, 1, params['sampler gammas'])

params['ensemble size'] = 5
params['run num'] = 27
params['explicit run enumeration'] = False
params['workdir'] = '/home/mkilgour/learnerruns'
# model parameters
params['ensemble size'] = 5 # number of models in the ensemble
params['model filters'] = 24
params['model layers'] = 3 # for cluster batching
params['embed dim'] = 4 # embedding dimension
params['max training epochs'] = 200
params['GPU'] = 0 # run model on GPU - not yet tested, may not work at all
params['batch size'] = 10 # model training batch size

from activeLearner import activeLearning
learner = activeLearning(params)
learner.loadEstimatorEnsemble()

t0 = time.time()
searcher = querier(params)
sampleDict = searcher.runSampling(learner.model,[1,0],1,parallel=True, useOracle=False)
tf = time.time()
print('sampler 1 took {} seconds'.format(int(tf-t0)))

t0 = time.time()
sampler2 = sampler2(params, 0, [1, 0], gammas)
dict = sampler2.sample(learner.model, useOracle=False)
tf = time.time()

print('sampler 2 took {} seconds'.format(int(tf-t0)))

''' # debug plots
plt.clf()
for i in range(len(dict['temprec'])):
    plt.plot(np.asarray(dict['scorerec'][i]))
plt.savefig('score')
plt.clf()
for i in range(len(dict['temprec'])):
    plt.plot(np.asarray(dict['stunrec'][i]))
plt.ylim(0,1)
plt.savefig('stun')
plt.clf()
for i in range(len(dict['temprec'])):
    plt.plot(np.asarray(dict['accrec'][i]))
plt.savefig('acc')
plt.clf()
for i in range(len(dict['temprec'])):
    plt.plot(np.asarray(dict['temprec'][i]))
plt.savefig('temp')
'''