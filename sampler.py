'''import statements'''
from utils import *
from oracle import *
import tqdm

'''
This script uses Markov Chain Monte Carlo, including the STUN algorithm, to optimize a given function

> Inputs: model to be optimized
> Outputs: sequences representing model extrema in 0123 format

To-Do:
==> add different observable targets
==> vectorize this for parallel gammasjj
'''

class sampler():
    '''
    finds optimum values of the function defined by the model
    '''
    def __init__(self, params, seedInd, scoreFunction):
        self.params = params
        self.params['STUN'] = 1 # use STUN rather than plain annealing
        self.params['Target Acceptance Rate'] = 0.234
        self.chainLength = self.params['max sample length']
        self.temperature = 1
        self.deltaIter = int(100)  # get outputs every this many of iterations with one iteration meaning one move proposed for each "particle" on average
        self.randintsResampleAt = int(1e4)  # larger takes up more memory but increases speed
        self.scoreFunction = scoreFunction
        self.seedInd = seedInd
        self.recordMargin = 0.2 # how close does a state have to be to the best found minimum to be recorded

        self.initialize()

        if self.params['dataset type'] == 'toy':
            self.oracle = oracle(self.params) # if we are using a toy model, initialize the oracle so we can optimize it directly for comparison

    def __call__(self, model):
        return self.converge(model)


    def initialize(self):
        '''
        get initial condition
        :return:
        '''
        self.config = np.random.randint(1, self.params['dict size'] + 1, self.chainLength)


    def resampleRandints(self):
        self.spinRandints = np.random.randint(1, self.params['dict size'] + 1, size=self.randintsResampleAt).astype('uint8')
        self.pickSpinRandint = np.random.randint(0, self.chainLength, size=self.randintsResampleAt).astype('uint32')
        self.alphaRandoms = np.random.random(self.randintsResampleAt).astype(float)
        self.changeLengthRandints = np.random.randint(-1,2,size=self.randintsResampleAt).astype('uint8')
        self.seqExtensionRandints = np.random.randint(1, self.params['dict size'] + 1, size=self.randintsResampleAt).astype('uint8')


    def initOptima(self, scores, energy, variance):
        '''
        initialize the minimum energies
        :return:
        '''
        self.optima = [] # record optima of the score function
        self.enAtOptima = [] # record energies at the optima
        self.varAtOptima = [] # record of uncertainty at the optima
        self.optimalSamples = [] # record the optimal samples
        self.optimalInds = []

        # set initial values
        self.E0 = scores[1] # initialize the 'best score' value
        self.optima.append(scores[1])
        self.enAtOptima.append(energy[1])
        self.varAtOptima.append(variance[1])
        self.optimalSamples.append(self.config)
        self.optimalInds.append(0)


    def computeSTUN(self, scores):
        '''
        compute the STUN function for the given energies
        :return:
        '''
        return (1 - np.exp(-self.gamma * (scores - self.E0)))  # compute STUN function


    def sample(self,model, gamma, useOracle=False):
        '''
        converge the sampling process
        :param model:
        :return:
        '''
        self.converge(model, gamma, useOracle)
        return self.__dict__


    def converge(self, model, gamma, useOracle):
        '''
        run the sampler until we converge to an optimum
        :return:
        '''
        self.gamma = gamma

        np.random.seed(int(self.params['sampler seed'] + np.abs((np.log10(self.gamma) * 100))) + self.seedInd) # randomize each gamma ru

        self.initConvergenceStats()

        while (self.converged == 0) and (self.iter < self.params['sampling time']): # we'll just sample a certain amount of time for now
            if self.iter % self.randintsResampleAt == 0:
                self.resampleRandints()
                
            self.iterate(model, useOracle) # try a monte-carlo step!
            
            if (self.iter % self.deltaIter == 0) and (self.iter > 0): # every N iterations do some reporting / updating
                self.updateAnnealing() # change temperature or other conditions

            self.iter += 1


    def iterate(self, model, useOracle):
        '''
        run chainLength cycles of the sampler
        process: 1) propose state, 2) compute acceptance ratio, 3) sample against this ratio and accept/reject move
        :return: config, energy, and stun function will update
        '''
        self.ind = self.iter % self.randintsResampleAt

        # propose a new state
        propConfig = np.copy(self.config)
        propConfig[self.pickSpinRandint[self.ind]] = self.spinRandints[self.ind]

        # propose changing sequence length
        if self.params['variable sample size']:
            if self.changeLengthRandints[self.ind] == 0: # do nothing
                pass
            elif self.changeLengthRandints[self.ind] == 1: # extend sequence by adding a new spin
                if len(propConfig) < self.params['max sample length']:
                    propConfig = np.append(propConfig, self.seqExtensionRandints[self.ind])
            elif self.changeLengthRandints[self.ind] == -1: # shorten sequence by trimming the end
                if len(propConfig > self.params['min sample length']):
                    propConfig = np.delete(propConfig, -1)


        # has to be a different state or we just skip the cycle
        if (not all(propConfig[0:len(self.config)] == self.config[0:len(propConfig)]) or (len(propConfig) != len(self.config))): # account for variable lengths
            # compute acceptance ratio
            scores, energy, variance = self.getScores(propConfig, self.config, model, useOracle)

            try:
                self.E0
            except:
                self.initOptima(scores, energy, variance) # if we haven't already assigned E0, initialize everything

            F, DE = self.getDE(scores)

            acceptanceRatio = np.minimum(1, np.exp(-DE / self.temperature))

            # accept or reject
            if self.alphaRandoms[self.ind] < acceptanceRatio: #accept
                self.config = np.copy(propConfig)

                if ((self.E0 - scores[0])/self.E0 < self.recordMargin) or (scores[0] < self.E0): # if we are near the best minimum, record what happens
                    self.saveOptima(scores,energy,variance,propConfig)

                if self.params['debug'] == True:
                    self.recordStats(scores,F)


    def getDE(self,scores):
        if self.params['STUN'] == 1:  # compute score difference using STUN
            F = self.computeSTUN(scores)
            DE = F[0] - F[1]
        else:  # compute raw score difference
            F = [0, 0]
            DE = scores[0] - scores[1]

        return F, DE


    def getScores(self, propConfig, config, model, useOracle):
        '''
        compute score against which we're optimizing
        :param propConfig: 
        :param config: 
        :return: 
        '''
        if useOracle:
            energy = self.oracle.score([propConfig,config])
            variance = [0, 0]
        else:
            #energy = model.evaluate(np.asarray([propConfig,config]),output="Average")
            #variance = model.evaluate(np.asarray([propConfig,config]),output="Variance")
            energy, variance = model.evaluate(np.asarray([propConfig,config]),output='Both')

        score = self.scoreFunction[0] * np.asarray(energy) - self.scoreFunction[1] * np.asarray(variance) # vary the relative importance of these two factors

        return score, energy, variance


    def saveOptima(self,scores,energy,variance,propConfig):
        if scores[0] < self.E0:
            self.E0 = scores[0]
        self.optima.append(scores[0])
        self.enAtOptima.append(energy[0])
        self.varAtOptima.append(variance[0])
        self.optimalInds.append(self.iter)
        self.optimalSamples.append(propConfig)


    def initConvergenceStats(self):
        # convergence stats
        self.converged = 0 # set convergence flag
        self.iter = 0 # iteration number
        self.resetInd = 0 # flag

        self.tempRec = []
        self.stunRec = []
        self.scoreRec = []
        self.recInds = []
        self.accRec = []
        self.acceptanceRate = 0


    def recordStats(self,scores,F):
        self.tempRec.append(self.temperature)
        self.stunRec.append(F[0])
        self.scoreRec.append(scores[0])
        self.recInds.append(self.iter)
        self.accRec.append(self.acceptanceRate)


    def updateAnnealing(self):
        '''
        Following "Adaptation in stochatic tunneling global optimization of complex potential energy landscapes"
        1) updates temperature according to STUN threshold to separate "Local search" and "tunneling" phases
        2) determines when the algorithm is no longer efficiently searching and adapts by resetting the config to a random value
        '''
        #1) if rejection rate is too high, switch to tunneling mode, if it is too low, switch to local search mode
        #acceptanceRate = len(self.stunRec)/self.iter # global acceptance rate
        history = 100
        acceptedRecently = np.sum((self.iter - np.asarray(self.recInds[-history:])) < history) # rolling acceptance rate - how many accepted out of the last hundred iters
        self.acceptanceRate = acceptedRecently / history

        if self.acceptanceRate < self.params['Target Acceptance Rate']:
            self.temperature = self.temperature * (1 + np.random.random(1)/5)
        else:
            self.temperature = self.temperature * (1 - np.random.random(1)/5)

        #if True:#self.iter > 1e4: # if we've searched for a while already, do Detrended Fluctuation Analysis
        #    self.hursts.append(nolds.dfa(np.asarray(self.scoreRec[-int(100/self.params['Target Acceptance Rate']):])))

        # if we haven't found a new minimum in a long time, randomize input and do a temperature boost
        if (self.iter - self.resetInd) > 1e4: # within xx of the last reset
            if (self.iter - self.optimalInds[-1]) > 1e4:
                self.resetInd = self.iter
                self.initialize() # re-randomize
                self.temperature = self.temperature * 2


    def printFinalStats(self):
        '''
        print the minimum energy found
        :return:
        '''
        printRecord(f"Sampling Complete! Lowest Energy Found = {bcolors.OKGREEN}%.3f{bcolors.ENDC}" % np.amin(self.enAtOptima[-1]) + " after %d" %self.iter + " iterations.")


    def plotSampling(self):
        '''
        plot Monte Carlo energy (and/or eventually STUN function) and annealing schedule
        :return:
        '''
        if not 'iteration' in self.params:
            self.params['iteration'] = 1


        columns = min(5,self.params['pipeline iterations'])
        plt.figure(2)
        rows = max([1,(self.params['pipeline iterations'] // 5)])

        plt.subplot(rows, columns, self.params['iteration'])
        plt.plot(self.recInds, self.scoreRec, '.')
        plt.plot(self.optimalInds, self.optima,'o-')
        plt.title('Iteration #%d' % self.params['iteration'])
        plt.ylabel('Sample Energy')
        plt.xlabel('Sample Iterations')

        if self.params['STUN'] == 1:
            plt.figure(3)
            plt.subplot(rows, columns, self.params['iteration'])
            plt.plot(self.recInds, self.stunRec, '.', label='STUN')
            plt.plot(self.recInds, self.accRec, '-', label='Acceptance Ratio')
            plt.title('Iteration #%d' % self.params['iteration'])
            plt.xlabel('Sample Iterations')
            plt.legend()

        plt.figure(4)
        plt.subplot(rows, columns, self.params['iteration'])
        plt.semilogy(self.recInds, self.tempRec, '.')
        plt.title('Iteration #%d' % self.params['iteration'])
        plt.ylabel('Temperature')
        plt.xlabel('Sample Iterations')


class sampler2:
    """
    finds optimum values of the function defined by the model
    intrinsically parallel, rather than via multiprocessing
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
        self.recordMargin = 0.2  # how close does a state have to be to the best found minimum to be recorded
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
        self.enAtOptima = [[] for i in range(self.nruns)]  # record energies near the optima
        self.varAtOptima = [[] for i in range(self.nruns)]  # record of uncertainty at the optima
        self.optimalSamples = [[] for i in range(self.nruns)]  # record the optimal samples
        self.optimalInds = [[] for i in range(self.nruns)]
        self.recInds = [[] for i in range(self.nruns)]
        self.newOptima = [[] for i in range(self.nruns)] # new minima
        self.newOptimaEn = [[] for i in range(self.nruns)] # new minima


        # set initial values
        self.E0 = scores[1]  # initialize the 'best score' value
        for i in range(self.nruns):
            self.optima[i].append(scores[1][i])
            self.enAtOptima[i].append(energy[1][i])
            self.varAtOptima[i].append(variance[1][i])
            self.newOptima[i].append(self.config[i])
            self.newOptimaEn[i].append(energy[1][i])
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
            if self.params['variable sample size']:
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

                newBest = False
                if ((self.E0[i] - scores[0][i]) / self.E0[i] < self.recordMargin):  # if we have a new minimum, record it  # or if near a minimum
                    if  (scores[0][i] < self.E0[i]):
                        newBest = True
                    self.saveOptima(scores, energy, variance, propConfig, i, newBest)


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


    def saveOptima(self, scores, energy, variance, propConfig, ind, newBest):
        equals = np.sum(propConfig[ind] == self.optimalSamples[ind],axis=1)
        if (not any(equals == propConfig[ind].shape[-1])) or newBest: # if there are no copies or we know it's a new minimum, record it # bit slower to keep checking like this but saves us checks later
            self.optima[ind].append(scores[0][ind])
            self.enAtOptima[ind].append(energy[0][ind])
            self.varAtOptima[ind].append(variance[0][ind])
            self.optimalSamples[ind].append(propConfig[ind])
        if newBest:
            self.E0[ind] = scores[0][ind]
            self.newOptima[ind].append(propConfig[ind])
            self.newOptimaEn[ind].append(energy[0][ind])
            self.optimalInds[ind].append(self.iter)


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
            if (self.iter - self.resetInd[i]) > 5e3:  # within xx of the last reset
                if (self.iter - self.optimalInds[i][-1]) > 1e3: # haven't seen a new near-minimum in xx steps
                    self.resetInd[i] = self.iter
                    self.resetConfig(i)  # re-randomize
                    self.temperature[i] = self.temperature[i] * 2 # boost temperature
