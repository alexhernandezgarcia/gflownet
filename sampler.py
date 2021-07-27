'''import statements'''
from utils import *
from models import getDataSize
from oracle import *
import os

#import nolds

#from oracle import *

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
        self.params['STUN'] = 1
        self.params['Target Acceptance Rate'] = 0.234
        self.chainLength = getDataSize(self.params)
        self.temperature = 1
        self.deltaIter = int(100)  # get outputs every this many of iterations with one iteration meaning one move proposed for each "particle" on average
        self.randintsResampleAt = int(1e4)  # larger takes up more memory but increases speed
        self.scoreFunction = scoreFunction
        self.seedInd = seedInd

        self.initialize()

        if self.params['dataset'] == 'toy':
            self.oracle = oracle(self.params) # if we are using a toy model, initialize the oracle so we can optimize it directly for comparison

    def __call__(self, model):
        return self.converge(model)


    def initialize(self):
        '''
        get initial condition
        :return:
        '''
        self.config = np.random.randint(0, 2, self.chainLength)


    def resampleRandints(self):
        self.spinRandints = np.random.randint(0, 2, size=self.randintsResampleAt).astype('uint8')
        self.pickSpinRandint = np.random.randint(0, self.chainLength, size=self.randintsResampleAt).astype('uint32')
        self.alphaRandoms = np.random.random(self.randintsResampleAt).astype(float)


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

        # has to be a different state or we just skip the cycle
        if not all(propConfig == self.config):
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

                if scores[0] < self.E0: # if we have found a new minimum
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

