'''import statements'''
from utils import *
from models import getDataSize
import os
import multiprocessing as mp

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
    def __init__(self, params):
        self.params = params
        self.params['STUN'] = 1
        self.params['Target Acceptance Rate'] = 0.234
        self.chainLength = getDataSize(self.params)
        self.temperature = 1
        self.gamma = 0.1#np.linspace(0,1,self.params['sampler runs'])
        self.deltaIter = int(100)  # get outputs every this many of iterations with one iteration meaning one move proposed for each "particle" on average
        self.randintsResampleAt = int(1e4)  # larger takes up more memory but increases speed

        np.random.seed(params['random seed'])
        self.initialize()

        #self.oracle = oracle(self.params)

    def __call__(self, model):
        return self.converge(model)

    def initialize(self):
        '''
        get initial condition
        :return:
        '''
        self.config = np.random.randint(0, 4, self.chainLength)


    def resampleRandints(self):
        self.spinRandints = np.random.randint(0, 4, size=self.randintsResampleAt).astype('uint8')
        self.pickSpinRandint = np.random.randint(0, self.chainLength, size=self.randintsResampleAt).astype('uint32')
        self.alphaRandoms = np.random.random(self.randintsResampleAt).astype(float)


    def initEmin(self, scores, config):
        '''
        initialize the minimum energies
        :return:
        '''
        self.emins = [] # record of lowest energies
        self.eminSequences = [] # record the lowest energy sequences
        self.eminInds = []

        self.E0 = scores[1]
        self.emins.append(scores[1])
        self.eminSequences.append(config)
        self.eminInds.append(0)


    def computeSTUN(self, scores):
        '''
        compute the STUN function for the given energies
        :return:
        '''
        return (1 - np.exp(-self.gamma * (scores - self.E0)))  # compute STUN function


    def sample(self,model):
        '''
        converge the sampling process
        :param model:
        :return:
        '''
        return self.converge(model)


    def converge(self, model):
        '''
        run the sampler until we converge to an optimum
        :return:
        '''
        self.converged = 0 # set convergence flag
        self.iter = 0 # iteration number
        self.resetInd = 0 # flag

        # convergence stats
        self.tempRec = []
        self.stunRec = []
        self.enRec = []
        self.recInds = []
        self.accRec = []
        self.hursts = []
        self.acceptanceRate = 0

        while (self.converged == 0) and (self.iter < self.params['sampling time']):
            if self.iter % self.randintsResampleAt == 0:
                self.resampleRandints()

            self.iterate(model)

            # every N iterations do some reporting / updating
            if (self.iter % self.deltaIter == 0) and (self.iter > 0):
                self.updateAnnealing()
                #self.reportRunningStats()
                #self.checkConvergence()

            self.iter += 1

        if self.params['debug'] == 1:
            self.plotSampling()

        self.printFinalStats()

        return self.eminSequences, self.emins, [] ### this is where we will put the uncertainty flag for exceptional sequences


    def iterate(self, model):
        '''
        run chainLength cycles of the sampler
        process: 1) propose state, 2) compute acceptance ratio, 3) sample against this ratio and accept/reject move
        :return: config, energy, and stun function will update
        '''
        self.ind = self.iter % self.randintsResampleAt

        # propose a new state
        prop_config = np.copy(self.config)
        prop_config[self.pickSpinRandint[self.ind]] = self.spinRandints[self.ind]

        # has to be a different state or we just skip the cycle
        if not all(prop_config == self.config):

            # compute acceptance ratio
            scores = model.evaluate(np.asarray([prop_config,self.config])).cpu().detach().numpy()
            # run with the oracle - for testing purposes
            #scores = self.oracle.score([prop_config, self.config])
            try:
                self.E0
            except:
                self.initEmin(scores, self.config)

            if self.params['STUN'] == 1:
                F = self.computeSTUN(scores)
                DE = F[0] - F[1]
            else:
                F = [0,0]
                DE = scores[0]-scores[1]

            acceptanceRatio = np.minimum(1, np.exp(-DE / self.temperature))

            # accept or reject
            if self.alphaRandoms[self.ind] < acceptanceRatio: #accept
                self.config = np.copy(prop_config)

                if scores[0] < self.E0: # if we have found a new minimum
                    self.E0 = scores[0]
                    self.emins.append(scores[0])
                    self.eminInds.append(self.iter)
                    self.eminSequences.append(prop_config)

                if self.params['debug'] == 1:
                    self.recordStats(scores,F)


        ### this is where we'd put flags for storing other exceptional samples


    def recordStats(self,scores,F):
        self.tempRec.append(self.temperature)
        self.stunRec.append(F[0])
        self.enRec.append(scores[0])
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
        #    self.hursts.append(nolds.dfa(np.asarray(self.enRec[-int(100/self.params['Target Acceptance Rate']):])))

        # if we haven't found a new minimum in a long time, randomize input and do a temperature boost
        if (self.iter - self.resetInd) > 1e4: # within xx of the last reset
            if (self.iter - self.eminInds[-1]) > 1e4:
                self.resetInd = self.iter
                self.initialize() # re-randomize
                self.temperature = self.temperature * 2


    def reportRunningStats(self):
        '''
        record simulation statistics
        :return:
        '''


    def printFinalStats(self):
        '''
        print the minimum energy found
        :return:
        '''
        print(f"Sampling Complete! Lowest Energy Found = {bcolors.FAIL}%.3f{bcolors.ENDC}" % np.amin(self.E0) + " after %d" %self.iter + " iterations.")


    def checkConvergence(self):
        '''
        check simulation records against convergence criteria
        :return:
        '''


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
        plt.plot(self.recInds, self.enRec, '.')
        plt.plot(self.eminInds, self.emins,'o-')
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

