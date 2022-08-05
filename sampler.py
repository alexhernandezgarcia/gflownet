'''import statements'''
from utils import *
from oracle import *
import tqdm

'''
This script uses Markov Chain Monte Carlo, including the STUN algorithm, to optimize a given function

> Inputs: model to be optimized
> Outputs: sequences representing model extrema in 1234... format with 0 padding for variable-length sequences

'''


class Sampler:
    """
    finds optimum values of the function defined by the model
    intrinsically parallel, rather than via multiprocessing
    """

    def __init__(self, config, seedInd, scoreFunction, gammas):
        self.config_main = config
        self.config_main.STUN = 1
        self.config_main.target_acceptance_rate = 0.234 # found this in a paper
        self.chainLength = self.config_main.dataset.max_length
        self.deltaIter = int(10)  # get outputs every this many of iterations with one iteration meaning one move proposed for each "particle" on average
        self.randintsResampleAt = int(1e4)  # larger takes up more memory but increases speed
        self.scoreFunction = scoreFunction
        self.seedInd = seedInd
        self.recordMargin = 0.2  # how close does a state have to be to the best found minimum to be recorded
        self.gammas = gammas
        self.nruns = len(gammas)
        self.temp0 = 0.1 # initial temperature for sampling runs
        self.temperature = [self.temp0 for _ in range(self.nruns)]


        if self.config_main.dataset.type == 'toy':
            self.oracle = Oracle(
                seed = self.config_main.seeds.dataset,
                seq_len = self.config_main.dataset.max_length,
                dict_size = self.config_main.dataset.dict_size,
                min_len = self.config_main.dataset.min_length,
                max_len = self.config_main.dataset.max_length,
                oracle = self.config_main.dataset.oracle,
                variable_len = self.config_main.dataset.variable_length,
                init_len = self.config_main.dataset.init_length,
                energy_weight = self.config_main.dataset.nupack_energy_reweighting,
                nupack_target_motif = self.config_main.dataset.nupack_target_motif,
                seed_toy = self.config_main.seeds.toy_oracle,
            ) # if we are using a toy model, initialize the oracle so we can optimize it directly for comparison

        np.random.seed(int(self.config_main.seeds.sampler + int(self.seedInd * 1000))) # initial seed is randomized over pipeline iterations

        self.getInitConfig()

        if self.config_main.debug:
            self.initRecs()


    def __call__(self, model):
        return self.converge(model)


    def getInitConfig(self):
        """
        get initial condition
        :return:
        """
        self.config = []
        for i in range(self.nruns):
            self.config.append(self.makeAConfig())

        self.config = np.asarray(self.config)


    def makeAConfig(self):
        '''
        initialize a random config with appropriate padding
        :return:
        '''

        if self.config_main.dataset.variable_length and self.config_main.dataset.min_length < self.config_main.dataset.max_length:
            # TODO: this should be max + 1
            randChainLen = np.random.randint(self.config_main.dataset.min_length,self.config_main.dataset.max_length)
            randConfig = np.random.randint(1, self.config_main.dataset.dict_size + 1, size = (1, randChainLen))
            if randChainLen < self.config_main.dataset.max_length: # add zero padding, if necessary
                randConfig = np.pad(randConfig[0],[0, self.config_main.dataset.max_length - randChainLen],mode='constant')
        else:
            randConfig = np.random.randint(1,self.config_main.dataset.dict_size + 1, size = (self.config_main.dataset.max_length))

        return randConfig

    def resetConfig(self,ind):
        """
        re-randomize a particular configuration
        :return:
        """

        self.config[ind,:] = self.makeAConfig()


    def resampleRandints(self):
        """
        periodically resample our relevant random numbers
        :return:
        """
        self.spinRandints = np.random.randint(1, self.config_main.dataset.dict_size + 1, size=(self.nruns,self.randintsResampleAt)).astype('uint8')
        self.pickSpinRandint = np.random.randint(0, self.chainLength, size=(self.nruns,self.randintsResampleAt)).astype('uint32')
        self.alphaRandoms = np.random.random((self.nruns,self.randintsResampleAt)).astype(float)
        self.changeLengthRandints = np.random.randint(-1, 2, size=(self.nruns,self.randintsResampleAt)).astype('int8')
        self.seqExtensionRandints = np.random.randint(1, self.config_main.dataset.dict_size + 1, size=(self.nruns,self.randintsResampleAt)).astype('uint8')


    def initOptima(self, scores, energy, std_dev):
        """
        initialize the minimum energies
        :return:
        """
        # trajectory
        self.all_scores = np.zeros((self.run_iters, self.nruns)) # record optima of the score function
        self.all_energies = np.zeros_like(self.all_scores)  # record energies near the optima
        self.all_uncertainties = np.zeros_like(self.all_scores)  # record of uncertainty at the optima
        self.all_samples = np.zeros((self.run_iters, self.nruns, self.config_main.dataset.max_length))

        # optima
        self.new_optima_inds = [[] for i in range(self.nruns)]
        self.recInds = [[] for i in range(self.nruns)]
        self.new_optima_scores = [[] for i in range(self.nruns)] # new minima
        self.new_optima_energies = [[] for i in range(self.nruns)] # new minima
        self.new_optima_samples = [[] for i in range(self.nruns)] # new minima


        # set initial values
        self.E0 = scores[1]  # initialize the 'best score' value
        self.absMin = np.amin(np.array(self.E0))
        self.all_scores[0] = scores[1]
        self.all_energies[0] = energy[1]
        self.all_uncertainties[0] = std_dev[1]
        self.all_samples[0] = self.config

        for i in range(self.nruns):
            self.new_optima_samples[i].append(self.config[i])
            self.new_optima_energies[i].append(energy[1][i])
            self.new_optima_scores[i].append(scores[1][i])
            self.new_optima_inds[i].append(0)



    def initRecs(self):
        '''
        step-by-step records for debugging purposes
        :return:
        '''
        self.temprec = [[] for i in range(self.nruns)]
        self.accrec = [[] for i in range(self.nruns)]
        self.stunrec = [[] for i in range(self.nruns)]
        self.scorerec = [[] for i in range(self.nruns)]
        self.enrec = [[] for i in range(self.nruns)]
        self.std_devrec = [[] for i in range(self.nruns)]


    def initConvergenceStats(self):
        # convergence stats
        self.resetInd = [0 for i in range(self.nruns)]  # flag
        self.acceptanceRate = np.zeros(self.nruns) # rolling MCMC acceptance rate


    def computeSTUN(self, scores):
        """
        compute the STUN function for the given energies
        :return:
        """
        return 1 - np.exp(-self.gammas * (scores - self.absMin))  # compute STUN function with shared global minimum


    def sample(self, model, useOracle=False, nIters = None):
        """
        converge the sampling process
        :param model:
        :return:
        """
        self.converge(model, useOracle, nIters = nIters)

        outputs = {
            "samples": np.concatenate(self.all_samples),
            "scores": np.concatenate(self.all_scores),
            "energies": np.concatenate(self.all_energies),
            "uncertainties": np.concatenate(self.all_uncertainties),
        }

        return outputs


    def converge(self, model, useOracle=False, nIters = False):
        """
        run the sampler until we converge to an optimum
        :return:
        """
        self.initConvergenceStats()
        self.resampleRandints()

        if nIters is None:
            self.run_iters = self.config_main.mcmc.sampling_time
        else:
            self.run_iters = nIters


        for self.iter in tqdm.tqdm(range(self.run_iters)):  # sample for a certain number of iterations
            self.iterate(model, useOracle)  # try a monte-carlo step!

            if (self.iter % self.deltaIter == 0) and (self.iter > 0):  # every N iterations do some reporting / updating
                self.updateAnnealing()  # change temperature or other conditions

            if self.iter % self.randintsResampleAt == 0: # periodically resample random numbers
                self.resampleRandints()

        printRecord("{} samples were recorded on this run".format(len(np.concatenate(self.all_samples))))


    def postSampleAnnealing(self, initConfigs, model, useOracle=False, seed = 0):
        '''
        run a sampling run with the following characteristics
        - low temperature so that we quickly crash to global minimum
        - no STUN function
        - instead of many parallel stun functions, many parallel initial configurations
        - return final configurations for each run as 'annealed samples'
        '''
        np.random.seed(seed)
        self.config_main.STUN = 0
        self.nruns = len(initConfigs)
        self.temp0 = 0.01 # initial temperature for sampling runs - start low and shrink
        self.temperature = [self.temp0 for _ in range(self.nruns)]
        self.config = initConfigs # manually overwrite configs
        self.run_iters = self.config_main.al.annealing_time

        self.initConvergenceStats()
        self.resampleRandints()
        for self.iter in tqdm.tqdm(range(self.run_iters)):
            self.iterate(model, useOracle)

            self.temperature = [temperature * 0.99 for temperature in self.temperature] # cut temperature at every time step

            if self.iter % self.randintsResampleAt == 0: # periodically resample random numbers
                self.resampleRandints()

        evals = self.getScores(self.config, self.config, model, useOracle=useOracle)
        annealedOutputs = {
            'samples': self.config,
            'scores': evals[0][0],
            'energies': evals[1][0],
            'uncertainties': evals[2][0]
        }

        return annealedOutputs


    def propConfigs(self,ind):
        """
        propose a new ensemble of configurations
        :param ind:
        :return:
        """
        self.propConfig = np.copy(self.config)
        for i in range(self.nruns):
            self.propConfig[i, self.pickSpinRandint[i,ind]] = self.spinRandints[i,ind]

            # propose changing sequence length
            if self.config_main.dataset.variable_length:
                if self.changeLengthRandints[i,ind] == 0:  # do nothing
                    pass
                else:
                    nnz = np.count_nonzero(self.propConfig[i])
                    if self.changeLengthRandints[i,ind] == 1:  # extend sequence by adding a new spin (nonzero element)
                        if nnz < self.config_main.dataset.max_length:
                            self.propConfig[i, nnz] = self.seqExtensionRandints[i, ind]
                    elif nnz == -1:  # shorten sequence by trimming the end (set last element to zero)
                        if nnz > self.config_main.dataset.min_length:
                            self.propConfig[i, nnz - 1] = 0


    def iterate(self, model, useOracle):
        """
        run chainLength cycles of the sampler
        process: 1) propose state, 2) compute acceptance ratio, 3) sample against this ratio and accept/reject move
        :return: config, energy, and stun function will update
        """
        self.ind = self.iter % self.randintsResampleAt # random number index

        # propose a new state
        self.propConfigs(self.ind)

        # even if it didn't change, just run it anyway (big parallel - to hard to disentangle)
        # compute acceptance ratio
        self.scores, self.energy, self.std_dev = self.getScores(self.propConfig, self.config, model, useOracle)

        if self.iter == 0: # initialize optima recording
            self.initOptima(self.scores, self.energy, self.std_dev)

        self.F, self.DE = self.getDelta(self.scores)
        self.acceptanceRatio = np.minimum(1, np.exp(-self.DE / self.temperature))
        self.updateConfigs()


    def updateConfigs(self):
        '''
        check Metropolis conditions, update configurations, and record statistics
        :return:
        '''
        # accept or reject
        for i in range(self.nruns):
            if self.alphaRandoms[i, self.ind] < self.acceptanceRatio[i]:  # accept
                self.config[i] = np.copy(self.propConfig[i])
                self.recInds[i].append(self.iter)

                if (self.scores[0][i] < self.E0[i]):
                    self.updateBest(i)

                #self.recordTrajectory(i) # if we accept the move, update the trajectory

        self.recordTrajectory()  # if we accept the move, update the trajectory

        if self.config_main.debug: # record a bunch of detailed outputs
            self.recordStats()


    def recordTrajectory(self):
        self.all_scores[self.iter] = self.scores[0]
        self.all_energies[self.iter] = self.energy[0]
        self.all_uncertainties[self.iter] = self.std_dev[0]
        self.all_samples[self.iter] = self.propConfig


    def getDelta(self, scores):
        if self.config_main.STUN == 1:  # compute score difference using STUN
            F = self.computeSTUN(scores)
            DE = F[0] - F[1]
        else:  # compute raw score difference
            F = [0, 0]
            DE = scores[0] - scores[1]

        return F, DE


    def recordStats(self):
        for i in range(self.nruns):
            self.temprec[i].append(self.temperature[i])
            self.accrec[i].append(self.acceptanceRate[i])
            self.scorerec[i].append(self.scores[0][i])
            self.enrec[i].append(self.energy[0][i])
            self.std_devrec[i].append(self.std_dev[0][i])
            if self.config_main.STUN:
                self.stunrec[i].append(self.F[0][i])


    def getScores(self, propConfig, config, model, useOracle):
        """
        compute score against which we're optimizing
        :param propConfig:
        :param config:
        :return:
        """
        if useOracle:
            energy = [self.oracle.score(propConfig),self.oracle.score(config)]
            std_dev = [[0 for _ in range(len(energy[0]))], [0 for _ in range(len(energy[1]))]]
            score = self.scoreFunction[0] * np.asarray(energy) - self.scoreFunction[1] * np.asarray(std_dev)  # vary the relative importance of these two factors
        else:
            if self.config_main.al.query_mode == 'fancy_acquisition':
                score1, out1, std1 = model.evaluate(np.asarray(config), output='fancy_acquisition')
                score2, out2, std2 = model.evaluate(np.asarray(propConfig), output='fancy_acquisition')
                energy = [out2,out1]
                std_dev = [std2,std1]
                score = [score2, score1]

            else: # manually specify score function
                r1, r2 = [model.evaluate(np.asarray(config), output='Both'),model.evaluate(np.asarray(propConfig), output='Both')] # two model evaluations, each returning score and variance for a propConfig or config
                energy = [r2[0], r1[0]]
                std_dev = [r2[1], r1[1]]

                # energy and variance both come out standardized against the training dataset
                score = self.scoreFunction[0] * np.asarray(energy) - self.scoreFunction[1] * np.asarray(std_dev)  # vary the relative importance of these two factors

        return score, energy, std_dev



    def updateBest(self,ind):
        self.E0[ind] = self.scores[0][ind]
        self.absMin = self.E0[ind]
        self.new_optima_samples[ind].append(self.propConfig[ind])
        self.new_optima_energies[ind].append(self.energy[0][ind])
        self.new_optima_scores[ind].append(self.scores[0][ind])
        self.new_optima_inds[ind].append(self.iter)

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

            if self.acceptanceRate[i] < self.config_main.target_acceptance_rate:
                self.temperature[i] = self.temperature[i] * (1 + np.random.random(1)[0]) # modulate temperature semi-stochastically
            else:
                self.temperature[i] = self.temperature[i] * (1 - np.random.random(1)[0])

            # if we haven't found a new minimum in a long time, randomize input and do a temperature boost
            if (self.iter - self.resetInd[i]) > 1e3:  # within xx of the last reset
                if (self.iter - self.new_optima_inds[i][-1]) > 1e3: # haven't seen a new near-minimum in xx steps
                    self.resetInd[i] = self.iter
                    self.resetConfig(i)  # re-randomize
                    self.temperature[i] = self.temp0 # boost temperature

