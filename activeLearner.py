from comet_ml import Experiment
from models import modelNet
from querier import *
from sampler import *
from gflownet import batch2dict
from utils import namespace2dict
from torch.utils import data
import torch.nn.functional as F
import torch
from Agent import ParameterUpdateAgent
from replay_buffer import ParameterUpdateReplayMemory
import pandas as pd
from oracle import Oracle
import numpy
import os
import glob


class ActiveLearning():
    def __init__(self, config):
        self.pipeIter = None
        self.homedir = os.getcwd()
        self.episode = 0
        self.config = config
        self.runNum = self.config.run_num
        self.oracle = Oracle(
            seed = self.config.seeds.dataset,
            seq_len = self.config.dataset.max_length,
            dict_size = self.config.dataset.dict_size,
            min_len = self.config.dataset.min_length,
            max_len = self.config.dataset.max_length,
            oracle = self.config.dataset.oracle,
            variable_len = self.config.dataset.variable_length,
            init_len = self.config.dataset.init_length,
            energy_weight = self.config.dataset.nupack_energy_reweighting,
            nupack_target_motif = self.config.dataset.nupack_target_motif,
            seed_toy = self.config.seeds.toy_oracle,
        ) # oracle needs to be initialized to initialize toy datasets
        self.agent = ParameterUpdateAgent(self.config)
        self.querier = Querier(self.config) # might as well initialize the querier here
        self.setup()
        self.getModelSize()
        # Comet
        if config.al.comet.project:
            self.comet = Experiment(
                project_name=config.al.comet.project, display_summary_level=0,
            )
            if config.al.comet.tags:
                if isinstance(config.al.comet.tags, list):
                    self.comet.add_tags(config.al.comet.tags)
                else:
                    self.comet.add_tag(config.al.comet.tags)

            self.comet.set_name("run {}".format(config.run_num))

            self.comet.log_parameters(vars(config))
            with open(Path(self.workDir) / "comet_al.url", "w") as f:
                f.write(self.comet.url + "\n")
        else:
            self.comet = None
        # Save YAML config
        with open(self.workDir + '/config.yml', 'w') as f:
            yaml.dump(numpy2python(namespace2dict(self.config)), f, default_flow_style=False)

    def setup(self):
        '''
        setup working directory
        move to relevant directory
        :return:
        '''
        if self.config.run_num == 0: # if making a new workdir
            self.makeNewWorkingDirectory()
            self.reset()
        elif (self.config.explicit_run_enumeration == True):
            self.workDir = self.config.workdir + '/run%d'%self.config.run_num # explicitly enumerate the new run directory
            os.mkdir(self.workDir)
            self.reset()
        else:
            # move to working dir
            self.workDir = self.config.workdir + '/' + 'run%d' %self.config.run_num
            os.chdir(self.workDir)
            printRecord('Resuming run %d' % self.config.run_num)


    def reset(self):
        os.chdir(self.homedir)
        os.mkdir(f'{self.workDir}/ckpts')
        os.mkdir(f'{self.workDir}/episode{self.episode}')
        os.mkdir(f'{self.workDir}/episode{self.episode}/ckpts')
        os.mkdir(f'{self.workDir}/episode{self.episode}/datasets')
        os.chdir(f'{self.workDir}/episode{self.episode}') # move to working dir
        printRecord('Starting Fresh Run %d' %self.runNum)
        self.oracle.initializeDataset() # generate toy model dataset
        self.stateDict = None
        self.totalLoss = None
        self.testMinima = None
        self.stateDictRecord = None
        self.reward = None
        self.terminal = None
        self.model = None
        self.cumulative_reward = None
        self.reward_list = None
        self.bottomTenLoss = None
        self.action = None
        self.trueMinimum = None
        self.oracleRecord = None
        self.bestScores = None
        self.prev_iter_best = None

    def makeNewWorkingDirectory(self):    # make working directory
        '''
        make a new working directory
        non-overlapping previous entries
        :return:
        '''
        workdirs = glob.glob(self.config.workdir + '/' + 'run*') # check for prior working directories
        if len(workdirs) > 0:
            prev_runs = []
            for i in range(len(workdirs)):
                prev_runs.append(int(workdirs[i].split('run')[-1]))

            prev_max = max(prev_runs)
            self.workDir = self.config.workdir + '/' + 'run%d' %(prev_max + 1)
            self.config.workdir = self.workDir
            os.mkdir(self.workDir)
            self.runNum = int(prev_max + 1)
        else:
            self.workDir = self.config.workdir + '/' + 'run1'
            os.mkdir(self.workDir)

    def runPipeline(self):
        '''
        run  the active learning pipeline for a number of iterations
        :return:
        '''
        self.config.dataset_size = self.config.dataset.init_length
        for _ in range(self.config.al.episodes):

            if self.config.dataset.type == 'toy':
                self.sampleOracle() # use the oracle to pre-solve the problem for future benchmarking

            self.testMinima = [] # best test loss of models, for each iteration of the pipeline
            self.bestScores = [] # best optima found by the sampler, for each iteration of the pipeline

            for self.pipeIter in range(self.config.al.n_iter):
                printRecord(f'Starting pipeline iteration #{bcolors.FAIL}%d{bcolors.ENDC}' % int(self.pipeIter+1))
                if self.pipeIter == (self.config.al.n_iter - 1):
                    self.terminal = 1
                else:
                    self.terminal = 0
                self.iterate() # run the pipeline
                self.saveOutputs() # save pipeline outputs


            # Train Policy Network - for learned AL acquisition function / policy only
            # self.agent.train(BATCH_SIZE=self.config.al.q_batch_size)
            #self.policy_error = self.agent.policy_error
            #if self.config.al.episodes > (self.episode + 1): # if we are doing multiple al episodes
            #    self.reset()
            #    self.episode += 1
            #Save Memory for Agent architecture testing
            #numpy.save(f'{self.workDir}/memory.npy', self.agent.memory.memory)
            #numpy.save(f'{self.workDir}/agent_error.npy', self.agent.policy_error)


    def iterate(self):
        '''
        run one iteration of the pipeline - train model, sample sequences, select sequences, consult oracle
        :return:
        '''

        t0 = time.time()
        self.retrainModels()
        printRecord('Retraining took {} seconds'.format(int(time.time()-t0)))

        t0 = time.time()
        self.getModelState(self.terminal) # run energy-only sampling and create model state dict
        self.getDatasetReward()
        printRecord('Model state calculation took {} seconds'.format(int(time.time()-t0)))


        if self.terminal == 0: # skip querying if this is our final pipeline iteration

            t0 = time.time()
            query = self.querier.buildQuery(self.model, self.stateDict, action=self.action, comet=self.comet)  # pick Samples to be scored
            printRecord('Query generation took {} seconds'.format(int(time.time()-t0)))

            t0 = time.time()
            energies = self.oracle.score(query) # score Samples
            printRecord('Oracle scoring took {} seconds'.format(int(time.time()-t0)))
            printRecord('Oracle scored' + bcolors.OKBLUE + ' {} '.format(len(energies)) + bcolors.ENDC + 'queries with average score of' + bcolors.OKGREEN + ' {:.3f}'.format(np.average(energies)) + bcolors.ENDC + ' and minimum score of {:.3f}'.format(np.amin(energies)))
            self.updateDataset(query, energies) # add scored Samples to dataset

            if self.comet: # report query scores to comet
                self.comet.log_histogram_3d(energies,name='query energies',step=self.pipeIter)


        # CODE FOR LEARNED POLICY
        #if self.config.al.hyperparams_learning:# and (self.pipeIter > 0):
        #    model_state_prev, model_state_curr = self.agent.updateModelState(self.stateDict, self.model)
        #    if model_state_prev is not None:
        #        self.agent.push_to_buffer(model_state_prev, self.action, model_state_curr, self.reward, self.terminal)
        #    self.action = self.agent.getAction()
        #else:
        #    self.action = None


    def getModelState(self, terminal):
        '''
        sample the model
        report on the status of dataset
        report on best scores according to models
        report on model confidence
        :return:
        '''

        # run the sampler
        self.loadEstimatorEnsemble()
        if terminal: # use the query-generating sampler for terminal iteration
            sampleDict = self.querier.runSampling(self.model, scoreFunction = [1, 0], al_iter = self.pipeIter) # sample existing optima using standard sampler
        else: # use a cheap sampler for mid-run model state calculations
            sampleDict = self.querier.runSampling(self.model, scoreFunction = [1, 0], al_iter = self.pipeIter,  method_overwrite = 'random') # sample existing optima cheaply with random + annealing

        sampleDict = filterOutputs(sampleDict)

        # we used to do clustering here, now strictly argsort direct from the sampler
        sort_inds = np.argsort(sampleDict['energies']) # sort by energy
        samples = sampleDict['samples'][sort_inds][:self.config.querier.model_state_size] # top-k samples from model state run
        energies = sampleDict['energies'][sort_inds][:self.config.querier.model_state_size]
        uncertainties = sampleDict['uncertainties'][sort_inds][:self.config.querier.model_state_size]

        # get distances to relevant datasets
        internalDist, datasetDist, randomDist = self.getDataDists(samples)
        self.getModelStateReward(energies, uncertainties)

        self.stateDict = {
            'test loss': np.average(self.testMinima), # losses are evaluated on standardized data, so we do not need to re-standardize here
            'test std': np.sqrt(np.var(self.testMinima)),
            'all test losses': self.testMinima,
            'best energies': energies, # these are already standardized #(energies - self.model.mean) / self.model.std, # standardize according to dataset statistics
            'best uncertanties': uncertainties, # these are already standardized #uncertainties / self.model.std,
            'best samples': samples,
            'best samples internal diff': internalDist,
            'best samples dataset diff': datasetDist,
            'best samples random set diff': randomDist,
            'clustering cutoff': self.config.al.minima_dist_cutoff, # could be a learned parameter
            'n proxy models': self.config.proxy.ensemble_size,
            'iter': self.pipeIter,
            'budget': self.config.al.n_iter,
            'model state reward': self.model_state_reward
        }

        printRecord('%d '%self.config.proxy.ensemble_size + f'Model ensemble training converged with average test loss of {bcolors.OKCYAN}%.5f{bcolors.ENDC}' % np.average(np.asarray(self.testMinima[-self.config.proxy.ensemble_size:])) + f' and std of {bcolors.OKCYAN}%.3f{bcolors.ENDC}'%(np.sqrt(np.var(self.testMinima[-self.config.proxy.ensemble_size:]))))
        printRecord('Model state contains {} samples'.format(self.config.querier.model_state_size) +
                    ' with minimum energy' + bcolors.OKGREEN + ' {:.2f},'.format(np.amin(energies)) + bcolors.ENDC +
                    ' average energy' + bcolors.OKGREEN +' {:.2f},'.format(np.average(energies[:self.config.querier.model_state_size])) + bcolors.ENDC +
                    ' and average std dev' + bcolors.OKCYAN + ' {:.2f}'.format(np.average(uncertainties[:self.config.querier.model_state_size])) + bcolors.ENDC)
        printRecord("Best sample in model state is {}".format(numbers2letters(samples[np.argmin(energies)])))
        printRecord('Sample average mutual distance is ' + bcolors.WARNING +'{:.2f} '.format(np.average(internalDist)) + bcolors.ENDC +
                    'dataset distance is ' + bcolors.WARNING + '{:.2f} '.format(np.average(datasetDist)) + bcolors.ENDC +
                    'and overall distance estimated at ' + bcolors.WARNING + '{:.2f}'.format(np.average(randomDist)) + bcolors.ENDC)


        if self.config.al.large_model_evaluation: # we can quickly check the test error against a huge random dataset
            self.largeModelEvaluation()
            if self.comet:
                self.comet.log_metric(name='proxy loss on best 10% of large random dataset',value = self.bottomTenLoss[0], step=self.pipeIter)
                self.comet.log_metric(name='proxy loss on large random dataset', value = self.totalLoss[0], step=self.pipeIter)

        if self.pipeIter == 0: # if it's the first round, initialize, else, append
            self.stateDictRecord = [self.stateDict]
        else:
            self.stateDictRecord.append(self.stateDict)


        if self.comet:
            self.comet.log_histogram_3d(sampleDict['energies'], name='model state total sampling run energies', step = self.pipeIter)
            self.comet.log_histogram_3d(sampleDict['uncertainties'], name='model state total sampling run std deviations', step = self.pipeIter)
            self.comet.log_histogram_3d(energies[:self.config.querier.model_state_size], name='model state energies', step=self.pipeIter)
            self.comet.log_histogram_3d(uncertainties[:self.config.querier.model_state_size], name='model state std deviations', step=self.pipeIter)
            self.comet.log_histogram_3d(internalDist, name='model state internal distance', step=self.pipeIter)
            self.comet.log_histogram_3d(datasetDist, name='model state distance from dataset', step=self.pipeIter)
            self.comet.log_histogram_3d(randomDist, name='model state distance from large random sample', step=self.pipeIter)
            self.comet.log_histogram_3d(self.testMinima[-1], name='proxy model test minima', step=self.pipeIter)

        self.logTopK(sampleDict, prefix = "Model state ")


    def getModelStateReward(self,bestEns,bestStdDevs):
        '''
        print the performance of the learner against a known best answer
        :param bestEns:
        :param bestVars:
        :return:
        '''
        # get the best results in the standardized basis
        best_ens_standardized = (bestEns - self.model.mean)/self.model.std
        standardized_standard_deviations = bestStdDevs / self.model.std
        adjusted_standardized_energies = best_ens_standardized + standardized_standard_deviations # consider std dev as an uncertainty envelope and take the high end
        best_standardized_adjusted_energy = np.amin(adjusted_standardized_energies)

        # convert to raw outputs basis
        adjusted_energies = bestEns + bestStdDevs
        best_adjusted_energy = np.amin(adjusted_energies) # best energy, adjusted for uncertainty
        if self.pipeIter == 0:
            self.model_state_reward = 0 # first iteration - can't define a reward
            self.model_state_cumulative_reward = 0
            self.model_state_reward_list = np.zeros(self.config.al.n_iter)
            self.model_state_prev_iter_best = [best_adjusted_energy]
        else: # calculate reward using current standardization
            stdprev_iter_best = (self.model_state_prev_iter_best[-1] - self.model.mean)/self.model.std
            self.model_state_reward = -(best_standardized_adjusted_energy - stdprev_iter_best) # reward is the delta between variance-adjusted energies in the standardized basis (smaller is better)
            self.model_state_reward_list[self.pipeIter] = self.model_state_reward
            self.model_state_cumulative_reward = sum(self.model_state_reward_list)
            self.model_state_prev_iter_best.append(best_adjusted_energy)
            printRecord('Iteration best uncertainty-adjusted result = {:.3f}, previous best = {:.3f}, reward = {:.3f}, cumulative reward = {:.3f}'.format(best_adjusted_energy, self.model_state_prev_iter_best[-2], self.model_state_reward, self.model_state_cumulative_reward))

        if self.config.dataset.type == 'toy': # if it's  a toy dataset, report the cumulative performance against the known minimum
            stdTrueMinimum = (self.trueMinimum - self.model.mean) / self.model.std
            if self.pipeIter == 0:
                self.model_state_abs_score = [1 - np.abs(self.trueMinimum - best_adjusted_energy) / np.abs(self.trueMinimum)]
                self.model_state_cumulative_score=0
            elif self.pipeIter > 0:
                # we will compute the distance from our best answer to the correct answer and integrate it over the number of samples in the dataset
                xaxis = self.config.dataset_size + np.arange(0,self.pipeIter + 1) * self.config.al.queries_per_iter # how many samples in the dataset used for each
                self.model_state_abs_score.append(1 - np.abs(self.trueMinimum - best_adjusted_energy) / np.abs(self.trueMinimum)) # compute proximity to correct answer in standardized basis
                self.model_state_cumulative_score = np.trapz(y=np.asarray(self.model_state_abs_score), x=xaxis)
                self.model_state_normed_cumulative_score = self.model_state_cumulative_score / xaxis[-1]
                printRecord('Total score is {:.3f} and {:.5f} per-sample after {} samples'.format(self.model_state_abs_score[-1], self.model_state_normed_cumulative_score, xaxis[-1]))
            else:
                print('Error! Pipeline iteration cannot be negative')
                sys.exit()

            if self.comet:
                self.comet.log_metric(name = "model state absolute score", value = self.model_state_abs_score[-1], step = self.pipeIter)
                self.comet.log_metric(name = "model state cumulative score", value = self.model_state_cumulative_score, step = self.pipeIter)
                self.comet.log_metric(name = "model state reward", value = self.model_state_reward, step = self.pipeIter)



    def getDatasetReward(self):
        '''
        print the performance of the learner against a known best answer
        :param bestEns:
        :param bestVars:
        :return:
        '''
        dataset = np.load('datasets/' + self.config.dataset.oracle + '.npy', allow_pickle=True).item()
        energies = dataset['energies']

        printRecord("Best sample in dataset is {}".format(numbers2letters(dataset['samples'][np.argmin(dataset['energies'])])))

        best_energy = np.amin(energies)
        if self.pipeIter == 0:
            self.dataset_reward = 0 # first iteration - can't define a reward
            self.dataset_cumulative_reward = 0
            self.dataset_reward_list = np.zeros(self.config.al.n_iter)
            self.dataset_prev_iter_best = [best_energy]
        else: # calculate reward using current standardization
            self.dataset_reward = (best_energy - self.dataset_prev_iter_best[-1]) / self.dataset_prev_iter_best[-1] # reward is the delta between variance-adjusted energies in the standardized basis (smaller is better)
            self.dataset_reward_list[self.pipeIter] = self.dataset_reward
            self.dataset_cumulative_reward = sum(self.dataset_reward_list)
            self.dataset_prev_iter_best.append(best_energy)
            printRecord('Dataset evolution metrics = {:.3f}, previous best = {:.3f}, reward = {:.3f}, cumulative reward = {:.3f}'.format(best_energy, self.dataset_prev_iter_best[-2], self.dataset_reward, self.dataset_cumulative_reward))

        if self.config.dataset.type == 'toy': # if it's  a toy dataset, report the cumulative performance against the known minimum
            stdTrueMinimum = (self.trueMinimum - self.model.mean) / self.model.std
            if self.pipeIter == 0:
                self.dataset_abs_score = [1 - np.abs(self.trueMinimum - best_energy) / np.abs(self.trueMinimum)]
                self.dataset_cumulative_score=0
            elif self.pipeIter > 0:
                # we will compute the distance from our best answer to the correct answer and integrate it over the number of samples in the dataset
                xaxis = self.config.dataset_size + np.arange(0,self.pipeIter + 1) * self.config.al.queries_per_iter # how many samples in the dataset used for each
                self.dataset_abs_score.append(1 - np.abs(self.trueMinimum - best_energy) / np.abs(self.trueMinimum)) # compute proximity to correct answer in standardized basis
                self.dataset_cumulative_score = np.trapz(y=np.asarray(self.dataset_abs_score), x=xaxis)
                self.dataset_normed_cumulative_score = self.dataset_cumulative_score / xaxis[-1]
                printRecord('Dataset Total score is {:.3f} and {:.5f} per-sample after {} samples'.format(self.dataset_abs_score[-1], self.dataset_normed_cumulative_score, xaxis[-1]))
            else:
                print('Error! Pipeline iteration cannot be negative')
                sys.exit()

            if self.comet:
                self.comet.log_metric(name = "dataset absolute score", value = self.dataset_abs_score[-1], step = self.pipeIter)
                self.comet.log_metric(name = "dataset cumulative score", value = self.dataset_cumulative_score, step = self.pipeIter)
                self.comet.log_metric(name = "dataset reward", value = self.dataset_reward, step = self.pipeIter)


    def retrainModels(self):
        testMins = []
        for i in range(self.config.proxy.ensemble_size):
            self.resetModel(i)  # reset between ensemble estimators EVERY ITERATION of the pipeline
            self.model.converge()  # converge model
            testMins.append(np.amin(self.model.err_te_hist))
            if self.comet:
                tr_hist = self.model.err_tr_hist
                te_hist = self.model.err_te_hist
                epochs = len(te_hist)
                for i in range(epochs):
                    self.comet.log_metric('proxy train loss iter {}'.format(self.pipeIter), step=i, value=tr_hist[i])
                    self.comet.log_metric('proxy test loss iter {}'.format(self.pipeIter), step=i, value=te_hist[i])

        self.testMinima.append(testMins)


    def loadEstimatorEnsemble(self):
        '''
        load all the trained models at their best checkpoints
        and initialize them in an ensemble model where they can all be queried at once
        :return:
        '''
        ensemble = []
        for i in range(self.config.proxy.ensemble_size):
            self.resetModel(i)
            self.model.load(i)
            ensemble.append(self.model.model)

        del self.model
        self.model = modelNet(self.config,0)
        self.model.loadEnsemble(ensemble)
        self.model.getMinF()

        #print('Loaded {} estimators'.format(int(self.config.proxy.ensemble_size)))


    def resetModel(self,ensembleIndex, returnModel = False):
        '''
        load a new instance of the model with reset parameters
        :return:
        '''
        try: # if we have a model already, delete it
            del self.model
        except:
            pass
        self.model = modelNet(self.config,ensembleIndex)
        #printRecord(f'{bcolors.HEADER} New model: {bcolors.ENDC}', getModelName(ensembleIndex))
        if returnModel:
            return self.model


    def getModelSize(self):
        self.model = modelNet(self.config, 0)
        nParams = get_n_params(self.model.model)
        printRecord('Proxy model has {} parameters'.format(int(nParams)))
        del(self.model)


    def largeModelEvaluation(self):
        '''
        if we are using a toy oracle, we should be able to easily get the test loss on a huge sample of the dataset
        :return:
        '''
        self.loadEstimatorEnsemble()

        numSamples = min(int(1e3), self.config.dataset.dict_size ** self.config.dataset.max_length // 100) # either 1e5, or 1% of the sample space, whichever is smaller
        randomData = self.oracle.initializeDataset(save=False, returnData=True, customSize=numSamples) # get large random dataset
        randomSamples = randomData['samples']
        randomScores = randomData['energies']

        sortInds = np.argsort(randomScores) # sort randoms
        randomSamples = randomSamples[sortInds]
        randomScores = randomScores[sortInds]

        modelScores, modelStd = [[],[]]
        sampleLoader = data.DataLoader(randomSamples, batch_size = self.config.proxy.mbsize, shuffle=False, num_workers = 0, pin_memory=False)
        for i, testData in enumerate(sampleLoader):
            score, std_dev = self.model.evaluate(testData.float(), output='Both')
            modelScores.extend(score)
            modelStd.extend(std_dev)

        bestTenInd = numSamples // 10
        totalLoss = F.mse_loss((torch.Tensor(modelScores).float() - self.model.mean) / self.model.std, (torch.Tensor(randomScores).float() - self.model.mean) / self.model.std) # full dataset loss (standardized basis)
        bottomTenLoss = F.mse_loss((torch.Tensor(modelScores[:bestTenInd]).float() - self.model.mean) / self.model.std, (torch.Tensor(randomScores[:bestTenInd]).float() - self.model.mean) / self.model.std) # bottom 10% loss (standardized basis)

        if self.pipeIter == 0: # if it's the first round, initialize, else, append
            self.totalLoss = [totalLoss]
            self.bottomTenLoss = [bottomTenLoss]
        else:
            self.totalLoss.append(totalLoss)
            self.bottomTenLoss.append(bottomTenLoss)

        printRecord("Model has overall loss of" + bcolors.OKCYAN + ' {:.5f}, '.format(totalLoss) + bcolors.ENDC + 'best 10% loss of' + bcolors.OKCYAN + ' {:.5f} '.format(bottomTenLoss) + bcolors.ENDC +  'on {} toy dataset samples'.format(numSamples))


    def runPureSampler(self):
        ti = time.time()
        self.model = None
        self.pipeIter = 0
        if self.config.al.sample_method == 'mcmc':
            gammas = np.logspace(self.config.mcmc.stun_min_gamma, self.config.mcmc.stun_max_gamma, self.config.mcmc.num_samplers)
            mcmcSampler = Sampler(self.config, self.config.seeds.sampler, [1,0], gammas)
            sampleDict = mcmcSampler.sample(self.model, useOracle=True)  # do a genuine search
        elif self.config.al.sample_method == 'random':
            samples = generateRandomSamples(self.config.al.num_random_samples,
                                            [self.config.dataset.min_length,self.config.dataset.max_length],
                                            self.config.dataset.dict_size,
                                            variableLength = self.config.dataset.variable_length,
                                            seed = self.config.seeds.sampler)
            outputs = {
                'samples': samples,
                'energies': self.oracle.score(samples),
                'scores': np.zeros(len(samples)),
                'uncertainties': np.zeros(len(samples))
            }
            sampleDict = self.querier.doAnnealing([1,0], self.model, outputs, useOracle=True)
        elif self.config.al.sample_method == 'gflownet':
            gflownet = GFlowNetAgent(self.config, comet = self.comet, proxy=None, al_iter=0, data_path=None)

            t0 = time.time()
            gflownet.train()
            printRecord('Training GFlowNet took {} seconds'.format(int(time.time()-t0)))
            t0 = time.time()
            sample_batch, times = gflownet.sample_batch(gflownet.env, 
                self.config.gflownet.n_samples, train=False)
            sampleDict, times = batch2dict(sample_batch, gflownet.env, get_uncertainties=False, query_function=self.config.al.query_mode)
            printRecord('Sampling {} samples from GFlowNet took {} seconds'.format(self.config.gflownet.n_samples, int(time.time()-t0)))
            sampleDict['uncertainties'] = np.zeros(len(sampleDict['energies']))
            sampleDict = filterOutputs(sampleDict)

            if self.config.gflownet.annealing:
                sampleDict = self.querier.doAnnealing([1, 0], self.model, sampleDict, useOracle=True)


        sampleDict = filterOutputs(sampleDict) # remove duplicates
        # take only the top XX samples, for memory purposes
        maxLen = 10000
        if len(sampleDict['samples']) > maxLen:
            bestInds = np.argsort(sampleDict['energies'])[:maxLen]
            for key in sampleDict.keys():
                sampleDict[key] = sampleDict[key][bestInds]

        self.logTopK(sampleDict, prefix = "Pure sampling")

        # run clustering as a form of diversity analysis
        # more clusters means more diverse
        # this way won't penalize one (e.g., MCMC) for badly oversampling one area
        # only penalize it for not sampling *enough distinct areas*
        clusters, clusterScores, clusterVars = doAgglomerativeClustering(
            sampleDict['samples'], sampleDict['scores'],
            sampleDict['uncertainties'], self.config.dataset.dict_size,
            cutoff=self.config.al.minima_dist_cutoff)

        clusterDict = {
            'energies': np.asarray([np.amin(cluster_scores) for cluster_scores in clusterScores]),
            'samples': np.asarray([cluster[0] for cluster in clusters]) # this one doesn't matter
        }

        top_cluster_energies = self.logTopK(clusterDict, prefix = "Pure sampling - clusters", returnScores=True)

        # identify the clusters within XX% of the known global minimum
        global_minimum = min(np.amin(sampleDict['energies']), self.getTrueMinimum(sampleDict))
        found_minimum = np.amin(sampleDict['energies'])
        bottom_ranges = [10, 25, 50] # percent difference from known minimum
        abs_cluster_numbers = []
        rel_cluster_numbers = []
        for bottom_range in bottom_ranges:

            global_minimum_cutoff = global_minimum - bottom_range * global_minimum / 100
            found_minimum_cutoff = found_minimum - bottom_range * found_minimum / 100

            n_low_clusters1 = np.sum(clusterDict['energies'] < global_minimum_cutoff)
            n_low_clusters2 = np.sum(clusterDict['energies'] < found_minimum_cutoff)
            abs_cluster_numbers.append(n_low_clusters1)
            rel_cluster_numbers.append(n_low_clusters2)
            if self.comet:
                self.comet.log_metric("Number of clusters {} % from known minimum with {} cutoff".format(bottom_range, self.config.al.minima_dist_cutoff),
                                      n_low_clusters1)
                self.comet.log_metric("Number of clusters {} % from found minimum with {} cutoff".format(bottom_range, self.config.al.minima_dist_cutoff),
                                      n_low_clusters2)

        if self.comet:
            self.comet.log_histogram_3d(sampleDict['energies'], name="pure sampling energies", step=0)
            self.comet.log_metric("Best energy", np.amin(sampleDict['energies']))
            self.comet.log_metric("Proposed true minimum", self.trueMinimum)
            self.comet.log_metric("Best sample", numbers2letters(sampleDict['samples'][np.argmin(sampleDict["energies"])]))


        print("Key metrics:")
        print("Best found sample was {}".format(numbers2letters(sampleDict['samples'][np.argmin(sampleDict['energies'])])))
        print("Top K Cluster Energies {:.3f} {:.3f} {:.3f}".format(top_cluster_energies[0], top_cluster_energies[1], top_cluster_energies[2]))
        print("Top K Absolute # Clusters {} {} {}".format(abs_cluster_numbers[0], abs_cluster_numbers[1], abs_cluster_numbers[2]))
        print("Top K Relative # Clusters {} {} {}".format(rel_cluster_numbers[0], rel_cluster_numbers[1], rel_cluster_numbers[2]))
        print("Proposed True Global Minimum is {}".format(global_minimum))
        print("Pure sampling took a total of {} seconds".format(int(time.time()-ti)))

        return sampleDict

    def sampleOracle(self):
        '''
        for toy models
        do global optimization directly on the oracle to find the true minimum
        :return:
        '''
        printRecord("Asking toy oracle for the true minimum")

        self.model = 'abc'
        gammas = np.logspace(self.config.mcmc.stun_min_gamma,self.config.mcmc.stun_max_gamma,self.config.mcmc.num_samplers)
        mcmcSampler = Sampler(self.config, 0, [1,0], gammas)
        if (self.config.dataset.oracle == 'linear') or ('nupack' in self.config.dataset.oracle):
            sampleDict = mcmcSampler.sample(self.model, useOracle=True, nIters = 100) # do a tiny number of iters - the minimum is known
        else:
            sampleDict = mcmcSampler.sample(self.model, useOracle=True) # do a genuine search

        bestMin = self.getTrueMinimum(sampleDict)


        printRecord(f"Sampling Complete! Lowest Energy Found = {bcolors.FAIL}%.3f{bcolors.ENDC}" % bestMin + " from %d" % self.config.mcmc.num_samplers + " sampling runs.")
        printRecord("Best sample found is {}".format(numbers2letters(sampleDict['samples'][np.argmin(sampleDict['energies'])])))

        self.oracleRecord = sampleDict
        self.trueMinimum = bestMin

        if self.comet:
            self.comet.log_histogram_3d(sampleDict['energies'], name="energies_true",step=0)


    def getTrueMinimum(self, sampleDict):

        if self.config.dataset.oracle == 'wmodel': # w model minimum is always zero - even if we don't find it
            bestMin = 0
        else:
            bestMin = np.amin(sampleDict['energies'])

        if 'nupack' in self.config.dataset.oracle: # compute minimum energy for this length - for reweighting purposes
            goodSamples = np.ones((4, self.config.dataset.max_length)) * 4 # GCGC CGCG GGGCCC CCCGGG
            goodSamples[0,0:-1:2] = 3
            goodSamples[1,1:-1:2] = 3
            goodSamples[2,:self.config.dataset.max_length//2] = 3
            goodSamples[3,self.config.dataset.max_length//2:] = 3
            min_nupack_ens = self.oracle.score(goodSamples)

        # append suggestions for known likely solutions
        if self.config.dataset.oracle == "linear":
            goodSamples = np.zeros((4,self.config.dataset.max_length)) # all of one class usually best
            goodSamples[0] = goodSamples[1] + 1
            goodSamples[1] = goodSamples[1] + 2
            goodSamples[2] = goodSamples[2] + 3
            goodSamples[3] = goodSamples[3] + 4
            ens = self.oracle.score(goodSamples)
            if np.amin(ens) < bestMin:
                bestMin = np.amin(ens)
                printRecord("Pre-loaded minimum was better than one found by sampler")

        elif (self.config.dataset.oracle == "nupack energy"):
            if np.amin(min_nupack_ens) < bestMin:
                bestMin = np.amin(min_nupack_ens)
                printRecord("Pre-loaded minimum was better than one found by sampler")

        elif self.config.dataset.oracle == "nupack pairs":
            goodSamples = np.ones((4, self.config.dataset.max_length)) * 4 # GCGC CGCG GGGCCC CCCGGG
            goodSamples[0,0:-1:2] = 3
            goodSamples[1,1:-1:2] = 3
            goodSamples[2,:self.config.dataset.max_length//2] = 3
            goodSamples[3,self.config.dataset.max_length//2:] = 3
            ens = self.oracle.score(goodSamples)
            if np.amin(ens) < bestMin:
                bestMin = np.amin(ens)
                printRecord("Pre-loaded minimum was better than one found by sampler")

        elif self.config.dataset.oracle == "nupack pins":
            max_pins = self.config.dataset.max_length // 12 # a conservative estimate - 12 bases per stable hairpin
            if max_pins < bestMin:
                bestMin = max_pins
                printRecord("Pre-run guess was better than one found by sampler")

        elif self.config.dataset.oracle == "nupack open loop":
            biggest_loop = self.config.dataset.max_length - 8 # a conservative estimate - 8 bases for the stem (10 would be more conservative) and the rest are open
            if biggest_loop < bestMin:
                bestMin = biggest_loop
                printRecord("Pre-run guess was better than one found by sampler")

        elif self.config.dataset.oracle == 'nupack motif':
            bestMin = -1 # 100% agreement is the best possible

        return bestMin


    def saveOutputs(self):
        '''
        save config and outputs in a dict
        :return:
        '''
        outputDict = {}
        outputDict['config'] = Namespace(**dict(vars(self.config)))
        if "comet" in outputDict['config']:
            del outputDict['config'].comet
        outputDict['state dict record'] = self.stateDictRecord
        outputDict['model state rewards'] = self.model_state_reward_list
        outputDict['dataset rewards'] = self.dataset_reward_list
        if self.config.al.large_model_evaluation:
            outputDict['big dataset loss'] = self.totalLoss
            outputDict['bottom 10% loss'] = self.bottomTenLoss
        if self.config.dataset.type == 'toy':
            outputDict['oracle outputs'] = self.oracleRecord
            if self.pipeIter > 1:
                outputDict['model state score record'] = self.model_state_abs_score
                outputDict['model state cumulative score'] = self.model_state_cumulative_score,
                outputDict['model state per sample cumulative score'] = self.model_state_normed_cumulative_score
                outputDict['dataset score record'] = self.dataset_abs_score
                outputDict['dataset cumulative score'] = self.dataset_cumulative_score,
                outputDict['dataset per sample cumulative score'] = self.dataset_normed_cumulative_score
        np.save('outputsDict', outputDict)


    def updateDataset(self, oracleSequences, oracleScores):
        '''
        loads dataset, appends new datapoints from oracle, and saves dataset
        :param params: model parameters
        :param oracleSequences: sequences which were sent to oracle
        :param oracleScores: scores of sequences sent to oracle
        :return: n/a
        '''
        dataset = np.load('datasets/' + self.config.dataset.oracle + '.npy', allow_pickle=True).item()
        dataset['samples'] = np.concatenate((dataset['samples'], oracleSequences))
        dataset['energies'] = np.concatenate((dataset['energies'], oracleScores))

        self.logTopK(dataset, prefix = "Dataset") # log statistics on top K samples from the dataset

        self.config.dataset_size = len(dataset['samples'])

        printRecord(f"Added{bcolors.OKBLUE}{bcolors.BOLD} %d{bcolors.ENDC}" % int(len(oracleSequences)) + " to the dataset, total dataset size is" + bcolors.OKBLUE + " {}".format(int(len(dataset['samples']))) + bcolors.ENDC)
        printRecord(bcolors.UNDERLINE + "=====================================================================" + bcolors.ENDC)
        np.save('datasets/' + self.config.dataset.oracle, dataset)
        np.save('datasets/' + self.config.dataset.oracle + '_iter_{}'.format(self.pipeIter),dataset)

        if self.comet:
            self.comet.log_histogram_3d(dataset['energies'], name='dataset energies', step=self.pipeIter)
            dataset2 = dataset.copy()
            dataset2['samples'] = numbers2letters(dataset['samples'])
            self.comet.log_table(filename = 'dataset_at_iter_{}.csv'.format(self.pipeIter), tabular_data=pd.DataFrame.from_dict(dataset2))


    def logTopK(self, dataset, prefix, returnScores = False):
        if self.comet:
            self.comet.log_histogram_3d(dataset['energies'], name=prefix + ' energies', step=self.pipeIter)
            idx_sorted = np.argsort(dataset["energies"])
            top_scores = []
            for k in [1, 10, 100]:
                topk_scores = dataset["energies"][idx_sorted[:k]]
                topk_samples = dataset["samples"][idx_sorted[:k]]
                top_scores.append(np.average(topk_scores))
                dist = binaryDistance(topk_samples, pairwise=False, extractInds=len(topk_samples))
                self.comet.log_metric(prefix + f" mean top-{k} energies", np.mean(topk_scores), step=self.pipeIter)
                self.comet.log_metric(prefix + f" std top-{k} energies", np.std(topk_scores), step=self.pipeIter)
                self.comet.log_metric(prefix + f" mean dist top-{k}", np.mean(dist), step=self.pipeIter)

            if returnScores:
                return np.asarray(top_scores)

    def getScalingFactor(self):
        '''
        since regression is not normalized, we identify a scaling factor against which we normalize our results
        :return:
        '''
        truncationFactor = 0.1 # cut off x% of the furthest outliers
        dataset = np.load('datasets/' + self.config.dataset.oracle + '.npy', allow_pickle=True).item()

        energies = dataset['energies']
        d1 = [np.sum(np.abs(energies[i] - energies)) for i in range(len(energies))]
        scores = energies[np.argsort(d1)] # sort according to mutual distance
        margin = int(len(scores) * truncationFactor)
        scores = scores[:-margin] # cut 'margin' of furthest points
        self.scalingFactor = np.ptp(scores)


    def addRandomSamples(self, samples, energies, uncertainties, minClusterSamples, minClusterEns, minClusterVars):
        rands = np.random.randint(0, len(samples), size=self.config.querier.model_state_size - len(minClusterSamples))
        randomSamples = samples[rands]
        randomEnergies = energies[rands]
        randomUncertainties = uncertainties[rands]
        minClusterSamples = np.concatenate((minClusterSamples, randomSamples))
        minClusterEns = np.concatenate((minClusterEns, randomEnergies))
        minClusterVars = np.concatenate((minClusterVars, randomUncertainties))
        printRecord('Padded model state with {} random samples from sampler run'.format(len(rands)))

        return minClusterSamples, minClusterEns, minClusterVars


    def getDataDists(self, samples):
        '''
        compute average binary distances between a set of samples and
        1 - itself
        2 - the training dataset
        3 - a large random sample
        :param samples:
        :return:
        '''
        # training dataset
        dataset = np.load('datasets/' + self.config.dataset.oracle + '.npy', allow_pickle=True).item()
        dataset = dataset['samples']

        # large, random sample
        numSamples = min(int(1e3), self.config.dataset.dict_size ** self.config.dataset.max_length // 100) # either 1eX, or 1% of the sample space, whichever is smaller
        randomData = self.oracle.initializeDataset(save=False, returnData=True, customSize=numSamples) # get large random dataset
        randomSamples = randomData['samples']

        internalDist = binaryDistance(samples, self.config.dataset.dict_size, pairwise=False,extractInds=len(samples))
        datasetDist = binaryDistance(np.concatenate((samples, dataset)), self.config.dataset.dict_size, pairwise=False, extractInds = len(samples))
        randomDist = binaryDistance(np.concatenate((samples,randomSamples)), self.config.dataset.dict_size, pairwise=False, extractInds=len(samples))

        return internalDist, datasetDist, randomDist


def trainModel(config, i):
    '''
    rewritten for training in a parallelized fashion
    needs to be outside the class method for multiprocessing to work
    :param i:
    :return:
    '''

    model = modelNet(config, i)
    err_te_hist = model.converge(returnHist = True)  # converge model

    return err_te_hist

