'''import statement'''
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
import sklearn.cluster as cluster


'''
This is a general utilities file for the active learning pipeline

To-Do:
'''

def getParamsDict(args):
    params = {}
    params['run num'] = args.run_num
    params['sampler seed'] = args.sampler_seed % 10  # seed for MCMC modelling (each set of gammas gets a slightly different seed)
    params['model seed'] = args.model_seed % 10  # seed used for model ensemble (each model gets a slightly different seed)
    params['dataset seed'] = args.dataset_seed % 10 # if we are using a toy dataset, it may take a specific seed
    params['query mode'] = args.query_mode  # 'random', 'energy', 'uncertainty', 'heuristic', 'learned' # different modes for query construction
    params['dataset'] = args.dataset

    # initialize control parameters
    params['device'] = args.device  # 'local' or 'cluster'
    params['GPU'] = args.GPU  # WIP - train and evaluate models on GPU
    params['explicit run enumeration'] = args.explicit_run_enumeration  # if this is True, the next run be fresh, in directory 'run%d'%run_num, if false, regular behaviour. Note: only use this on fresh runs
    params['test mode'] = args.test_mode  # WIP # if true, automatically set parameters for a quick test run

    # Pipeline parameters
    params['pipeline iterations'] = args.pipeline_iterations  # number of cycles with the oracle
    params['minima dist cutoff'] = args.minima_dist_cutoff  # minimum distance (normalized, binary) between distinct minima

    params['queries per iter'] = args.queries_per_iter  # maximum number of questions we can ask the oracle per cycle
    params['mode'] = args.mode  # 'training'  'evaluation' 'initialize'
    params['debug'] = args.debug
    params['training parallelism'] = args.training_parallelism  # True hangs on Linux systems # distribute training across a CPU multiprocessing pool (each CPU may still access a GPU, if GPU == True)

    # toy data parameters
    params['dataset type'] = args.dataset_type # oracle is very fast to sample
    params['init dataset length'] = args.init_dataset_length  # number of items in the initial (toy) dataset
    params['dict size'] = args.dict_size  # number of possible choices per-state, e.g., [0,1] would be two, [1,2,3,4] (representing ATGC) would be 4
    params['variable sample length'] = args.variable_sample_length  # if true, 'max sample length' should be a list with the smallest and largest size of input sequences [min, max]. If 'false', model is MLP, if 'true', transformer encoder -> MLP output. - false isn't really working/maintained
    params['min sample length'], params['max sample length'] = [args.min_sample_length, args.max_sample_length]  # minimum input sequence length and # maximum input sequence length (inclusive) - or fixed sample size if 'variable sample length' is false

    # querier settings
    params['model state size'] = args.model_state_size

    # model parameters
    params['model type'] = args.model_type # type of proxy model
    params['model ensemble size'] = args.model_ensemble_size  # number of models in the ensemble
    params['model filters'] = args.model_filters # number of neurons per proxy NN layer
    params['model layers'] = args.model_layers  # number of layers in NN proxy models (transformer encoder layers OR MLP layers)
    params['embed dim'] = args.embedding_dim  # embedding dimension for transformer only
    params['max training epochs'] = args.max_epochs
    params['batch size'] = args.training_batch_size

    # sampler parameters
    params['sampling time'] = args.sampling_time
    params['num samplers'] = args.num_samplers  # minimum number of gammas over which to search for each sampler (if doing in parallel, we may do more if we have more CPUs than this)
    params['min gamma'] = args.min_gamma
    params['max gamma'] = args.max_gamma

    return params


def printRecord(statement):
    '''
    print a string to command line output and a text file
    :param statement:
    :return:
    '''
    print(statement)
    if os.path.exists('record.txt'):
        with open('record.txt', 'a') as file:
            file.write('\n' + statement)
    else:
        with open('record.txt', 'w') as file:
            file.write('\n' + statement)


def letters2numbers(sequences): #Tranforming letters to numbers:
    '''
    Converts ATCG sequences to numerical values
    :param sequences: ATCG-format DNA sequences to be converted
    :return: DNA sequences in 1234 format
    '''


    my_seq=np.zeros((len(sequences), len(sequences[0])))
    row=0

    for seq in sequences:
        assert (type(seq) == str) and (len(seq) == my_seq.shape[1]), 'Function inputs must be a list of equal length strings'
        col=0
        for na in seq:
            if (na=="a") or (na == "A"):
                my_seq[row,col]=1
            elif (na=="u") or (na == "U") or (na=="t") or (na == "T"):
                my_seq[row,col]=2
            elif (na=="c") or (na == "C"):
                my_seq[row,col]=3
            elif (na=="g") or (na == "G"):
                my_seq[row,col]=4
            col+=1
        row+=1

    return my_seq


def numbers2letters(sequences): #Tranforming letters to numbers:
    '''
    Converts numerical values to ATGC-format
    :param sequences: numerical DNA sequences to be converted
    :return: DNA sequences in ATGC format
    '''
    if type(sequences) != np.ndarray:
        sequences = np.asarray(sequences)

    my_seq=["" for x in range(len(sequences))]
    row=0
    for j in range(len(sequences)):
        seq = sequences[j,:]
        assert type(seq) != str, 'Function inputs must be a list of equal length strings'
        for i in range(len(sequences[0])):
            na = seq[i]
            if na==1:
                my_seq[row]+='A'
            elif na==2:
                my_seq[row]+='T'
            elif na==3:
                my_seq[row]+='C'
            elif na==4:
                my_seq[row]+='G'
        row+=1
    return my_seq


def getModelName(ensembleIndex):
    '''
    :param params: parameters of the pipeline we are training
    :return: directory label
    '''
    dirName = "estimator=" + str(ensembleIndex)

    return dirName


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def resultsAnalysis(outDir):
    '''
    analyze the results of a bunch of parallel runs of the active learning pipeline
    '''
    outDicts = []
    os.chdir(outDir)
    for dirs in os.listdir(outDir):
        out = np.load(dirs + '/outputsDict.npy',allow_pickle=True).item()
        outDicts.append(out)

    # collect info for plotting
    numIter = out['params']['pipeline iterations']
    numModels = out['params']['model ensemble size']
    numSampler = out['params']['num samplers']
    optima = []
    testLoss = []
    oracleOptima = []
    for dict in outDicts:
        oracleOptima.append(np.amin(dict['oracle outputs']['energy']))
        optima.append(np.amin(dict['best optima found']))
        testLoss.append(np.amin(dict['model test minima']))

    # average over repeated runs
    oracleOptima = np.asarray(oracleOptima)
    optima = np.asarray(optima)
    testLoss = np.asarray(testLoss)

    avgDiff = []
    avgLoss = []

    for i in range(5): #
        avgDiff.append(np.average(np.abs((oracleOptima[i:-1:5] - optima[i:-1:5])/oracleOptima[i:-1:5])))
        avgLoss.append(np.average(testLoss[i:-1:5]))

    plt.clf()
    plt.plot(avgLoss / np.amax(avgLoss),label='test loss')
    plt.plot(avgDiff / np.amax(avgDiff),label='pipeline error')
    plt.legend()



def binaryDistance(samples, pairwise = False):
    '''
    compute simple sum of distances between sample vectors
    :param samples:
    :return:
    '''
    # determine if all samples have equal length
    lens = np.array([i.shape[-1] for i in samples])
    if len(np.unique(lens)) > 1: # if there are multiple lengths, we need to pad up to a constant length
        raise ValueError('Attempted to compute binary distances between samples with different lengths!')

    if pairwise: # compute every pairwise distances
        distances = np.zeros((len(samples), len(samples)))
        for i in range(len(samples)):
            distances[i, :] = np.sum(samples[i] != samples, axis = 1) / len(samples[i])
    else: # compute average distance of each sample from all the others
        distances = np.zeros(len(samples))
        for i in range(len(samples)):
            distances[i] = np.sum(samples[i] != samples) / len(samples.flatten())

    return distances


def sortTopXSamples(sortedSamples, nSamples = int(1e6), distCutoff = 0.2):
    # collect top distinct samples

    bestSamples = np.expand_dims(sortedSamples[0], 0) # start with the best identified sequence
    bestInds = [0]
    i = -1
    while (len(bestInds) < nSamples) and (i < len(sortedSamples) - 1):
        i += 1
        candidate = np.expand_dims(sortedSamples[i], 0)
        sampleList = np.concatenate((bestSamples, candidate))

        dists = binaryDistance(sampleList, pairwise=True)[-1, :-1]  # pairwise distances between candiate and prior samples
        if all(dists > distCutoff):  # if the samples are all distinct
            bestSamples = np.concatenate((bestSamples, candidate))
            bestInds.append(i)

    return bestInds


def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out


def filterDuplicateSamples(samples, oldDatasetPath = None, returnInds = False):
    """
    assumes original dataset contains no duplicates
    :param samples: must be np array padded to equal length. If a combination of new and original datasets, critical that the original data comes first.
    : param origDatasetLen: if samples is a combination of new and old datasets, set old dataset first with length 'origDatasetLen'
    :return: non-duplicate samples and/or indices of such samples
    """
    origDatasetLen = 0 # if there is no old dataset, take everything
    if oldDatasetPath is not None:
        dataset = np.load(oldDatasetPath, allow_pickle=True).item()['samples']
        origDatasetLen = len(dataset)
        samples = np.concatenate((dataset,samples),axis=0)

    samplesTuple = [tuple(row) for row in samples]
    seen = set()
    seen_add = seen.add

    if returnInds:
        filtered = [[samplesTuple[i], i] for i in range(len(samplesTuple)) if not (samplesTuple[i] in seen or seen_add(samplesTuple[i]))]
        filteredSamples = [filtered[i][0] for i in range(len(filtered))]
        filteredInds = [filtered[i][1] for i in range(len(filtered))]

        return np.asarray(filteredSamples[origDatasetLen:]), np.asarray(filteredInds[origDatasetLen:]) - origDatasetLen
    else:
        filteredSamples = [samplesTuple[i] for i in range(len(samplesTuple)) if not (samplesTuple[i] in seen or seen_add(samplesTuple[i]))]

        return np.asarray(filteredSamples[origDatasetLen:])


def generateRandomSamples(nSamples, sampleLengthRange, dictSize, oldDatasetPath = None, variableLength = True):
    '''
    randomly generate a non-repeating set of samples of the appropriate size and composition
    :param nSamples:
    :param sampleLengthRange:
    :param dictSize:
    :param variableLength:
    :return:
    '''

    if variableLength:
        samples = []
        while len(samples) < nSamples:
            for i in range(sampleLengthRange[0], sampleLengthRange[1] + 1):
                samples.extend(np.random.randint(1, dictSize + 1, size=(int(10 * dictSize * i), i)))

            samples = numpy_fillna(np.asarray(samples)).astype(int)  # pad sequences up to maximum length
            samples = filterDuplicateSamples(samples, oldDatasetPath)  # this will naturally proportionally punish shorter sequences
            if len(samples) < nSamples:
                samples = samples.tolist()

    else:  # fixed sample size
        samples = []
        while len(samples) < nSamples:
            samples.extend(np.random.randint(1, dictSize + 1, size=(2 * nSamples, sampleLengthRange[1])))
            samples = numpy_fillna(np.asarray(samples)).astype(int)  # pad sequences up to maximum length
            samples = filterDuplicateSamples(samples, oldDatasetPath)  # this will naturally proportionally punish shorter sequences
            if len(samples) < nSamples:
                samples = samples.tolist()

    np.random.shuffle(samples)  # shuffle so that sequences with different lengths are randomly distributed
    samples = samples[:nSamples]  # after shuffle, reduce dataset to desired size, with properly weighted samples

    return samples

def runSampling(params, sampler, model, useOracle=False):
    '''
    run sampling and return key outputs in a dictionary
    :param sampler:
    :return:
    '''
    sampleOutputs = sampler.sample(model, useOracle=useOracle)

    samples = []
    scores = []
    energies = []
    uncertainties = []
    for i in range(params['num samplers']):
        samples.extend(sampleOutputs['optimalSamples'][i])
        scores.extend(sampleOutputs['optima'][i])
        energies.extend(sampleOutputs['enAtOptima'][i])
        uncertainties.extend(sampleOutputs['varAtOptima'][i])

    outputs = {
        'samples': np.asarray(samples),
        'scores': np.asarray(scores),
        'energies': np.asarray(energies),
        'uncertainties': np.asarray(uncertainties)
    }

    return outputs

def get_n_params(model):
    '''
    count parameters for a pytorch model
    :param model:
    :return:
    '''
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def doAgglomerativeClustering(samples,energies, uncertainties,cutoff = 0.25):
    '''
    agglomerative clustering and sorting with pairwise binary distance metric
    :param samples:
    :param energies:
    :param cutoff:
    :return:
    '''
    agglomerate = cluster.AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average', compute_full_tree=True, distance_threshold=cutoff).fit(binaryDistance(samples, pairwise=True))
    labels = agglomerate.labels_
    nClusters = agglomerate.n_clusters_
    clusters = []
    totInds = []
    clusterEns = []
    clusterVars = []
    for i in range(len(np.unique(labels))):
        inds = np.where(labels == i)[0].astype(int)
        totInds.extend(inds)
        clusters.append([samples[j] for j in inds])
        clusterEns.append([energies[j] for j in inds])
        clusterVars.append([uncertainties[j] for j in inds])


    return clusters, clusterEns, clusterVars


def clusterAnalysis(clusters, clusterEns, clusterVars):
    '''
    get the average and minimum energies and variances at these points
    :param clusters:
    :param clusterEns:
    :param clusterVars:
    :return:
    '''
    clusterSize = np.asarray([len(cluster) for cluster in clusters])
    avgClusterEns = np.asarray([np.average(cluster) for cluster in clusterEns])
    minClusterEns = np.asarray([np.amin(cluster) for cluster in clusterEns])
    avgClusterVars = np.asarray([np.average(cluster) for cluster in clusterVars])
    minClusterVars = np.asarray([clusterVars[i][np.argmin(clusterEns[i])] for i in range(len(clusterVars))])
    minClusterSamples = np.asarray([clusters[i][np.argmin(clusterEns[i])] for i in range(len(clusterEns))])

    clusterOrder = np.argsort(minClusterEns)
    clusterSize = clusterSize[clusterOrder]
    avgClusterEns = avgClusterEns[clusterOrder]
    minClusterEns = minClusterEns[clusterOrder]
    avgClusterVars = avgClusterVars[clusterOrder]
    minClusterVars = minClusterVars[clusterOrder]
    minClusterSamples = minClusterSamples[clusterOrder]

    return clusterSize, avgClusterEns, minClusterEns, avgClusterVars, minClusterVars, minClusterSamples


# TODO graph outputs from state dict record

class resultsPlotter():
    def __init__(self):
        self.i = 0
        self.j = 0

    def process(self, directory):
        # get simulation results
        os.chdir(directory)
        results = np.load('outputsDict.npy',allow_pickle=True).item()

        self.niters = len(results['state dict record'])
        self.nmodels = results['state dict record'][0]['n proxy models']

        self.trueMin = np.amin(results['oracle outputs']['energies'])
        self.trueMinSample = results['oracle outputs']['samples'][np.argmin(results['oracle outputs']['energies'])]

        self.avgTestLoss = np.asarray([results['state dict record'][i]['test loss'] for i in range(self.niters)])
        self.testStd = np.asarray([results['state dict record'][i]['test disagreement'] for i in range(self.niters)])
        self.allTestLosses = np.asarray([results['state dict record'][i]['all test losses'] for i in range(self.niters)])
        self.stdEns = np.asarray([results['state dict record'][i]['best cluster energies'] for i in range(self.niters)]) # these come standardized out of the box
        self.stdDevs = np.asarray([results['state dict record'][i]['best cluster deviations'] for i in range(self.niters)])
        self.stateSamples = np.asarray([results['state dict record'][i]['best cluster samples'] for i in range(self.niters)])
        self.internalDiffs = np.asarray([results['state dict record'][i]['best clusters internal diff'] for i in range(self.niters)])
        self.datasetDiffs = np.asarray([results['state dict record'][i]['best clusters dataset diff'] for i in range(self.niters)])
        self.randomDiffs = np.asarray([results['state dict record'][i]['best clusters random set diff'] for i in range(self.niters)])
        self.bigDataLoss = np.asarray([results['big dataset loss'][i] for i in range(self.niters)])
        self.bottom10Loss = np.asarray([results['bottom 10% loss'][i] for i in range(self.niters)])

        # get dataset mean and std
        target = os.listdir('datasets')[0]
        dataset = np.load('datasets/' + target,allow_pickle=True).item()
        datasetScores = dataset['scores']
        self.mean = np.mean(datasetScores)
        self.std = np.sqrt(np.var(datasetScores))

        # standardize results
        self.stdTrueMin = (self.trueMin - self.mean) / self.std

        # normalize against true answer
        self.normedEns = 1 - np.abs(self.stdTrueMin - self.stdEns) / np.abs(self.stdTrueMin)
        self.normedDevs = self.stdDevs / np.abs(self.stdTrueMin)

        self.xrange = np.arange(self.niters) + 1


    def plotLosses(self, fignum, color, label):
        plt.figure(fignum)
        plt.semilogy(self.xrange, self.bigDataLoss, color + '.-', label=label + ' big sample loss')
        plt.semilogy(self.xrange, self.bottom10Loss, color + 'o-', label=label + ' bottom 10% loss')
        plt.fill_between(self.xrange, self.avgTestLoss - self.testStd / 2, self.avgTestLoss + self.testStd / 2, alpha = 0.2, edgecolor = color, facecolor = color, label = label + ' test losses')
        plt.xlabel('AL Iterations')
        plt.ylabel('Smooth L1 Loss')
        plt.legend()

    def plotPerformance(self, fignum, color, label, ind):
        plt.figure(fignum)
        plt.fill_between(self.xrange, self.normedEns[:,0] - self.normedDevs[:,0] / 2, self.normedEns[:,0] + self.normedDevs[:,0] / 2, alpha = 0.2, edgecolor = color, facecolor = color, label = label + ' best optimum')
        avgens = np.average(self.normedEns, axis=1)
        plt.errorbar(self.xrange + ind / 10, avgens, yerr = [avgens-self.normedEns[:,0], avgens-self.normedEns[:,1]], fmt = color + '.', ecolor=color, elinewidth=3, capsize=1.5, alpha=0.2, label=label + ' state range')
        #for i in range(self.normedEns.shape[1]):
        #    plt.plot(self.xrange + self.i / 10, self.normedEns[:,i], color + '.')
        plt.xlabel('AL Iterations')
        plt.ylabel('Performance')
        plt.legend()
