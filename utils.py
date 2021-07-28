'''import statement'''
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time


'''
This is a general utilities file for the active learning pipeline

To-Do:
'''

def get_input():
    '''
    get the command line in put for the run-num. defaulting to a new run (0)
    :return:
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_num', type=int, default = 0)
    parser.add_argument('--sampler_seed', type=int, default = 0)
    parser.add_argument('--model_seed', type=int, default = 0)
    parser.add_argument('--dataset_seed', type=int, default=0)
    parser.add_argument('--query_type', type=str, default='random')
    cmd_line_input = parser.parse_args()
    run = cmd_line_input.run_num
    samplerSeed = cmd_line_input.sampler_seed
    modelSeed = cmd_line_input.model_seed % 10
    datasetSeed = cmd_line_input.dataset_seed
    queryMode = cmd_line_input.query_type

    return [run, samplerSeed, modelSeed, datasetSeed, queryMode]

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

    assert type(sequences) == list, 'Function inputs must be a list'

    my_seq=np.zeros((len(sequences), len(sequences[0])))
    row=0

    for seq in sequences:
        assert (type(seq) == str) and (len(seq) == my_seq.shape[1]), 'Function inputs must be a list of equal length strings'
        col=0
        for na in seq:
            if (na=="a") or (na == "A"):
                my_seq[row,col]=0
            elif (na=="u") or (na == "U") or (na=="t") or (na == "T"):
                my_seq[row,col]=1
            elif (na=="c") or (na == "C"):
                my_seq[row,col]=2
            elif (na=="g") or (na == "G"):
                my_seq[row,col]=3
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
            if na==0:
                my_seq[row]+='A'
            elif na==1:
                my_seq[row]+='T'
            elif na==2:
                my_seq[row]+='C'
            elif na==3:
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


def updateDataset(params, oracleSequences, oracleScores):
    '''
    loads dataset, appends new datapoints from oracle, and saves dataset
    :param params: model parameters
    :param oracleSequences: sequences which were sent to oracle
    :param oracleScores: scores of sequences sent to oracle
    :return: n/a
    '''
    dataset = np.load('datasets/' + params['dataset'] + '.npy', allow_pickle=True).item()

    nDuplicates = 0 # this check should no longer be necessary, but I guess it doesn't hurt anything for now
    for i in range(len(oracleSequences)):
        #assert type(oracleSequences[i]) == str, "Sequences must be in string format before saving to the dataset"
        assert len(oracleSequences[i]) == len(dataset['samples'][0]), "Added sequences must be the same length as those already in the dataset!"
        duplicate = 0
        for j in range(len(dataset['samples'])): # search for duplicates
            if all(oracleSequences[i] == dataset['samples'][j]):
                duplicate = 1
                nDuplicates += 1

        if duplicate == 0:
            dataset['samples'] = np.concatenate((dataset['samples'],np.expand_dims(oracleSequences[i],0)))
            dataset['scores'] = np.concatenate((dataset['scores'],np.expand_dims(oracleScores[i],0)))
            #dataset['samples'].append(oracleSequences[i])
            #dataset['scores'] = np.append(dataset['scores'], oracleScores[i])

    if nDuplicates > 0:
        printRecord("%d duplicates found" % nDuplicates)


    printRecord(f"Added{bcolors.OKBLUE}{bcolors.BOLD} %d{bcolors.ENDC}" %int(len(oracleSequences) - nDuplicates) + " to the dataset")
    printRecord("=====================================================================")
    #printRecord("New dataset size =%d" %len(dataset['samples']))
    np.save('datasets/' + params['dataset'], dataset)



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
    numModels = out['params']['ensemble size']
    numSampler = out['params']['sampler gammas']
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
    if pairwise: # compute every pairwise distances
        distances = np.zeros((len(samples), len(samples)))
        for i in range(len(samples)):
            distances[i, :] = np.sum(samples[i] != samples, axis = 1) / len(samples[i])
    else: # compute average distance of each sample from all the others
        distances = np.zeros(len(samples))
        for i in range(len(samples)):
            distances[i] = np.sum(samples[i] != samples)

    return distances


def sortTopXSamples(sortedSamples, samples = 10, distCutoff = 0.2):
    # collect top distinct samples

    bestSamples = np.expand_dims(sortedSamples[0], 0)
    bestInds = [0]

    for i in range(1, len(sortedSamples)):
        candidate = np.expand_dims(sortedSamples[i], 0)
        dists = binaryDistance(np.concatenate((bestSamples, candidate)), pairwise=True)[-1, :-1]  # pairwise distances between candiate and prior samples
        if all(dists > distCutoff):  # if the samples are all distinct
            bestSamples = np.concatenate((bestSamples, candidate))
            bestInds.append(i)

    return bestInds[:samples]
