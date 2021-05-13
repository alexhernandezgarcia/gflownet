'''import statement'''
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


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
    cmd_line_input = parser.parse_args()
    run = cmd_line_input.run_num

    return run

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
        print("%d duplicates found" % nDuplicates)


    print(f"Added{bcolors.OKBLUE}{bcolors.BOLD} %d{bcolors.ENDC}" %int(len(oracleSequences) - nDuplicates) + " to the dataset")
    print("=====================================================================")
    #print("New dataset size =%d" %len(dataset['samples']))
    np.save('datasets/' + params['dataset'], dataset)

    '''
    if params['debug'] == True:
        plt.figure(5)
        columns = min(5,params['pipeline iterations'])

        rows = max([1,(params['pipeline iterations'] // 5)])
        plt.subplot(rows, columns, params['iteration'])
        plt.hist(dataset['scores'],bins=100,density=True)
        plt.title('Iteration #%d' % params['iteration'])
        plt.xlabel('Dataset Scores')
        
    '''


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
    numSampler = out['params']['sampler runs']
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