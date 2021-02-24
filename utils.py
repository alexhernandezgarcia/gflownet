'''import statement'''
import numpy as np
import matplotlib.pyplot as plt


'''
This is a general utilities file for the active learning pipeline

To-Do:
'''


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


def getDirName(params):
    '''
    :param params: parameters of the pipeline we are training
    :return: directory label
    '''
    dirName = "dataset="+params['dataset']+"_filters=%d_layers=%d_seed=%d" %\
                   (params['model filters'], params['model layers'],params['random seed'])

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

    nDuplicates = 0
    for i in range(len(oracleSequences)):
        assert type(oracleSequences[i]) == str, "Sequences must be in string format before saving to the dataset"
        assert len(oracleSequences[i]) == len(dataset['sequences'][0]), "Added sequences must be the same length as those already in the dataset!"
        duplicate = 0
        for j in range(len(dataset['sequences'])): # search for duplicates
            if oracleSequences[i] == dataset['sequences'][j]:
                duplicate = 1
                nDuplicates += 1

        if duplicate == 0:
            dataset['sequences'].append(oracleSequences[i])
            dataset['scores'] = np.append(dataset['scores'], oracleScores[i])

    if nDuplicates > 0:
        print("%d duplicates found" % nDuplicates)


    print(f"Added{bcolors.OKBLUE}{bcolors.BOLD} %d{bcolors.ENDC}" %int(len(oracleSequences) - nDuplicates) + " to the dataset")
    print("=====================================================================")
    print("=====================================================================")
    print("=====================================================================")
    #print("New dataset size =%d" %len(dataset['sequences']))
    np.save('datasets/'+params['dataset'], dataset)

    if params['debug'] == 1:
        plt.figure(5)
        columns = min(5,params['pipeline iterations'])

        rows = max([1,(params['pipeline iterations'] // 5)])
        plt.subplot(rows, columns, params['iteration'])
        plt.hist(dataset['scores'],bins=100,density=True)
        plt.title('Iteration #%d' % params['iteration'])
        plt.xlabel('Dataset Scores')


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