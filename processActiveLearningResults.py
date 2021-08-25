import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats
from utils import binaryDistance

'''
to do
==> plot multiple runs together
'''


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def numbers2letters(sequences):  # Tranforming letters to numbers (1234 --> ATGC)
    '''
    Converts numerical values to ATGC-format
    :param sequences: numerical DNA sequences to be converted
    :return: DNA sequences in ATGC format
    '''
    if type(sequences) != np.ndarray:
        sequences = np.asarray(sequences)

    my_seq = ["" for x in range(len(sequences))]
    row = 0
    for j in range(len(sequences)):
        seq = sequences[j, :]
        assert type(seq) != str, 'Function inputs must be a list of equal length strings'
        for i in range(len(sequences[0])):
            na = seq[i]
            if na == 1:
                my_seq[row] += 'A'
            elif na == 2:
                my_seq[row] += 'T'
            elif na == 3:
                my_seq[row] += 'C'
            elif na == 4:
                my_seq[row] += 'G'
        row += 1
    return my_seq


def plotResults(directory, color, labelname, ind, clear = False, singleRun=False):
    '''

    :param directory:
    :param color:
    :param labelname:
    :param clear: erase current figure
    :param singleRun: analyzing a single run rather than an ensemble
    :return:
    '''
    os.chdir(directory)

    # load the outputs
    outputs = []
    if singleRun:
        dirs = [0]
        outputs.append(np.load('outputsDict.npy',allow_pickle=True).item())
    else:
        dirs = os.listdir()
        for dir in dirs:
            outputs.append(np.load(dir + '/outputsDict.npy', allow_pickle=True).item())

    nruns = len(dirs)
    nmodels = outputs[0]['params']['model ensemble size']
    niters = outputs[0]['params']['pipeline iterations']

    # collate results
    minima = []
    totalMinima = []
    bestTenMinima = []
    for i in range(len(outputs)):
        minima.append(outputs[i]['model test minima'])
        totalMinima.append(outputs[i]['big dataset loss'])
        bestTenMinima.append(outputs[i]['bottom 10% loss'])

    # analysis
    # average test loss per iteration, per seed
    minima = np.asarray(minima)  # first axis is seed, second axis is iterations, third axis is models
    niters = minima.shape[1]
    minima = np.average(minima, axis=0)
    minima2 = minima.reshape(niters, nmodels)  # combine all the models
    avgMinima = np.average(minima2, axis=1)

    minimaMins = np.asarray([np.amin(minima[i,:]) for i in range(len(minima))])
    minimaMaxs = np.asarray([np.amax(minima[i,:]) for i in range(len(minima))])

    totalMinima = np.asarray(totalMinima)  # first axis is seed, second axis is iterations
    totalMinima = totalMinima.transpose(1, 0)
    totalMinima2 = totalMinima.reshape(niters, nruns)  # combine all the models
    avgTotalMinima = np.average(totalMinima2, axis=1)

    bestTenMinima = np.asarray(bestTenMinima)  # first axis is seed, second axis is iterations
    bestTenMinima = bestTenMinima.transpose(1, 0)
    bestTenMinima2 = bestTenMinima.reshape(niters, nruns)  # combine all the models
    avgBestTenMinima = np.average(bestTenMinima2, axis=1)



    # plot average test loss over all runs, with confidence intervals
    if clear:
        plt.clf()
    plt.subplot(1,2,1)
    #plt.errorbar(np.arange(niters) + 1, np.log10(avgMinima), yerr=np.log10(minimaCI), fmt=color + 'o-', ecolor=color, elinewidth=0.5, capsize=4, label=labelname + ' average of test minima')
    plt.semilogy(np.arange(niters) + 1, avgTotalMinima, color + 'd-', label=labelname + ' average of large sample')
    plt.semilogy(np.arange(niters) + 1, avgBestTenMinima, color + '.-', label=labelname+ ' average of best 10% of large sample')
    plt.fill_between(np.arange(niters) + 1, minimaMins, minimaMaxs, alpha=0.2, edgecolor = color, facecolor=color, label=labelname + ' per-model test losses')  # average of best samples
    plt.xlabel('AL Iterations')
    plt.ylabel('Test Losses')
    plt.title('Average of Best Test Losses Over {} Ensembles of {} Models'.format(nruns, nmodels))
    plt.legend()

    #for i in range(nmodels):
    #    plt.plot(np.arange(niters) + 1, np.log10(minima[:,i]), color + 'o', alpha = 0.1)

    '''
    plt.subplot(2, 1, 2)
    
    plt.semilogy(np.arange(niters) + 1, avgMinima, color + 'd-', label=labelname + ' average of test minima')
    plt.semilogy(np.arange(niters) + 1, avgTotalMinima, color + '.-', label=labelname + ' average of large sample')
    plt.semilogy(np.arange(niters) + 1, avgBestTenMinima, color + 'o-', label=labelname + ' average of best 10% of large sample')
    plt.xlabel('AL Iterations')
    plt.ylabel('Test Losses')
    plt.legend()
    '''


    # average normalized minimum energy and related std deviations
    bestEns = []
    bestSamples = []
    bestVars = []
    oracleMins = []
    for i in range(len(outputs)):
        bestSamples.append(outputs[i]['best samples'])
        bestEns.append(outputs[i]['best energies'])
        bestVars.append(outputs[i]['best vars'])
        oracleMins.append(np.amin(outputs[i]['oracle outputs']['energies']))

    bestSamples = np.asarray(bestSamples)
    bestEns = np.asarray(bestEns)
    bestVars = np.asarray(bestVars)
    oracleMins = np.asarray(oracleMins)

    normedEns = np.zeros_like(bestEns)
    normedVars = np.zeros_like(normedEns)

    if np.amin(oracleMins) >= 0:
        enMaxs = [np.amax(bestEn) for bestEn in bestEns[0]]
        enMax = np.amax(enMaxs)
        oracleMins = oracleMins - enMax
        bestEns = bestEns - enMax

    for i in range(len(outputs)):
        for j in range(len(bestEns[i])):
            normedEns[i][j] = bestEns[i][j] / np.amin(oracleMins) # assume all runs with same oracle
            normedVars[i][j] = np.sqrt(bestVars[i][j]) / np.abs(np.average(bestEns[i][j]))

    minNormedEns = np.asarray([[np.amax(normedEns[i][j]) for j in range(niters)] for i in range(nruns)])
    minNormedVars = np.asarray([[normedVars[i][j][np.argmax(normedEns[i][j])] for j in range(niters)] for i in range(nruns)])
    if minNormedVars.ndim > 2:
        minNormedVars = minNormedVars[:,:,0]


    nOptima = [len(normedEns[0][i]) for i in range(niters)]
    # TODO rebuild this for positive energies
    plt.subplot(1,2,2)
    #plt.errorbar(np.arange(niters) + 1, minNormedEns[0], yerr = minNormedVars[0], fmt = color + 'o-', ecolor=color, elinewidth=0.5, capsize=8, label=labelname + ' best score')  # average of best samples
    plt.plot(np.arange(niters) + 1, minNormedEns[0], color + 'o-', label=labelname + ' best score')  # average of best samples
    plt.fill_between(np.arange(niters) + 1, minNormedEns[0] - minNormedVars[0]/2, minNormedEns[0] + minNormedVars[0]/2, alpha=0.2, edgecolor = color, facecolor=color, label=labelname + ' best score')  # average of best samples

    for i in range(niters):
        plt.plot(np.ones(len(normedEns[0][i])) * (i + 1) + ind /10, normedEns[0][i], color + '.')#, label=labelname + 'distinct minima')
    plt.twinx()
    plt.plot(np.arange(niters) + 1, nOptima, color + 'd-',label=labelname + ' # distinct optima')
    plt.legend()
    plt.title('Ensemble Average Performance over {} runs'.format(nruns))
    plt.ylabel('Score vs. known minimum')
    plt.xlabel('AL Iterations')


plt.figure(1)
directory = 'C:/Users\mikem\Desktop/activeLearningRuns/run1251'
plotResults(directory ,'k','wide',0, clear=True,singleRun=True)


'''
directory = 'C:/Users\mikem\Desktop/activeLearningRuns/run1215'
plotResults(directory ,'r','transformer, energy', 1, singleRun=True)
directory = 'C:/Users\mikem\Desktop/activeLearningRuns/run1216'
plotResults(directory ,'b','transformer, uncertainty', 2, singleRun=True)
'''

# PLOT NUMBER (MAGNITUDE) OF DISTINCT MINIMA OVER TIME

'''
directory = 'C:/Users\mikem\Desktop/activeLearningRuns/run1222'
plotResults(directory ,'y','MLP, random',4, clear=False,singleRun=True)
directory = 'C:/Users\mikem\Desktop/activeLearningRuns/run1223'
plotResults(directory ,'m','MLP, energy', 5, singleRun=True)
directory = 'C:/Users\mikem\Desktop/activeLearningRuns/run1224'
plotResults(directory ,'c','MLP, uncertainty', 6, singleRun=True)
'''

'''
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run30x'
plotResults(directory ,'k','default, random')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run31x'
plotResults(directory ,'r','default, energy')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run32x'
plotResults(directory ,'b','default, uncertainty')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run33x'
plotResults(directory ,'g','2 layers')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run34x'
plotResults(directory ,'m','wide layers')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run35x'
plotResults(directory ,'y','6 layers')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run36x'
plotResults(directory ,'c','5 samplers')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run37x'
plotResults(directory ,'c','2 models')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run38x'
plotResults(directory ,'c','1 layer')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run39x'
plotResults(directory ,'c','10 layers')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run50x'
plotResults(directory ,'c','20 samplers')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run51x'
plotResults(directory ,'c','40 samplers')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run54x'
plotResults(directory ,'c','20 models')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run53x'
plotResults(directory ,'c','20 models, 20 samplers')



directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run40x'
plotResults(directory ,'k','Random')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run41x'
plotResults(directory ,'r','Energy')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run42x'
plotResults(directory ,'b','Uncertainty')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run43x'
plotResults(directory ,'g','2 layers')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run44x'
plotResults(directory ,'m','wide layers')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run45x'
plotResults(directory ,'y','6 layers')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run46x'
plotResults(directory ,'c','5 samplers')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run47x'
plotResults(directory ,'c','2 models')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run48x'
plotResults(directory ,'c','1 layer')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run49x'
plotResults(directory ,'c','10 layers')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run60x'
plotResults(directory ,'c','20 samplers')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run61x'
plotResults(directory ,'c','40 samplers')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run62x'
plotResults(directory ,'c','20 models')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run63x'
plotResults(directory ,'c','20 models, 20 samplers')
'''