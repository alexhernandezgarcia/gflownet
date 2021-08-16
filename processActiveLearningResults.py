import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats

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


def plotResults(directory, color, labelname,clear = False):
    os.chdir(directory)

    # load the outputs
    outputs = []
    dirs = os.listdir()
    for dir in dirs:
        outputs.append(np.load(dir + '/outputsDict.npy', allow_pickle=True).item())

    nruns = len(dirs)
    nmodels = outputs[0]['params']['ensemble size']
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
    minima = np.asarray(minima)  # first axis is seed, second axis is interations, third axis is models
    minima = minima.reshape(nmodels, niters, nruns)  # test minima
    minima = minima.transpose(1, 0, 2)
    minima = np.average(minima, axis=2)
    minima2 = minima.reshape(niters, nmodels)  # combine all the models
    avgMinima = np.average(minima2, axis=1)

    totalMinima = np.asarray(totalMinima)  # first axis is seed, second axis is interations, third axis is models
    totalMinima = totalMinima.transpose(1, 0)
    totalMinima2 = totalMinima.reshape(niters, nruns)  # combine all the models
    avgTotalMinima = np.average(totalMinima2, axis=1)

    bestTenMinima = np.asarray(bestTenMinima)  # first axis is seed, second axis is interations, third axis is models
    bestTenMinima = bestTenMinima.transpose(1, 0)
    bestTenMinima2 = bestTenMinima.reshape(niters, nruns)  # combine all the models
    avgBestTenMinima = np.average(bestTenMinima2, axis=1)

    minimaCI = []
    totalMinimaCI = []
    bestTenMinimaCI = []
    for i in range(niters):
        mean, Cm, Cp = mean_confidence_interval(minima2[i, :], confidence=0.95)
        minimaCI.append([Cm, Cp])
        mean, Cm, Cp = mean_confidence_interval(totalMinima2[i, :], confidence=0.95)
        totalMinimaCI.append([Cm, Cp])
        mean, Cm, Cp = mean_confidence_interval(bestTenMinima2[i, :], confidence=0.95)
        bestTenMinimaCI.append([Cm, Cp])

    minimaCI = np.transpose(np.asarray(minimaCI))
    totalMinimaCI = np.transpose(np.asarray(totalMinimaCI))
    bestTenMinimaCI = np.transpose(np.asarray(bestTenMinimaCI))

    # plot average test loss over all runs, with confidence intervals
    plt.subplot(1,2,1)
    #plt.subplot(2, 1, 1)
    #plt.errorbar(np.arange(niters) + 1, np.log10(avgMinima), yerr=np.log10(minimaCI), fmt=color + 'o-', ecolor=color, elinewidth=0.5, capsize=4, label=labelname + ' average of test minima')
    plt.errorbar(np.arange(niters) + 1, np.log10(avgTotalMinima), yerr=np.log10(totalMinimaCI), fmt=color + 'd-', ecolor=color, elinewidth=0.5, capsize=4, label=labelname + ' average of large sample')
    plt.errorbar(np.arange(niters) + 1, np.log10(avgBestTenMinima), yerr=np.log10(bestTenMinimaCI), fmt=color + '.-', ecolor=color, elinewidth=0.5, capsize=4, label=labelname+ ' average of best 10% of large sample')
    plt.xlabel('AL Iterations')
    plt.ylabel('Test Losses')
    plt.title('Average of Best Test Losses Over {} Ensembles of {} Models'.format(nruns, nmodels))
    plt.legend()
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
    for i in range(len(outputs)):
        normedEns[i] = bestEns[i] / np.amin(oracleMins) # assume all runs with same oracle
        normedVars[i] = np.sqrt(bestVars[i]) / np.abs(np.average(bestEns[i]))

    minNormedEns = normedEns[:, :, 0]
    minNormedVars = normedVars[:, :, 0]

    minEnCI = []
    for i in range(niters):
        mean, Cm, Cp = mean_confidence_interval(minNormedEns[:, i], confidence=0.95)
        minEnCI.append([Cm, Cp])

    minEnCI = np.transpose(np.asarray(minEnCI))

    bestSampleVar = np.zeros(niters)
    varinds = np.argmax(minNormedEns, axis=0)
    for i in range(niters):
        bestSampleVar = minNormedVars[varinds[i]]

    plt.subplot(1,2,2)
    #if clear:
        #plt.clf()
    plt.errorbar(np.arange(niters) + 1, np.average(normedEns[:, :, 0], axis=0), yerr=np.average(minNormedVars, axis=0), fmt=color + '.-', ecolor=color, elinewidth=0.5, capsize=4, label=labelname + ' average of top scores')
    plt.errorbar(np.arange(niters) + 1, np.amax(minNormedEns, axis=0), yerr = bestSampleVar, fmt = color + 'o-', ecolor=color, elinewidth=0.5, capsize=4, label=labelname + ' best score')  # average of best samples
    plt.legend()
    plt.title('Ensemble Average Performance over {} runs'.format(nruns))
    plt.ylabel('Score vs. known minimum')
    plt.xlabel('AL Iterations')


plt.figure(1)
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run30x'
plotResults(directory ,'k','Random',clear=True)
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run31x'
plotResults(directory ,'r','Energy')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run32x'
plotResults(directory ,'b','Uncertainty')

plt.figure(2)
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run40x'
plotResults(directory ,'k','Random',clear=True)
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run41x'
plotResults(directory ,'r','Energy')
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run42x'
plotResults(directory ,'b','Uncertainty')
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

'''