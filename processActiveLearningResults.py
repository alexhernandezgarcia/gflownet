import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

# find the stuff
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/run16x'
os.chdir(directory)

# load the outputs
outputs = []
dirs = os.listdir()
for dir in dirs:
    outputs.append(np.load(dir + '/outputsDict.npy',allow_pickle=True).item())

# collate results
minima = []
for i in range(len(outputs)):
    minima.append(outputs[i]['model test minima'])


# analysis
# average test loss per iteration, per seed
minima = np.asarray(minima) # first axis is seed, second axis is interations, third axis is models
avgMinima1 = np.average(minima,axis=2) # average over models
avgMinima2 = np.average(avgMinima1,axis=0) # average over toy datasets
varMinima2 = np.var(avgMinima1,axis=0)

CI = []
for i in range(len(avgMinima2)):
    mean, Cm, Cp = mean_confidence_interval(avgMinima1[:,i],confidence = 0.95)
    CI.append([Cm,Cp])

CI = np.transpose(np.asarray(CI))

# plot average test loss over all runs, with confidence intervals
plt.figure(1)
plt.clf()
plt.errorbar(np.arange(20)+1, avgMinima2, yerr=CI, fmt='k.-',ecolor='r',elinewidth=0.5,capsize=4)
plt.xlabel('AL Iterations')
plt.ylabel('Test Losses')
plt.title('Average of Best Test Losses Over {} Ensembles of {} Models'.format(minima.shape[0],minima.shape[-1]))


# average normalized minimum energy and related std deviations
bestEns = []
bestSamples = []
bestVars = []
oracleMins = []
for i in range(len(outputs)):
    bestSamples.append(outputs[i]['best samples'])
    bestEns.append(outputs[i]['best energies'])
    bestVars.append(outputs[i]['best vars'])
    oracleMins.append(np.amin(outputs[i]['oracle outputs']['energy']))

bestSamples = np.asarray(bestSamples)
bestEns = np.asarray(bestEns)
bestVars = np.asarray(bestVars)
oracleMins = np.asarray(oracleMins)

normedEns = np.zeros_like(bestEns)
normedVars = np.zeros_like(normedEns)
for i in range(len(outputs)):
    normedEns[i] = bestEns[i] / oracleMins[i]
    normedVars[i] = np.sqrt(bestVars[i]) / np.abs(np.average(bestEns[i]))


plt.figure(2)
plt.clf()
#plt.subplot(1,2,1)
plt.plot(np.average(normedEns[:,:,0],axis=0),'.-') # average of best samples

plt.title('Ensemble Average Performance')
plt.ylabel('Score vs. known minimum')
plt.xlabel('AL Iterations')
#plt.subplot(1,2,2)
#plt.plot(np.average(bestVars[:,:,0],axis=0))
