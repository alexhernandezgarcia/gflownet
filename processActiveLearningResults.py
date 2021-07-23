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
directory = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster/random_toy_test1'
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
plt.errorbar(np.arange(20)+1, avgMinima2, yerr=CI, fmt='k.-',ecolor='c',elinewidth=0.5,capsize=4)
plt.xlabel('AL Iterations')
plt.ylabel('Test Losses')
plt.title('Average of Best Test Losses Over Ensemble of {} Models and Toy Functions {}'.format(minima.shape[-1],minima.shape[0]))


