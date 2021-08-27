from utils import *

''' # sub 301 and 302 tests for shuffling vs no shuffling
directories = []
directories.append([
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run210',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run211',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run212',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run213',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run214',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run215',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run216',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run217',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run218',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run219'
])

directories.append([
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run220',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run221',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run222',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run223',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run224',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run225',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run226',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run227',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run228',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run229'

colors = ['m','c']
labels = ['shuffle','no shuffle']

])
'''



for i in range(len(directories)):
    directory = directories[i]
    color = colors[i]
    label = labels[i]
    nsubplots = len(directories)

    plotter = resultsPlotter()
    plotter.averageResults(directory)
    plotter.plotLosses(fignum = 1, color = color, label = label)
    plotter.plotPerformance(fignum = 2, color = color, label = label, ind = i)
    plotter.plotDiversity(fignum = 3, nsubplots = nsubplots, subplot = i + 1, color = color, label = label)
    plotter.plotDiversityProduct(fignum = 4, color = color, label = label)
    plotter.plotDiversityMesh(fignum = 5, color = color, nsubplots = nsubplots, subplot = i + 1, label = label)

