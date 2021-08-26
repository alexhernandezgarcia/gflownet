from utils import *

directories = []
directories.append(['C:/Users\mikem\Desktop/activeLearningRuns/cluster/run10',
              'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run11',
              'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run12'])
directories.append(['C:/Users\mikem\Desktop/activeLearningRuns/cluster/run20',
              'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run21',
              'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run22'])
directories.append(['C:/Users\mikem\Desktop/activeLearningRuns/cluster/run30',
              'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run31',
              'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run32'])

colors = ['c','k','r']
labels = ['batch = 10','batch = 50','batch = 100']

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

