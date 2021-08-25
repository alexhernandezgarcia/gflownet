from utils import *

directory = 'C:/Users\mikem\Desktop/activeLearningRuns/run1264'
plotter = resultsPlotter()
plotter.process(directory)
plotter.plotLosses(fignum = 1, color = 'r', label = 'random')
plotter.plotPerformance(fignum = 2, color = 'r', label = 'random', ind = 0)
plotter.plotDiversity(fignum = 3, nsubplots = 3, subplot = 1, color = 'r', label = 'random')
plotter.plotDiversityProduct(fignum = 4, color = 'r', label = 'random')
plotter.plotDiversityMesh(fignum = 5, color = 'r', nsubplots = 3, subplot = 1, label = 'random')


directory = 'C:/Users\mikem\Desktop/activeLearningRuns/run1265'
plotter = resultsPlotter()
plotter.process(directory)
plotter.plotLosses(fignum = 1, color = 'k', label = 'energy')
plotter.plotPerformance(fignum = 2, color = 'k', label = 'energy', ind = 1)
plotter.plotDiversity(fignum = 3, nsubplots = 3, subplot = 2,color =  'k', label = 'energy')
plotter.plotDiversityProduct(fignum = 4, color = 'k', label = 'energy')
plotter.plotDiversityMesh(fignum = 5, color = 'k', nsubplots = 3, subplot = 2, label = 'energy')


directory = 'C:/Users\mikem\Desktop/activeLearningRuns/run1266'
plotter = resultsPlotter()
plotter.process(directory)
plotter.plotLosses(fignum = 1,color = 'c', label ='uncertainty')
plotter.plotPerformance(fignum = 2,color = 'c',label = 'uncertainty', ind = 2)
plotter.plotDiversity(fignum = 3, nsubplots = 3, subplot = 3,color =  'c',label = 'uncertainty')
plotter.plotDiversityProduct(fignum = 4, color = 'c', label = 'uncertainty')
plotter.plotDiversityMesh(fignum = 5, color = 'c', nsubplots = 3, subplot = 3, label = 'uncertainty')
