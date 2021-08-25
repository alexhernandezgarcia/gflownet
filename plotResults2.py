from utils import *

directory = 'C:/Users\mikem\Desktop/activeLearningRuns/run1264'
plotter = resultsPlotter()
plotter.process(directory)
plotter.plotLosses(1, 'r', 'random')
plotter.plotPerformance(2, 'r', 'random', 0)

directory = 'C:/Users\mikem\Desktop/activeLearningRuns/run1265'
plotter = resultsPlotter()
plotter.process(directory)
plotter.plotLosses(1, 'k', 'energy')
plotter.plotPerformance(2, 'k', 'energy', 1)

directory = 'C:/Users\mikem\Desktop/activeLearningRuns/run1266'
plotter = resultsPlotter()
plotter.process(directory)
plotter.plotLosses(1, 'c', 'uncertainty')
plotter.plotPerformance(2, 'c', 'uncertainty', 2)