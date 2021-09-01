from utils import *

'''# one-off
directories = ['C:/Users\mikem\Desktop/activeLearningRuns/run1330']
colors = ['m']
labels = ['test']
averaging = False
'''
'''
# sub 101-103 and 201-203 for batch size on Nupack with small and large models
directories= []
directories.append([
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run10',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run11',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run12',
]
)

directories.append([
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run20',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run21',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run22',
]
)

directories.append([
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run30',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run31',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run32',
]
)

directories.append([
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run110',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run111',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run112',
]
)

directories.append([
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run120',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run121',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run122',
]
)

directories.append([
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run130',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run131',
    'C:/Users\mikem\Desktop/activeLearningRuns/cluster/run132',
]
)

colors = ['m','r','y','c','b','m']
labels = ['small, small', 'small, med', 'small, large', 'large, small', 'large, med','large, large'] # model size, iteration batch size
averaging = True

'''
 # sub 301 and 302 tests for shuffling vs no shuffling
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
    ])

colors = ['m','c']
labels = ['shuffle','no shuffle']
averaging = True





for i in range(len(directories)):
    directory = directories[i]
    color = colors[i]
    label = labels[i]
    nsubplots = len(directories)

    plotter = resultsPlotter()
    if averaging:
        plotter.averageResults(directory)
    else:
        plotter.process(directory)

    plotter.plotLosses(fignum = 1, color = color, label = label)
    plotter.plotPerformance(fignum = 2, color = color, label = label, ind = i)
    plotter.plotDiversity(fignum = 3, nsubplots = nsubplots, subplot = i + 1, color = color, label = label)
    plotter.plotDiversityProduct(fignum = 4, color = color, label = label)
    plotter.plotDiversityMesh(fignum = 5, color = color, nsubplots = nsubplots, subplot = i + 1, label = label)

