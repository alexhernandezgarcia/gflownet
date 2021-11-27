from utils import *


def getScores(experiment, experiments_dir):
    os.chdir(experiments_dir + experiment)
    runs = os.listdir()
    run_outputs = []
    for run in runs:
        run_outputs.append(np.load(run + '/episode0/outputsDict.npy',allow_pickle=True).item())

    scores = np.asarray([run_outputs[i]['score record'] for i in range(len(runs))])
    rewards = np.asarray([run_outputs[i]['rewards'] for i in range(len(runs))])
    cumulative_score = np.asarray([run_outputs[i]['cumulative score'] for i in range(len(runs))])
    dists = np.asarray([[[run_outputs[i]['state dict record'][j]['best clusters internal diff'],run_outputs[i]['state dict record'][j]['best clusters dataset diff'],run_outputs[i]['state dict record'][j]['best clusters random set diff']] for i in range(len(runs))] for j in range(20)])
    return scores, rewards, cumulative_score, dists

scoreList = []
rewardsList = []
cumScoreList = []
distList = []

#experiments = ['A1', 'A2', 'A3', 'A4', 'A5', 'A7']
experiments = ['B1', 'B2', 'B3','C1','C2','C3']
#experiments = ['D3','D5']
#experiments = ['ubuntu_test']
plotAll = True

for experiment in experiments:
    experiments_dir = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster\production/'
    #experiments_dir = 'C:/Users/mikem/Desktop/activeLearningRuns/'

    scores, rewards, cumulative_score, dists = getScores(experiment, experiments_dir)
    scoreList.append(scores)
    rewardsList.append(rewards)
    cumScoreList.append(cumulative_score)
    distList.append(dists)
    if plotAll:
        plt.figure()
        plt.clf()
        plt.subplot(1,3,1)
        plt.plot(np.arange(1,21),scores.transpose(),'o-')
        plt.title(experiment)
        plt.ylabel('Score',size=24)
        plt.xlabel('AL Iteration',size=24)
        plt.subplot(1,3,2)
        plt.plot(np.arange(1,21),rewards.transpose(),'o-')
        plt.ylabel('Per-Iteration Reward',size=24)
        plt.xlabel('AL Iteration',size=24)
        plt.subplot(1,3,3)
        plt.plot(np.arange(1,21),np.average(dists[:,:,0,:],axis=-1),'o-')
        plt.plot(np.arange(1,21),np.average(dists[:,:,1,:],axis=-1),'o-')
        plt.plot(np.arange(1,21),np.average(dists[:,:,2,:],axis=-1),'o-')
        plt.title(experiment)
        plt.ylabel('Random, dataset and internal model state dists',size=24)
        plt.xlabel('AL Iteration',size=24)

scoreList = np.asarray(scoreList)
rewardsList = np.asarray(rewardsList)
cumScoreList = np.asarray(cumScoreList)
distList = np.asarray(distList)

plt.figure()
plt.clf()
for i in range(len(scoreList)):
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(1, 21), np.average(scoreList[i],axis=0), 'o-',label=experiments[i])
    plt.ylabel('Cumulative Score', size=24)
    plt.xlabel('AL Iteration', size=24)
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(np.arange(1, 21), np.average(rewardsList[i],axis=0).transpose(), 'o-',label=experiments[i])
    plt.ylabel('Per-Iteration Reward', size=24)
    plt.xlabel('AL Iteration', size=24)
    plt.legend()
    plt.subplot(4, 3, 3)
    plt.plot(np.arange(1, 21), np.average(distList[i,:,:,2,:],axis=(1,-1)), 'o-',label=experiments[i])
    plt.subplot(4, 3, 6)
    plt.ylabel('Random, dataset and internal dists', size=24)
    plt.plot(np.arange(1, 21), np.average(distList[i,:,:,1,:],axis=(1,-1)), 'o-',label=experiments[i])
    plt.subplot(4, 3, 9)
    plt.plot(np.arange(1, 21), np.average(distList[i,:,:,0,:],axis=(1,-1)), 'o-',label=experiments[i])
    plt.xlabel('AL Iteration', size=24)
    plt.subplot(4, 3, 12)
    plt.plot(np.arange(1, 21), np.average(distList[i,:,:,0,:],axis=(1,-1)) + np.average(distList[i,:,:,1,:],axis=(1,-1)) - np.average(distList[i,:,:,2,:],axis=(1,-1)), 'o-',label=experiments[i])
    plt.ylabel('Combined Dist Metric')
    plt.xlabel('AL Iteration', size=24)
    plt.legend()