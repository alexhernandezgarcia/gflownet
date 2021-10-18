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

    return scores, rewards, cumulative_score

scoreList = []
rewardsList = []
cumScoreList = []

#experiments = ['A1', 'A2', 'A3', 'A4', 'A5', 'A7']
#experiments = ['B1', 'B2', 'B3','C1','C2','C3']
experiments = ['D5']

for experiment in experiments:
    experiments_dir = 'C:/Users\mikem\Desktop/activeLearningRuns\cluster\production/'

    scores, rewards, cumulative_score = getScores(experiment, experiments_dir)
    scoreList.append(scores)
    rewardsList.append(rewards)
    cumScoreList.append(cumulative_score)
    plt.figure()
    plt.clf()
    plt.subplot(1,2,1)
    plt.plot(np.arange(1,21),scores.transpose(),'o-')
    plt.title(experiment)
    plt.ylabel('Cumulative Score',size=24)
    plt.xlabel('AL Iteration',size=24)
    plt.subplot(1,2,2)
    plt.plot(np.arange(1,21),rewards.transpose(),'o-')
    plt.ylabel('Per-Iteration Reward',size=24)
    plt.xlabel('AL Iteration',size=24)

scoreList = np.asarray(scoreList)
rewardsList = np.asarray(rewardsList)
cumScoreList = np.asarray(cumScoreList)

plt.figure()
plt.clf()
for i in range(len(scoreList)):
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, 21), np.average(scoreList[i],axis=0), 'o-',label=experiments[i])
    plt.ylabel('Cumulative Score', size=24)
    plt.xlabel('AL Iteration', size=24)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, 21), np.average(rewardsList[i],axis=0).transpose(), 'o-',label=experiments[i])
    plt.ylabel('Per-Iteration Reward', size=24)
    plt.xlabel('AL Iteration', size=24)
    plt.legend()