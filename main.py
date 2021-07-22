'''import statements'''
import activeLearner
from utils import get_input
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # annoying numpy error


'''
This code implements an active learning protocol, intended to optimize the binding score of aptamers to certain analytes

Modules:
> Controller
--> This script
--> Contains parameters and controls which modules are used

> Model
--> Inputs: dataset
--> Outputs: scores and confidence factors

> Sampler
--> Inputs: Model (as energy function)
--> Outputs: Uncertainty and/or score maxima

> Querier
--> Inputs: sampler, model
--> Outputs: query scoring function, Sequences to be scored by oracle

> Oracle
--> Inputs: sequences
--> Outputs: binding scores


To-Do
==> querier
    -> drop-in new models
==> model
    -> upgrade
==> combine MCMC with querier and think about batching
==> parallelize sampling runs
==> large-scale testing scripts
==> print list summaries, maybe a table - indeed, collate and collect all found optima (and test them against oracle? - maybe for cheap ones)
==> return accuracy not as minmum energy but as comparison to known optimum
==> incorporate sampling, models & queries for mixed-length sequences
==> decide on representation - currently binary
==> improved flag in sampler for when we aren't getting anything new
==> need to think about how we report closeness to true minimum - account for uncertainty?

low priority
==> check that relevant params (ensemble size) are properly overwritten when picking up old jobs (maybe we just don't care about old jobs)

'''

# initialize control parameters
params = {}
params['device'] = 'cluster' # 'local' or 'cluster'
params['explicit run enumeration'] = True # if this is True, the next run be fresh, in directory 'run%d'%run_num, if false, regular behaviour. Note: only use this on fresh runs

# get command line input
if params['device'] == 'cluster':
    input = get_input()
    params['run num'] = input[0]
    params['sampler seed'] = input[1]  # seed for MCMC modelling (each set of gammas gets a slightly different seed)
    params['model seed'] = input[2]  # seed used for model ensemble (each model gets a slightly different seed)
    params['dataset seed'] = input[3]  # if we are using a toy dataset, it may take a specific seed
elif params['device'] == 'local':
    params['run num'] = 0 # manual setting, for 0, do a fresh run, for != 0, pickup on a previous run.
    params['sampler seed'] = 0  # seed for MCMC modelling (each set of gammas gets a slightly different seed)
    params['model seed'] = 0  # seed used for model ensemble (each model gets a slightly different seed)
    params['dataset seed'] = 0  # if we are using a toy dataset, it may take a specific seed

# Pipeline parameters
params['pipeline iterations'] = 20 # number of cycles with the oracle
params['query mode'] = 'random' # 'random', 'score', 'uncertainty', 'heuristic', 'learned' # different modes for query construction

params['queries per iter'] = 100 # maximum number of questions we can ask the oracle per cycle
params['mode'] = 'training' # 'training'  'evaluation' 'initialize'
params['debug'] = False

# toy data parameters
params['dataset'] = 'toy'
params['init dataset length'] = 100 # number of items in the initial (toy) dataset
params['variable sample size'] = False # WIP - NON-FUNCTIONAL: if true, 'sample length' should be a list with the smallest and largest size of input sequences [min, max]
params['sample length'] = 40 # number of input dimensions

# model parameters
params['ensemble size'] = 10 # number of models in the ensemble
params['model filters'] = 20
params['model layers'] = 2 # for cluster batching
params['max training epochs'] = 200
params['GPU'] = 0 # run model on GPU - not yet tested, may not work at all
params['batch size'] = 10 # model training batch size

# sampler parameters
params['sampling time'] = 1e4
params['sampler gammas'] = 10 # minimum number of gammas over which to search for each sampler (if doing in parallel, we may do more if we have more CPUs than this)


#====================================
if params['mode'] == 'evaluation':
    params['pipeline iterations'] = 1

# paths
if params['device'] == 'cluster':
    params['workdir'] = '/home/kilgourm/scratch/learnerruns'
elif params['device'] == 'local':
    params['workdir'] = 'C:/Users\mikem\Desktop/activeLearningRuns'

#=====================================
if __name__ == '__main__':
    al = activeLearner.activeLearning(params)
    if params['mode'] == 'initalize':
        print("Initialized!")
    elif params['mode'] == 'training':
        al.runPipeline()
    elif params['mode'] == 'evaluation':
        sampleDict = al.sampleEnsemble()



