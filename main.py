'''import statements'''
import activeLearner
from utils import *
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
==? test new pool method 
==> think carefully about how we split test and train datasets
==> large-scale testing scripts
==> print list summaries, maybe a table - indeed, collate and collect all found optima (and test them against oracle? - maybe for cheap ones)
==> return accuracy not as minimum energy but as comparison to known optimum
==> incorporate sampling, models & queries for mixed-length sequences
    -> maybe the last thing - look at how we can compute binary distances between sequences of different lengths

low priority
==> check that relevant params (ensemble size) are properly overwritten when picking up old jobs (maybe we just don't care about old jobs)

'''

# initialize control parameters
params = {}
params['device'] = 'cluster' # 'local' or 'cluster'
params['explicit run enumeration'] = True # if this is True, the next run be fresh, in directory 'run%d'%run_num, if false, regular behaviour. Note: only use this on fresh runs
params['test mode'] = False # WIP # if true, automatically set parameters for a quick test run

# get command line input
if params['device'] == 'cluster':
    input = get_input()
    params['run num'] = input[0]
    params['sampler seed'] = input[1]  # seed for MCMC modelling (each set of gammas gets a slightly different seed)
    params['model seed'] = input[2]  # seed used for model ensemble (each model gets a slightly different seed)
    params['dataset seed'] = input[3]  # if we are using a toy dataset, it may take a specific seed
    params['query mode'] = input[4]  # 'random', 'score', 'uncertainty', 'heuristic', 'learned' # different modes for query construction

elif params['device'] == 'local':
    params['run num'] = 0 # manual setting, for 0, do a fresh run, for != 0, pickup on a previous run.
    params['sampler seed'] = 0  # seed for MCMC modelling (each set of gammas gets a slightly different seed)
    params['model seed'] = 0  # seed used for model ensemble (each model gets a slightly different seed)
    params['dataset seed'] = 0  # if we are using a toy dataset, it may take a specific seed
    params['query mode'] = 'score'  # 'random', 'score', 'uncertainty', 'heuristic', 'learned' # different modes for query construction

# Pipeline parameters
params['pipeline iterations'] = 20 # number of cycles with the oracle

params['queries per iter'] = 1000 # maximum number of questions we can ask the oracle per cycle
params['mode'] = 'training' # 'training'  'evaluation' 'initialize'
params['debug'] = False

# toy data parameters
params['dataset'] = 'toy'
params['init dataset length'] = 1000 # number of items in the initial (toy) dataset
params['variable sample size'] = False # WIP - NON-FUNCTIONAL: if true, 'max sample length' should be a list with the smallest and largest size of input sequences [min, max]
params['min sample length'] = 11 # minimum input sequence length
params['max sample length'] = 20 # maximum input sequence length (inclusive) - or fixed sample size if 'variable sample size' is false

# model parameters
params['ensemble size'] = 10 # number of models in the ensemble
params['model filters'] = 12
params['model layers'] = 2 # for cluster batching
params['embed dim'] = 4 # embedding dimension
params['max training epochs'] = 200
params['GPU'] = 0 # run model on GPU - not yet tested, may not work at all
params['batch size'] = 10 # model training batch size
if params['variable sample size']:
    params['batch size'] = 1 # we can only train one at a time right now with variable length models

# sampler parameters
params['sampling time'] = 1e5
params['sampler gammas'] = 10 # minimum number of gammas over which to search for each sampler (if doing in parallel, we may do more if we have more CPUs than this)


#====================================
if params['mode'] == 'evaluation':
    params['pipeline iterations'] = 1

if params['test mode']:
    params['pipeline iterations'] = 2
    params['init dataset length'] = 100
    params['queries per iter'] = 100
    params['sampling time'] = 1e3
    params['sampler gammas'] = 3
    params['ensemble size'] = 3
    params['max training epochs'] = 10


# paths
if params['device'] == 'cluster':
    params['workdir'] = '/home/kilgourm/scratch/learnerruns'
elif params['device'] == 'local':
    params['workdir'] = 'C:/Users\mikem\Desktop/activeLearningRuns'

#=====================================
if __name__ == '__main__':
    al = activeLearner.activeLearning(params)
    if params['mode'] == 'initalize':
        printRecord("Initialized!")
    elif params['mode'] == 'training':
        al.runPipeline()
    elif params['mode'] == 'evaluation':
        sampleDict = al.sampleEnsemble()



