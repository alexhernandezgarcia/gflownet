'''import statements'''
import activeLearner
from utils import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # annoying numpy error


'''
This code implements an active learning protocol for global minimization of some function

To-Do
==> testing
    -> model size
    -> batch selection methods
    -> analysis methods
==> draw a nice diagram for the team which explains all the moving parts

low priority /long term
==> check that relevant params (ensemble size) are properly overwritten when picking up old jobs
==> think carefully about how we split test and train datasets
==> augmentation and other regularization
==> maybe print outputs at the end of each iteration as a lovely table
==> add a debug mode for saving training results
==> characterize and track specific local minima, including suboptimal ones, over iterations
==> optimize transformer initialization

'''

# initialize control parameters
params = {}
params['device'] = 'local' # 'local' or 'cluster'
params['GPU'] = False # WIP - train and evaluate models on GPU
params['explicit run enumeration'] = False # if this is True, the next run be fresh, in directory 'run%d'%run_num, if false, regular behaviour. Note: only use this on fresh runs
params['test mode'] = False # WIP # if true, automatically set parameters for a quick test run

# get command line input
if params['device'] == 'cluster':
    input = get_input()
    params['run num'] = input[0]
    params['sampler seed'] = input[1]  # seed for MCMC modelling (each set of gammas gets a slightly different seed)
    params['model seed'] = input[2]  # seed used for model ensemble (each model gets a slightly different seed)
    params['dataset seed'] = input[3]  # if we are using a toy dataset, it may take a specific seed
    params['query mode'] = input[4]  # 'random', 'score', 'uncertainty', 'heuristic', 'learned' # different modes for query construction
    params['dataset'] = input[5]
elif params['device'] == 'local':
    params['run num'] = 0 # manual setting, for 0, do a fresh run, for != 0, pickup on a previous run.
    params['sampler seed'] = 0  # seed for MCMC modelling (each set of gammas gets a slightly different seed)
    params['model seed'] = 0  # seed used for model ensemble (each model gets a slightly different seed)
    params['dataset seed'] = 0  # if we are using a toy dataset, it may take a specific seed
    params['query mode'] = 'score'  # 'random', 'score', 'uncertainty', 'heuristic', 'learned' # different modes for query construction
    params['dataset'] = 'nupack'  # 'linear', 'inner product', 'potts', 'seqfold', 'nupack' in order of increasing complexity. Note distributions particulary for seqfold and nupack may not be natively well-behaved.

# Pipeline parameters
params['pipeline iterations'] = 10 # number of cycles with the oracle
params['distinct minima'] = 10 # number of distinct minima
params['minima dist cutoff'] = 0.2 # minimum distance (normalized, binary) between distinct minima

params['queries per iter'] = 1000 # maximum number of questions we can ask the oracle per cycle
params['mode'] = 'training' # 'training'  'evaluation' 'initialize'
params['debug'] = True
params['training parallelism'] = True # distribute training across a CPU multiprocessing pool (each CPU may still access a GPU, if GPU == True)

# toy data parameters
params['dataset type'] = 'toy' # oracle is very fast to sample
params['init dataset length'] = 100000 # number of items in the initial (toy) dataset
params['dict size'] = 4 # number of possible choices per-state, e.g., [0,1] would be two, [1,2,3,4] (representing ATGC) would be 4
params['variable sample length'] = True #if true, 'max sample length' should be a list with the smallest and largest size of input sequences [min, max]. If 'false', model is MLP, if 'true', transformer encoder -> MLP output
params['min sample length'], params['max sample length'] = [10, 20] # minimum input sequence length and # maximum input sequence length (inclusive) - or fixed sample size if 'variable sample length' is false

# model parameters
params['ensemble size'] = 2 # number of models in the ensemble
params['model filters'] = 64
params['model layers'] = 8 # for cluster batching
params['embed dim'] = 64 # embedding dimension
params['max training epochs'] = 100
params['batch size'] = 1000 # model training batch size

# sampler parameters
params['sampling time'] = int(1e4)
params['sampler gammas'] = 20 # minimum number of gammas over which to search for each sampler (if doing in parallel, we may do more if we have more CPUs than this)


#====================================
if params['mode'] == 'evaluation':
    params['pipeline iterations'] = 1

if params['test mode']:
    params['pipeline iterations'] = 2
    params['init dataset length'] = 100
    params['queries per iter'] = 100
    params['sampling time'] = int(1e3)
    params['sampler gammas'] = 2
    params['ensemble size'] = 2
    params['max training epochs'] = 2
    params['model filters'] = 12
    params['model layers'] = 1  # for cluster batching
    params['embed dim'] = 12  # embedding dimension
    params['batch size'] = 10  # model training batch size
    params['min sample length'], params['max sample length'] = [10, 20]  # minimum input sequence length and # maximum input sequence length (inclusive) - or fixed sample size if 'variable sample length' is false
    params['dict size'] = 4  # number of possible choices per-state, e.g., [0,1] would be two, [1,2,3,4] (representing ATGC) would be 4

# paths
if params['device'] == 'cluster':
    params['workdir'] = '/home/kilgourm/scratch/learnerruns'
elif params['device'] == 'local':
    params['workdir'] = '/home/mkilgour/learnerruns'#'C:/Users\mikem\Desktop/activeLearningRuns'#

#=====================================
if __name__ == '__main__':
    al = activeLearner.activeLearning(params)
    if params['mode'] == 'initalize':
        printRecord("Initialized!")
    elif params['mode'] == 'training':
        al.runPipeline()
    elif params['mode'] == 'evaluation':
        ValueError("No function for this! Write a function to load torch models and evaluate inputs.")


