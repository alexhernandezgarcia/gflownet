'''import statements'''
import activeLearner
from utils import *
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # annoying numpy error

'''
This code implements an active learning protocol for global minimization of some function

# TODO
==> incorporate gFlowNet
==> incorporate 
==> augment binary distance metric with multi-base motifs
==> see if we can do dist metric faster in one-hot encoding

low priority /long term
==> speedtest one-hot binary distance check
==> consider augment binary distance metric with multi-base motifs - or keep current setup (minimum single mutations)
==> check that relevant params (ensemble size) are properly overwritten when picking up old jobs
==> think carefully about how we split test and train datasets
==> augmentation and other regularization
==> maybe print outputs at the end of each iteration as a lovely table

known issues
==> training parallelism hangs on iteration #2 on linux

'''

# get command line input
parser = argparse.ArgumentParser()
# high level
parser.add_argument('--run_num', type=int, default=0)
parser.add_argument('--sampler_seed', type=int, default=0) # seed for MCMC modelling (each set of gammas gets a slightly different seed)
parser.add_argument('--model_seed', type=int, default=0) # seed used for model ensemble (each model gets a slightly different seed)
parser.add_argument('--init_dataset_seed', type=int, default=0) # if we are using a toy dataset, it may take a specific seed
parser.add_argument('--toy_oracle_seed', type=int, default=0) # if we are using a toy dataset, it may take a specific seed
parser.add_argument('--device', type = str, default = 'local') # 'local' or 'cluster' (assumed linux env)
parser.add_argument('--GPU', type = bool, default = True) # train and evaluate models on GPU
parser.add_argument('--explicit_run_enumeration', type = bool, default = False) # if this is True, the next run be fresh, in directory 'run%d'%run_num, if false, regular behaviour. Note: only use this on fresh runs
# dataset settings
parser.add_argument('--dataset_type', type = str, default = 'toy') # Toy oracle is very fast to sample
parser.add_argument('--dataset', type=str, default='linear')
parser.add_argument('--init_dataset_length', type = int, default = int(1e2)) # number of items in the initial (toy) dataset
parser.add_argument('--dict_size', type = int, default = 4) # number of possible choices per-state, e.g., [0,1] would be two, [1,2,3,4] (representing ATGC) would be 4 - with variable length, 0's are added for padding
parser.add_argument('--variable_sample_length', type = bool, default = True) # models will sample within ranges set below
parser.add_argument('--min_sample_length', type = int, default = 10)
parser.add_argument('--max_sample_length', type = int, default = 40)
# AL settings
parser.add_argument('--query_mode', type=str, default='learned') # 'random', 'energy', 'uncertainty', 'heuristic', 'learned' # different modes for query construction
parser.add_argument('--test_mode', type = bool, default = True) # if true, automatically set parameters for a quick test run
parser.add_argument('--pipeline_iterations', type = int, default = 20) # number of cycles with the oracle
parser.add_argument('--minima_dist_cutoff', type = float, default = 0.25) # minimum distance (normalized, binary) between distinct minima or between clusters in agglomerative clustering
# TODO add toggle between agglomerative clustering and simple item-by-item batching
parser.add_argument('--queries_per_iter', type = int, default = 100) # maximum number of questions we can ask the oracle per cycle
parser.add_argument('--mode', type = str, default = 'training') # 'training'  'evaluation' 'initialize' - only training currently useful
parser.add_argument('--debug', type = bool, default = False)
# querier settings
parser.add_argument('--model_state_size', type = int, default = 30) # number of selected datapoints of model evaluations
parser.add_argument('--qmodel_opt', type = str, default = 'SGD') # optimizer for q-network
parser.add_argument('--qmodel_momentum', type = float, default = 0.95) # momentum for q-network
parser.add_argument('--qmodel_preload_path', type = str, default = None) # location of pre-trained qmodel
# gFlownet settings

# proxy model settings
parser.add_argument('--proxy_model_type', type = str, default = 'mlp') # type of proxy model - mlp or transformer
parser.add_argument('--training_parallelism', type = bool, default = False) # fast enough on GPU without paralellism - True doesn't always work on linux
parser.add_argument('--proxy_model_ensemble_size', type = int, default = 10) # number of models in the ensemble
parser.add_argument('--proxy_model_width', type = int, default = 64) # number of neurons per proxy NN layer
parser.add_argument('--embedding_dim', type = int, default = 64) # embedding dimension for transformer only
parser.add_argument('--proxy_model_layers', type = int, default = 4) # number of layers in NN proxy models (transformer encoder layers OR MLP layers)
parser.add_argument('--proxy_training_batch_size', type = int, default = 10)
parser.add_argument('--proxy_max_epochs', type = int, default = 200)
parser.add_argument('--proxy_shuffle_dataset', type = bool, default = False) # give each model in the ensemble a uniquely shuffled dataset
#sampler settings
parser.add_argument('--mcmc_sampling_time', type = int, default = int(1e4)) # at least 1e4 is recommended for convergence
parser.add_argument('--mcmc_num_samplers', type = int, default = 20) # minimum number of gammas over which to search for each sampler (if doing in parallel, we may do more if we have more CPUs than this)
parser.add_argument('--stun_min_gamma', type = float, default = -3)
parser.add_argument('--stun_max_gamma', type = float, default = 1)

params = parser.parse_args()

# normalize seeds
params.model_seed = params.model_seed % 10
params.init_dataset_seed = params.init_dataset_seed % 10
params.toy_oracle_seed = params.toy_oracle_seed % 10
params.sampler_seed = params.sampler_seed % 10


#====================================
if params.mode == 'evaluation':
    params.pipeline_iterations = 1

if params.test_mode:
    params.pipeline_iterations = 3
    params.init_dataset_length = 100
    params.queries_per_iter = 100
    params.mcmc_sampling_time = int(1e3)
    params.mcmc_num_samplers = 2
    params.proxy_model_ensemble_size = 2
    params.proxy_max_epochs = 5
    params.proxy_model_width = 12
    params.proxy_model_layers = 1  # for cluster batching
    params.embedding_dim = 12  # embedding dimension
    params.proxy_training_batch_size = 10  # model training batch size
    params.min_sample_length, params.max_sample_length = [10, 20]  # minimum input sequence length and # maximum input sequence length (inclusive) - or fixed sample size if 'variable sample length' is false
    params.dict_size = 4  # number of possible choices per-state, e.g., [0,1] would be two, [1,2,3,4] (representing ATGC) would be 4

# paths
if params.device == 'cluster':
    params.workdir = '/home/kilgourm/scratch/learnerruns'
elif params.device == 'local':
    params.workdir = 'C:/Users\mikem\Desktop/activeLearningRuns'#'/home/mkilgour/learnerruns'#

#=====================================
if __name__ == '__main__':
    al = activeLearner.activeLearning(params)
    if params.mode == 'initalize':
        printRecord("Initialized!")
    elif params.mode == 'training':
        al.runPipeline()
    elif params.mode == 'evaluation':
        ValueError("No function for this! Write a function to load torch models and evaluate inputs.")


