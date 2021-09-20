"""
This code implements an active learning protocol for global minimization of some function

# TODO
==> incorporate gFlowNet
    -> model state calculation
    -> training and sampling print statements
        => training performance
        => sample quality e.g., diversity, span, best scores averages, whatever
        -> print flag on gflownet convergence - epoch limit OR loss convergence
    -> hardcode padding rules - in case it is poorly trained, it should never be able to add a base after the end of the chain
    -> iteratively resample gflownet to remove duplicates until desired sample number is reached 
    -> merge gflownet oracles with standard oracle class
    -> switch gflownet training tqdm from iters to log convergence
    -> make sure gflownet scores are aligned with AL optimization target (minimization)
    -> add option for test mode to slash model size and training epochs
==> RL training and testing
==> add a function for tracking dataset distances and adjusting the cutoff
==> update and test beluga requirements


low priority /long term
==> consider augment binary distance metric with multi-base motifs - or keep current setup (minimum single mutations)
==> check that relevant params (ensemble size) are properly overwritten when picking up old jobs 
==> augmentation regularization
==> maybe print outputs at the end of each iteration as a lovely table
==> add detection for oracle.initializeDataset for if the requested number of samples is a significant fraction of the total sample space - may be faster to return full space or large fraction of all permutations

known issues
==> training parallelism hangs on iteration #2 on linux

"""
print("Imports...", end="")
import sys
from argparse import ArgumentParser
import yaml
from comet_ml import Experiment
from pathlib import Path
import activeLearner
from utils import *
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # annoying numpy error


def get_config(args, override_args, args2config, save=True):
    """
    Combines YAML configuration file, command line arguments and default arguments into
    a single configuration dictionary.

    - Values in YAML file override default values
    - Command line arguments override values in YAML file

    Returns
    -------
        Namespace
    """

    def _update_config(arg, val, config, override=False):
        config_aux = config
        for k in args2config[arg]:
            if k not in config_aux:
                if k is args2config[arg][-1]:
                    config_aux.update({k: val})
                else:
                    config_aux.update({k: {}})
                    config_aux = config_aux[k]
            else:
                if k is args2config[arg][-1] and override:
                    config_aux[k] = val
                else:
                    config_aux = config_aux[k]


    # Read YAML config
    if args.yaml_config:
        yaml_path = Path(args.yaml_config)
        assert yaml_path.exists()
        assert yaml_path.suffix in {".yaml", ".yml"}
        with yaml_path.open("r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    # Add args to config: add if not provided; override if in command line
    override_args = [arg.strip("--") for arg in override_args if "--" in arg]
    for k, v in vars(args).items():
        print(k, v)
        if k in override_args:
            _update_config(k, v, config, override=True)
        else:
            _update_config(k, v, config, override=False)
    return dict2namespace(config)


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """
    args2config = {}
    # YAML config
    parser.add_argument(
        "-y",
        "--yaml_config",
        default=None,
        type=str,
        help="YAML configuration file",
    )
    args2config.update({"yaml_config": ["yaml_config"]})
    # General
    parser.add_argument(
        "--test_mode",
        action="store_true",
        default=False,
        help="Set parameters for a quick test run",
    )
    args2config.update({"test_mode": ["test_mode"]})
    parser.add_argument("--debug", action="store_true", default=False)
    args2config.update({"debug": ["debug"]})
    parser.add_argument("--run_num", type=int, default=0, help="Experiment ID")
    args2config.update({"run_num": ["run_num"]})
    parser.add_argument(
        "--explicit_run_enumeration",
        action="store_true",
        default=False,
        help="If True, the next run be fresh, in directory 'run%d'%run_num; if False, regular behaviour. Note: only use this on fresh runs",
    )
    args2config.update({"explicit_run_enumeration": ["explicit_run_enumeration"]})
    # Seeds
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=0,
        help="Seed for MCMC modelling (each set of gammas gets a different seed)",
    )
    args2config.update({"sampler_seed": ["seeds", "sampler"]})
    parser.add_argument(
        "--model_seed",
        type=int,
        default=0,
        help="seed used for model ensemble (each model gets a different seed)",
    )
    args2config.update({"model_seed": ["seeds", "model"]})
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=0,
        help="if we are using a toy dataset, it may take a specific seed",
    )
    args2config.update({"dataset_seed": ["seeds", "dataset"]})
    parser.add_argument(
        "--toy_oracle_seed",
        type=int,
        default=0,
        help="if we are using a toy dataset, it may take a specific seed",
    )
    args2config.update({"toy_oracle_seed": ["seeds", "toy_oracle"]})
    parser.add_argument(
        "--machine",
        type=str,
        default="local",
        help="'local' or 'cluster' (assumed linux env)",
    )
    args2config.update({"machine": ["machine"]})
    parser.add_argument("--device", default="cuda", type=str, help="'cuda' or 'cpu'")
    args2config.update({"device": ["device"]})
    parser.add_argument("--workdir", type=str, default=None, help="Working directory")
    args2config.update({"workdir": ["workdir"]})
    # Dataset
    parser.add_argument("--dataset", type=str, default="linear")
    args2config.update({"dataset": ["dataset", "oracle"]})
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="toy",
        help="Toy oracle is very fast to sample",
    )
    args2config.update({"dataset_type": ["dataset", "type"]})
    parser.add_argument(
        "--init_dataset_length",
        type=int,
        default=int(1e2),
        help="number of items in the initial (toy) dataset",
    )
    args2config.update({"init_dataset_length": ["dataset", "init_length"]})
    parser.add_argument(
        "--dict_size",
        type=int,
        default=4,
        help="number of possible choices per-state, e.g., [0,1] would be two, [1,2,3,4] (representing ATGC) would be 4 - with variable length, 0's are added for padding",
    )
    args2config.update({"dict_size": ["dataset", "dict_size"]})
    parser.add_argument(
        "--fixed_sample_length",
        dest="variable_sample_length",
        action="store_false",
        default=True,
        help="models will sample within ranges set below",
    )
    args2config.update({"variable_sample_length": ["dataset", "variable_length"]})
    parser.add_argument("--min_sample_length", type=int, default=10)
    args2config.update({"min_sample_length": ["dataset" "min_length"]})
    parser.add_argument("--max_sample_length", type=int, default=40)
    args2config.update({"max_sample_length": ["dataset", "max_length"]})
    parser.add_argument(
        "--sample_tasks",
        type=int,
        default=1,
        help="WIP unfinished for multi-task training - how many outputs per oracle? (only nupack currently  setup for > 1 output)",
    )
    args2config.update({"sample_tasks": ["dataset", "sample_tasks"]})
    # AL
    parser.add_argument(
        "--sample_method", type=str, default="gflownet", help="'mcmc', 'gflownet'"
    )
    args2config.update({"sample_method": ["al", "sample_method"]})
    parser.add_argument(
        "--query_mode",
        type=str,
        default="learned",
        help="'random', 'energy', 'uncertainty', 'heuristic', 'learned' # different modes for query construction",
    )
    args2config.update({"query_mode": ["al", "query_mode"]})
    parser.add_argument(
        "--pipeline_iterations",
        type=int,
        default=1,
        help="number of cycles with the oracle",
    )
    args2config.update({"pipeline_iterations": ["al", "n_iter"]})
    parser.add_argument(
        "--query_selection",
        type=str,
        default="clustering",
        help="agglomerative 'clustering', 'cutoff' or strictly 'argmin' based query construction",
    )
    args2config.update({"query_selection": ["al", "query_selection"]})
    parser.add_argument(
        "--minima_dist_cutoff",
        type=float,
        default=0.25,
        help="minimum distance (normalized, binary) between distinct minima or between clusters in agglomerative clustering OR 'cutoff' batch selection",
    )
    args2config.update({"minima_dist_cutoff": ["al", "minima_dist_cutoff"]})
    parser.add_argument(
        "--queries_per_iter",
        type=int,
        default=100,
        help="maximum number of questions we can ask the oracle per cycle",
    )
    args2config.update({"queries_per_iter": ["al", "queries_per_iter"]})
    parser.add_argument(
        "--mode",
        type=str,
        default="training",
        help="'training'  'evaluation' 'initialize' - only training currently useful",
    )
    args2config.update({"mode": ["al", "mode"]})
    # Querier
    parser.add_argument(
        "--model_state_size",
        type=int,
        default=30,
        help="number of selected datapoints of model evaluations",
    )
    args2config.update({"model_state_size": ["querier", "model_state_size"]})
    parser.add_argument(
        "--qmodel_opt", type=str, default="SGD", help="optimizer for q-network"
    )
    args2config.update({"qmodel_opt": ["querier", "opt"]})
    parser.add_argument(
        "--qmodel_momentum", type=float, default=0.95, help="momentum for q-network"
    )
    args2config.update({"qmodel_momentum": ["querier", "momentum"]})
    parser.add_argument(
        "--qmodel_preload_path",
        type=str,
        default=None,
        help="location of pre-trained qmodel",
    )
    args2config.update({"qmodel_preload_path": ["querier", "model_ckpt"]})
    parser.add_argument("--querier_latent_space_width", type=int, default=10)
    args2config.update(
        {"querier_latent_space_width": ["querier", "latent_space_width"]}
    )
    # GFlowNet
    parser.add_argument("--gflownet_model_ckpt", default=None, type=str)
    args2config.update({"gflownet_model_ckpt": ["gflownet" "model_ckpt"]})
    parser.add_argument("--gflownet_progress", action="store_true")
    args2config.update({"gflownet_progress": ["gflownet", "progress"]})
    parser.add_argument(
        "--gflownet_learning_rate", default=1e-4, help="Learning rate", type=float
    )
    args2config.update({"gflownet_learning_rate": ["gflownet", "learning_rate"]})
    parser.add_argument("--gflownet_opt", default="adam", type=str)
    args2config.update({"gflownet_opt": ["gflownet", "opt"]})
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    args2config.update({"adam_beta1": ["gflownet", "adam_beta1"]})
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    args2config.update({"adam_beta2": ["gflownet", "adam_beta2"]})
    parser.add_argument("--gflownet_momentum", default=0.9, type=float)
    args2config.update({"gflownet_momentum": ["gflownet", "momentum"]})
    parser.add_argument("--gflownet_mbsize", default=16, help="Minibatch size", type=int)
    args2config.update({"gflownet_mbsize": ["gflownet", "mbsize"]})
    parser.add_argument("--train_to_sample_ratio", default=1, type=float)
    args2config.update({"train_to_sample_ratio": ["gflownet", "train_to_sample_ratio"]})
    parser.add_argument("--gflownet_n_hid", default=256, type=int)
    args2config.update({"gflownet_n_hid": ["gflownet", "n_hid"]})
    parser.add_argument("--gflownet_n_layers", default=2, type=int)
    args2config.update({"gflownet_n_layers": ["gflownet", "n_layers"]})
    parser.add_argument(
        "--gflownet_n_iter", default=20000, type=int, help="gflownet training steps"
    )
    args2config.update({"gflownet_n_iter": ["gflownet", "n_iter"]})
    parser.add_argument(
        "--num_empirical_loss",
        default=200000,
        type=int,
        help="Number of samples used to compute the empirical distribution loss",
    )
    args2config.update({"num_empirical_loss": ["gflownet", "num_empirical_loss"]})
    parser.add_argument(
        "--no_batch_reward",
        dest="batch_reward",
        action="store_false",
        default=True,
        help="If True, compute rewards after batch is formed",
    )
    parser.add_argument(
        "--gflownet_n_samples",
        type=int,
        default=1000,
        help="Sequences to sample from GFLowNet",
    )
    args2config.update({"gflownet_n_samples": ["gflownet", "n_samples"]})
    args2config.update({"batch_reward": ["gflownet", "batch_reward"]})
    parser.add_argument("--bootstrap_tau", default=0.0, type=float)
    args2config.update({"bootstrap_tau": ["gflownet", "bootstrap_tau"]})
    parser.add_argument("--clip_grad_norm", default=0.0, type=float)
    args2config.update({"clip_grad_norm": ["gflownet", "clip_grad_norm"]})
    parser.add_argument("--comet_project", default=None, type=str)
    args2config.update({"comet_project": ["gflownet", "comet", "project"]})
    parser.add_argument(
        "-t", "--tags", nargs="*", help="Comet.ml tags", default=[], type=str
    )
    args2config.update({"tags": ["gflownet", "comet", "tags"]})
    # Proxy model
    parser.add_argument(
        "--proxy_model_type",
        type=str,
        default="mlp",
        help="type of proxy model - mlp or transformer",
    )
    args2config.update({"proxy_model_type": ["proxy", "model_type"]})
    parser.add_argument(
        "--training_parallelism",
        action="store_true",
        default=False,
        help="fast enough on GPU without paralellism - True doesn't always work on linux",
    )
    args2config.update({"training_parallelism": ["proxy", "training_parallelism"]})
    parser.add_argument(
        "--proxy_model_ensemble_size",
        type=int,
        default=10,
        help="number of models in the ensemble",
    )
    args2config.update({"proxy_model_ensemble_size": ["proxy", "ensemble_size"]})
    parser.add_argument(
        "--proxy_model_width",
        type=int,
        default=256,
        help="number of neurons per proxy NN layer",
    )
    args2config.update({"proxy_model_width": ["proxy", "width"]})
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=256,
        help="embedding dimension for transformer only",
    )
    args2config.update({"embedding_dim": ["proxy", "embedding_dim"]})
    parser.add_argument(
        "--proxy_model_layers",
        type=int,
        default=2,
        help="number of layers in NN proxy models (transformer encoder layers OR MLP layers)",
    )
    args2config.update({"proxy_model_layers": ["proxy", "n_layers"]})
    parser.add_argument("--proxy_training_batch_size", type=int, default=10)
    args2config.update({"proxy_training_batch_size": ["proxy", "mbsize"]})
    parser.add_argument("--proxy_max_epochs", type=int, default=200)
    args2config.update({"proxy_max_epochs": ["proxy", "max_epochs"]})
    parser.add_argument(
        "--proxy_no_shuffle_dataset",
        dest="proxy_shuffle_dataset",
        action="store_false",
        default=True,
        help="give each model in the ensemble a uniquely shuffled dataset",
    )
    args2config.update({"proxy_shuffle_dataset": ["proxy", "shuffle_dataset"]})
    # MCMC
    parser.add_argument(
        "--mcmc_sampling_time",
        type=int,
        default=int(1e4),
        help="at least 1e4 is recommended for convergence",
    )
    args2config.update({"mcmc_sampling_time": ["mcmc", "sampling_time"]})
    parser.add_argument(
        "--mcmc_num_samplers",
        type=int,
        default=40,
        help="minimum number of gammas over which to search for each sampler (if doing in parallel, we may do more if we have more CPUs than this)",
    )
    args2config.update({"mcmc_num_samplers": ["mcmc", "num_samplers"]})
    parser.add_argument("--stun_min_gamma", type=float, default=-3)
    args2config.update({"stun_min_gamma": ["mcmc", "stun_min_gamma"]})
    parser.add_argument("--stun_max_gamma", type=float, default=1)
    args2config.update({"stun_max_gamma": ["mcmc", "stun_max_gamma"]})
    return parser, args2config


def process_config(config):
    # Normalize seeds
    config.seeds.model = config.seeds.model % 10
    config.seeds.dataset = config.seeds.dataset % 10
    config.seeds.toy_oracle = config.seeds.toy_oracle % 10
    config.seeds.sampler = config.seeds.sampler % 10
    # Evaluation mode
    if config.al.mode == "evaluation":
        config.al.pipeline_iterations = 1
    # Test mode
    if config.test_mode:
        config.gflownet.n_train_steps = 100
        config.al.pipeline_iterations = 3
        config.dataset.init_length = 100
        config.al.queries_per_iter = 100
        config.mcmc.sampling_time = int(1e3)
        config.mcmc.num_samplers = 2
        config.proxy.ensemble_size = 2
        config.proxy.max_epochs = 5
        config.proxy.width = 12
        config.proxy.n_layers = 1  # for cluster batching
        config.proxy.embedding_dim = 12  # embedding dimension
        config.proxy.mbsize = 10  # model training batch size
        config.dataset.min_length, config.dataset.max_length = [
            10,
            20,
        ]
        config.dataset.dict_size = 4
    # GFlowNet
    config.gflownet.horizon = config.dataset.max_length
    config.gflownet.nalphabet = config.dataset.dict_size
    config.gflownet.func = config.dataset
    # Paths
    if not config.workdir and config.machine == "cluster":
        config.workdir = "/home/kilgourm/scratch/learnerruns"
    elif not config.workdir and config.machine == "local":
        config.workdir = (
            "C:/Users\mikem\Desktop/activeLearningRuns"  #'/home/mkilgour/learnerruns'#
        )
    return config


if __name__ == "__main__":
    # Handle command line arguments and configuration
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser, args2config = add_args(parser)
    args = parser.parse_args()
    config = get_config(args, override_args, args2config)
    config = process_config(config)
#     print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(config).items()]))
# TODO: save final config in workdir
    al = activeLearner.ActiveLearning(config)
    if config.al.mode == "initalize":
        printRecord("Initialized!")
    elif config.al.mode == "training":
        al.runPipeline()
    elif config.al.mode == "evaluation":
        ValueError(
            "No function for this! Write a function to load torch models and evaluate inputs."
        )
