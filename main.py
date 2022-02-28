"""
This code implements an active learning protocol for global minimization of some function

# TODO
==> incorporate gFlowNet
    -> training and sampling print statements
        => sample quality e.g., diversity, span, best scores averages, whatever
        -> print flag on gflownet convergence - epoch limit OR loss convergence
    -> iteratively resample gflownet to remove duplicates until desired sample number is reached
    -> merge gflownet oracles with standard oracle class
==> RL training and testing
==> comet for key outputs (reward, toy score)

low priority /long term
==> consider augment binary distance metric with multi-base motifs - or keep current setup (minimum single mutations)
==> check that relevant params (ensemble size) are properly overwritten when picking up old jobs
==> augmentation regularization
==> maybe print outputs at the end of each iteration as a lovely table
==> add detection for oracle.initializeDataset for if the requested number of samples is a significant fraction of the total sample space - may be faster to return full space or large fraction of all permutations

known issues

"""
print("Imports...", end="")
import sys
from argparse import ArgumentParser
from comet_ml import Experiment
import activeLearner
from utils import *
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)  # annoying numpy error


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
    parser.add_argument("--no_debug", action="store_false", dest="debug", default=False)
    args2config.update({"no_debug": ["debug"]})
    parser.add_argument("--run_num", type=int, default=0, help="Experiment ID")
    args2config.update({"run_num": ["run_num"]})
    parser.add_argument(
        "--explicit_run_enumeration",
        action="store_true",
        default=False,
        help="If True, the next run be fresh, in directory 'run%d'%run_num; if False, regular behaviour. Note: only use this on fresh runs",
    )
    args2config.update({"explicit_run_enumeration": ["explicit_run_enumeration"]})
    parser.add_argument("--comet_project", default=None, type=str)
    args2config.update({"comet_project": ["comet_project"]})
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
        "--gflownet_seed",
        type=int,
        default=0,
        help="Seed for GFlowNet random number generator",
    )
    args2config.update({"gflownet_seed": ["seeds", "gflownet"]})
    # Misc
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
    parser.add_argument(
        "--dataset", type=str, default="linear"
    )  # 'linear' 'potts' 'nupack energy' 'nupack pairs' 'nupack pins'
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
    args2config.update({"min_sample_length": ["dataset", "min_length"]})
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
        "--sample_method",
        type=str,
        default="gflownet",
        help="'mcmc', 'gflownet', 'random'",
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
        "--acquisition_function",
        type=str,
        default="learned",
        help="'none', 'ucb','ei' # different 'things to do' with model uncertainty",
    )
    args2config.update({"acquisition_function": ["al", "acquisition_function"]})
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
        "--UCB_kappa",
        type=float,
        default=0.1,
        help="wighting of the uncertainty in BO-UCB acquisition function",
    )
    args2config.update({"UCB_kappa": ["al", "UCB_kappa"]})
    parser.add_argument(
        "--EI_max_percentile",
        type=float,
        default=80,
        help="max percentile for expected improvement (EI) acquisition function",
    )
    args2config.update({"EI_max_percentile": ["al", "EI_max_percentile"]})
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
    parser.add_argument("--q_network_width", type=int, default=10)
    args2config.update({"q_network_width": ["al", "q_network_width"]})
    parser.add_argument(
        "--agent_buffer_size",
        type=int,
        default=10000,
        help="RL agent buffer size",
    )
    args2config.update({"agent_buffer_size": ["al", "buffer_size"]})
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="RL episodes (runs of the full AL pipeline)",
    )
    args2config.update({"episodes": ["al", "episodes"]})
    parser.add_argument(
        "--action_state_size",
        type=int,
        default=1,
        help="number of actions RL agent can choose from",
    )
    args2config.update({"action_state_size": ["al", "action_state_size"]})
    parser.add_argument("--hyperparams_learning", action="store_true")
    args2config.update({"hyperparams_learning": ["al", "hyperparams_learning"]})
    parser.add_argument(
        "--tags_al", nargs="*", help="Comet.ml tags", default=[], type=str
    )
    args2config.update({"tags_al": ["al", "comet", "tags"]})
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

    # GFlowNet
    parser.add_argument(
        "--gflownet_device", default="cpu", type=str, help="'cuda' or 'cpu'"
    )
    args2config.update({"gflownet_device": ["gflownet", "device"]})
    parser.add_argument("--gflownet_model_ckpt", default=None, type=str)
    args2config.update({"gflownet_model_ckpt": ["gflownet", "model_ckpt"]})
    parser.add_argument("--gflownet_reload_ckpt", action="store_true")
    args2config.update({"gflownet_reload_ckpt": ["gflownet", "reload_ckpt"]})
    parser.add_argument("--gflownet_ckpt_period", default=None, type=int)
    args2config.update({"gflownet_ckpt_period": ["gflownet", "ckpt_period"]})
    parser.add_argument("--gflownet_progress", action="store_true")
    args2config.update({"gflownet_progress": ["gflownet", "progress"]})
    parser.add_argument(
        "--gflownet_loss",
        default="flowmatch",
        type=str,
        help="flowmatch | trajectorybalance/tb",
    )
    args2config.update({"gflownet_loss": ["gflownet", "loss"]})
    parser.add_argument(
        "--gflownet_lr_z_mult",
        default=10,
        type=int,
        help="Multiplicative factor of the Z learning rate",
    )
    args2config.update({"gflownet_lr_z_mult": ["gflownet", "lr_z_mult"]})
    parser.add_argument(
        "--gflownet_learning_rate", default=1e-4, help="Learning rate", type=float
    )
    args2config.update({"gflownet_learning_rate": ["gflownet", "learning_rate"]})
    parser.add_argument("--gflownet_min_word_len", default=1, type=int)
    args2config.update({"gflownet_min_word_len": ["gflownet", "min_word_len"]})
    parser.add_argument("--gflownet_max_word_len", default=1, type=int)
    args2config.update({"gflownet_max_word_len": ["gflownet", "max_word_len"]})
    parser.add_argument("--gflownet_opt", default="adam", type=str)
    args2config.update({"gflownet_opt": ["gflownet", "opt"]})
    parser.add_argument(
        "--reward_beta_init",
        default=1,
        type=float,
        help="Initial beta for exponential reward scaling",
    )
    args2config.update({"reward_beta_init": ["gflownet", "reward_beta_init"]})
    parser.add_argument(
        "--reward_max",
        default=1e6,
        type=float,
        help="Max reward to prevent numerical issues",
    )
    args2config.update({"reward_max": ["gflownet", "reward_max"]})
    parser.add_argument(
        "--reward_beta_mult",
        default=1.25,
        type=float,
        help="Multiplier for rescaling beta during training",
    )
    args2config.update({"reward_beta_mult": ["gflownet", "reward_beta_mult"]})
    parser.add_argument(
        "--reward_beta_period",
        default=-1,
        type=float,
        help="Period (number of iterations) for beta rescaling",
    )
    args2config.update({"reward_beta_period": ["gflownet", "reward_beta_period"]})
    parser.add_argument(
        "--gflownet_early_stopping",
        default=0.01,
        help="Threshold loss for GFlowNet early stopping",
        type=float,
    )
    args2config.update({"gflownet_early_stopping": ["gflownet", "early_stopping"]})
    parser.add_argument(
        "--gflownet_ema_alpha",
        default=0.5,
        help="alpha coefficient for exponential moving average (early stopping)",
        type=float,
    )
    args2config.update({"gflownet_ema_alpha": ["gflownet", "ema_alpha"]})
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    args2config.update({"adam_beta1": ["gflownet", "adam_beta1"]})
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    args2config.update({"adam_beta2": ["gflownet", "adam_beta2"]})
    parser.add_argument("--gflownet_momentum", default=0.9, type=float)
    args2config.update({"gflownet_momentum": ["gflownet", "momentum"]})
    parser.add_argument(
        "--gflownet_mbsize", default=16, help="Minibatch size", type=int
    )
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
        default=10000,
        help="Sequences to sample from GFLowNet",
    )
    args2config.update({"gflownet_n_samples": ["gflownet", "n_samples"]})
    args2config.update({"batch_reward": ["gflownet", "batch_reward"]})
    parser.add_argument("--bootstrap_tau", default=0.0, type=float)
    args2config.update({"bootstrap_tau": ["gflownet", "bootstrap_tau"]})
    parser.add_argument("--clip_grad_norm", default=0.0, type=float)
    args2config.update({"clip_grad_norm": ["gflownet", "clip_grad_norm"]})
    parser.add_argument("--random_action_prob", default=0.0, type=float)
    args2config.update({"random_action_prob": ["gflownet", "random_action_prob"]})
    parser.add_argument("--gflownet_comet_project", default=None, type=str)
    args2config.update({"gflownet_comet_project": ["gflownet", "comet", "project"]})
    parser.add_argument("--gflownet_no_comet", action="store_true")
    args2config.update({"gflownet_no_comet": ["gflownet", "comet", "skip"]})
    parser.add_argument("--no_log_times", action="store_true")
    args2config.update({"no_log_times": ["gflownet", "no_log_times"]})
    parser.add_argument(
        "--tags_gfn", nargs="*", help="Comet.ml tags", default=[], type=str
    )
    args2config.update({"tags_gfn": ["gflownet", "comet", "tags"]})
    parser.add_argument("--gflownet_annealing", action="store_true")
    args2config.update({"gflownet_annealing": ["gflownet", "annealing"]})
    parser.add_argument(
        "--gflownet_annealing_samples",
        type=int,
        default=1000,
        help="number of init configs for post sample annealing",
    )
    args2config.update(
        {"gflownet_annealing_samples": ["gflownet", "post_annealing_samples"]}
    )
    parser.add_argument(
        "--gflownet_post_annealing_time",
        type=int,
        default=1000,
        help="number MCMC steps for post sample annealing",
    )
    args2config.update(
        {"gflownet_post_annealing_time": ["gflownet", "post_annealing_time"]}
    )
    parser.add_argument("--gflownet_test_period", default=500, type=int)
    args2config.update({"gflownet_test_period": ["gflownet", "test", "period"]})
    parser.add_argument("--gflownet_pct_test", default=500, type=int)
    args2config.update({"gflownet_pct_test": ["gflownet", "test", "pct_test"]})
    # Proxy model
    parser.add_argument(
        "--proxy_model_type",
        type=str,
        default="mlp",
        help="type of proxy model - mlp or transformer",
    )
    args2config.update({"proxy_model_type": ["proxy", "model_type"]})
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
    parser.add_argument("--proxy_uncertainty_estimation", type=str, default="dropout", help="dropout or ensemble")
    args2config.update({"proxy_uncertainty_estimation": ["proxy", "uncertainty_estimation"]})
    parser.add_argument("--proxy_dropout", type=float, default = 0.1)
    args2config.update({"proxy_dropout": ["proxy", "dropout"]})
    parser.add_argument("--proxy_dropout_samples", type=int, default = 25, help="number of times to resample via stochastic dropout")
    args2config.update({"proxy_dropout_samples": ["proxy", "dropout_samples"]})
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
        config.proxy.mbsize = 10  # model training batch size
        config.dataset.min_length, config.dataset.max_length = [
            10,
            20,
        ]
        config.dataset.dict_size = 4
    # GFlowNet
    config.gflownet.max_seq_length = config.dataset.max_length
    config.gflownet.min_seq_length = config.dataset.min_length
    config.gflownet.nalphabet = config.dataset.dict_size
    config.gflownet.func = config.dataset.oracle
    config.gflownet.test.score = config.gflownet.func.replace("nupack ", "")
    # Comet: same project for AL and GFlowNet
    if config.comet_project:
        config.gflownet.comet.project = config.comet_project
        config.al.comet.project = config.comet_project
    # sampling method - in case we forget to revert ensemble size
    if config.proxy.uncertainty_estimation == "dropout":
        config.proxy.ensemble_size = 1
        print("Ensemble size set to 1 due to dropout uncertainty estimation being 'on'")
    # Paths
    if not config.workdir and config.machine == "cluster":
        config.workdir = "/home/kilgourm/scratch/learnerruns"
    elif not config.workdir and config.machine == "local":
        config.workdir = "/home/mkilgour/learnerruns"  # "C:/Users\mikem\Desktop/activeLearningRuns"  #
    return config


if __name__ == "__main__":
    # Handle command line arguments and configuration
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser, args2config = add_args(parser)
    args = parser.parse_args()
    config = get_config(args, override_args, args2config)
    config = process_config(config)
    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(config).items()]))
    # TODO: save final config in workdir
    al = activeLearner.ActiveLearning(config)
    if config.al.mode == "initalize":
        printRecord("Initialized!")
    elif config.al.mode == "training":
        al.runPipeline()
    elif config.al.mode == "deploy":
        al.runPipeline()
    elif config.al.mode == "test_rl":
        al.agent.train_from_file()
    elif config.al.mode == "evaluation":
        ValueError(
            "No function for this! Write a function to load torch models and evaluate inputs."
        )
