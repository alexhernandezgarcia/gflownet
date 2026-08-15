#!/usr/bin/env python

import sys
import math
import optparse
import pickle as pickle
import numpy as np
from scipy.special import gammaln, digamma
from scipy.special import gdtrc         # required only for regression
from scipy.optimize import fsolve       # required only for regression
from copy import copy
import matplotlib.pyplot as plt
import scipy.stats
from .utils import hist_count, logsumexp, softmax, sample_multinomial, \
        sample_multinomial_scores, empty, assert_no_nan


def parser_add_common_options():
    parser = optparse.OptionParser()
    parser.add_option('--dataset', dest='dataset', default='toy',
            help='name of the dataset  [default: %default]')
    parser.add_option('--optype', dest='optype', default='class',
            help='nature of outputs in your dataset (class/real)'\
            'for (classification/regression)  [default: %default]')
    parser.add_option('--data_path', dest='data_path', default='../../process_data/',
            help='path of the dataset')
    parser.add_option('--debug', dest='debug', default='0', type='int',
            help='debug or not? (1=True/0=False)')
    parser.add_option('--op_dir', dest='op_dir', default='.', 
            help='output directory for pickle files (NOTE: make sure directory exists) [default: %default]')
    parser.add_option('--tag', dest='tag', default='', 
            help='additional tag to identify results from a particular run')
    parser.add_option('--save', dest='save', default=0, type='int',
            help='do you wish to save the results? (1=True/ 0=False)') 
    parser.add_option('-v', '--verbose',dest='verbose', default=1, type='int',
            help='verbosity level (0 is minimum, 4 is maximum)')
    parser.add_option('--init_id', dest='init_id', default=1, type='int',
            help='init_id (changes random seed for multiple initializations)')
    #
    group = optparse.OptionGroup(parser, "Prior specification / Hyperparameters")
    group.add_option('--prior', dest='prior', default='cgm',
            help='nature of prior (cgm for classification, cgm/bart for regression)')
    group.add_option('--tree_prior', dest='tree_prior', default='cgm',
            help='tree prior that specifies probability of splitting a node'\
            ' (only cgm prior has been implemented till now) [default: %default]')
    group.add_option('--alpha_split', dest='alpha_split', default=0.95, type='float',
            help='alpha-split for cgm tree prior  [default: %default]')   
    group.add_option('--beta_split', dest='beta_split', default=0.5, type='float',
            help='beta_split for cgm tree prior [default: %default]')    
    group.add_option('--alpha', dest='alpha', default=1.0, type='float',
            help='alpha denotes the concentration of dirichlet parameter'\
            ' (NOTE: each of K classes will have mass alpha/K) [default: %default]')
    # kappa_0 < 1 implies that the prior mean can exhibit higher variance around the empirical mean (different means  in different partitions)
    group.add_option('--alpha_0', dest='alpha_0', default=2.0, type='float',
            help='alpha_0 is parameter of Normal-Gamma prior')
    group.add_option('--beta_0', dest='beta_0', default=1.0, type='float',
            help='beta_0 is parameter of Normal-Gamma prior')
    group.add_option('--mu_0', dest='mu_0', default=0.0, type='float',
            help='mu_0 is parameter of Normal-Gamma prior')
    group.add_option('--kappa_0', dest='kappa_0', default=0.3, type='float',   
            help='kappa_0 is parameter of Normal-Gamma prior')
    group.add_option('--alpha_bart', dest='alpha_bart', default=3.0, type='float',
            help='alpha_bart is the df parameter in BART')  # they try just 3 and 10
    group.add_option('--k_bart', dest='k_bart', default=2.0, type='float',
            help='k_bart controls the prior over mu (mu_prec) in BART')
    group.add_option('--q_bart', dest='q_bart', default=0.9, type='float',
            help='q_bart controls the prior over sigma^2 in BART')
    parser.add_option_group(group)
    return parser


def parser_add_smc_options(parser):
    group = optparse.OptionGroup(parser, "SMC options")
    group.add_option('--n_particles', dest='n_particles', default=10, type='int',
            help='number of particles')
    group.add_option('--n_islands', dest='n_islands', default=1, type='int',
            help='number of islands')
    group.add_option('--ess_threshold', dest='ess_threshold', default=0.1, type='float',
            help='ess_threshold [default: %default]')
    group.add_option('--max_iterations', dest='max_iterations', default=5000, type='int',
            help='max iterations of tree building [default: %default]')
    group.add_option('--resample', dest='resample', default='multinomial',
            help='resampling method (multinomial/systematic) [default: %default]')
    group.add_option('--grow', dest='grow', default='next',
            help='expansion procedure (layer=layerwise/next=nodewise) [default: %default]')
    group.add_option('--proposal', dest='proposal', default='prior',
            help='proposal (prior/empirical/posterior[=one-step optimal]) [default: %default]')
    group.add_option('--demo', dest='demo', default=0, type='int',
            help='do you wish to see a demo? you might not be lucky always ;) (1/0)') 
    group.add_option('--priority', dest='priority', default='breadthwise', 
            help='priority = loglik or breadthwise?')
    group.add_option('--weight_predictions', dest='weight_predictions', default=1, type='int',
            help='do you want to weight predictions? (1/0)')
    group.add_option('--weight_islands', dest='weight_islands', default=0, type='int',
            help='do you want to weight islands? (1/0)')
    group.add_option('--temper_factor', dest='temper_factor', default=1.0, type='float',
            help='fraction for tempering the loglikelihood factor? float within [0,1]')
    group.add_option('--include_child_prob', dest='include_child_prob', default=0, type='int',
            help='do you want to include probability of children in the weights? (1/0)')
    group.add_option('--store_history', dest='store_history', default=0, type='int',
            help='do you want to store history of all the particles (required for constructing ancestry plots)? (1/0)')
    parser.add_option_group(group)
    #
    # I have tested the code only with the default values below, 
    # but I left them in incase you want to modify the code later (remember to update filename for saving)
    group = optparse.OptionGroup(parser, "Not-so-optional options",
                    "I have tested the code only with the default values for these options, " 
                    "but I left them incase you want to modify the code")
    group.add_option('--min_size', dest='min_size', default=1, type='int',
            help='minimum number of data points at leaf nodes')
    group.add_option('--frac_features', dest='frac_features', default=1.0, type='float',
            help='fraction of features used at each grow step')
    group.add_option('--frac_splitpoints', dest='frac_splitpoints', default=1.0, type='float',
            help='fraction of split points used at each grow step')
    group.add_option('--choose_greedy', dest='choose_greedy', default=0, type='int',
            help='choose feature (NOT feature-threshold pair) greedily (1/0)')
    parser.add_option_group(group)
    return parser


def parser_check_common_options(parser, settings):
    fail(parser, not(settings.save==0 or settings.save==1), 'save needs to be 0/1')
    fail(parser, not(settings.optype=='real' or settings.optype=='class'), 'optype needs to be real/class')
    fail(parser, not(settings.prior=='cgm' or settings.prior=='bart'), 'prior needs to be cgm/bart')
    fail(parser, not(settings.tree_prior=='cgm'), 'tree_prior needs to be cgm (mondrian yet to be implemented)')


def parser_check_smc_options(parser, settings):
    fail(parser, settings.n_particles < 1, 'number of particles needs to be > 1')
    fail(parser, settings.n_islands > settings.n_particles, 'n_islands needs to be < n_particles')
    fail(parser, settings.n_particles % settings.n_islands != 0, 'n_islands should be a divisor of n_particles')
    fail(parser, settings.max_iterations < 1, 'max_iterations needs to be > 1')
    fail(parser, settings.min_size < 1, 'min_size needs to be > 1')
    fail(parser, (settings.resample != 'multinomial') and (settings.resample != 'systematic'), 
            'unknown resample (valid = multinomial/systematic)')
    fail(parser, (settings.grow != 'layer') and (settings.grow != 'next'), 
            'unknown grow option (valid = layer/next)')
    fail(parser, (settings.proposal != 'posterior') and (settings.proposal != 'prior') \
            and (settings.proposal != 'empirical'), \
            'unknown proposal (valid = posterior/prior/empirical)')
    fail(parser, abs(settings.frac_features - 1) > 1e-12, 'Only frac_features = 1 has been implemented till now') 
    fail(parser, abs(settings.frac_splitpoints - 1) > 1e-12, 'Only frac_splitpoints = 1 has been implemented till now') 
    fail(parser, not(settings.choose_greedy==0), 'choose_greedy needs to be 0 (==1 has not been implemented)')
    fail(parser, not(settings.weight_predictions==0 or settings.weight_predictions==1), 'weight_predictions needs to be 0/1')
    fail(parser, not(settings.weight_islands==0 or settings.weight_islands==1), 'weight_islands needs to be 0/1')
    fail(parser, not(settings.include_child_prob==0 or settings.include_child_prob==1), 'include_child_prob needs to be 0/1')


def fail(parser, condition, msg):
    if condition:
        print(msg)
        print()
        parser.print_help()
        sys.exit(1)


def check_dataset(settings):
    classification_datasets = set(['bc-wisc', 'spambase', 'letter-recognition', 'shuttle', 'covtype', \
            'pendigits', 'arcene', 'gisette', 'madelon', 'iris', 'magic04', 'glass'])
    regression_datasets = set(['toy-reg'])
    if not (settings.dataset[:3] == 'toy' or settings.dataset[:4] == 'test'):
        try:
            if settings.optype == 'class':
                assert(settings.dataset in classification_datasets)
            else:
                assert(settings.dataset in regression_datasets)
        except AssertionError:
            print('Invalid dataset for optype; dataset = %s, optype = %s' % \
                    (settings.dataset, settings.optype))
            raise AssertionError


def load_data(settings):
    data = {}
    check_dataset(settings)
    if (settings.dataset == 'bc-wisc') or (settings.dataset == 'spambase') \
            or (settings.dataset == 'letter-recognition') or (settings.dataset == 'shuttle') \
            or (settings.dataset == 'covtype') or (settings.dataset == 'pendigits') \
            or (settings.dataset == 'arcene') or (settings.dataset == 'gisette') \
            or (settings.dataset == 'madelon') \
            or (settings.dataset == 'iris') or (settings.dataset == 'magic04'):
        data = pickle.load(open(settings.data_path + settings.dataset + '/' + settings.dataset + '.p', \
            "rb"))
    elif settings.dataset == 'toy':
        data = load_toy_data()
    elif settings.dataset == 'toy-small':
        data = load_toy_small_data()
    elif settings.dataset == 'toy2':
        data = load_toy_data2()
    elif settings.dataset == 'toy-spam':
        data = load_toy_spam_data(20)
    elif settings.dataset[:8] == 'toy-spam':
        n_points = int(settings.dataset[9:])
        data = load_toy_spam_data(n_points)
    elif settings.dataset == 'toy-reg':
        data = load_toy_reg()
    elif settings.dataset == 'test-1':
        data = load_test_dataset_1()
    else:
        print('Unknown dataset: ' + settings.dataset)
        raise Exception
    assert(not data['is_sparse'])
    return data


def load_toy_data2():       
    """ easier than toy_data: 1 d marginal gives away the best split """
    n_dim = 2
    n_train_pc = 20
    n_class = 2
    n_train = n_train_pc * n_class
    n_test = n_train
    y_train = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    y_test = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    mag = 5
    x_train = np.random.randn(n_train, n_dim)
    x_test = np.random.randn(n_train, n_dim)
    for i, y_ in enumerate(y_train):
        x_train[i, :] += (2 * y_ - 1) * mag
    for i, y_ in enumerate(y_test):
        x_test[i, :] += (2 * y_ - 1) * mag
    x_train = np.round(x_train)
    x_test = np.round(x_test)
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data


def load_toy_small_data():
    n_dim = 2 
    n_train_pc = 2
    n_class = 2
    n_train = n_train_pc * n_class
    n_test = n_train
    y_train = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    y_test = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    rand_mag = 1.
    x_train = rand_mag * np.random.randn(n_train, n_dim)
    x_test = rand_mag * np.random.randn(n_train, n_dim)
    mag = 5
    for i, y_ in enumerate(y_train):
        if y_ == 0:
            x_train[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_train[i, :] += np.array([tmp, -tmp]) * mag
    for i, y_ in enumerate(y_test):
        if y_ == 0:
            x_test[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_test[i, :] += np.array([tmp, -tmp]) * mag
    if True:
        x_train = np.array(([-5,5],[5,-5],[-8,-8],[8,8]))
        x_test = np.array(([-4,4],[4,-4],[-7,-7],[7,7]))
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data


def load_toy_data():
    n_dim = 2
    n_train_pc = 20
    n_class = 2
    n_train = n_train_pc * n_class
    n_test = n_train
    y_train = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    y_test = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    x_train = np.random.randn(n_train, n_dim)
    x_test = np.random.randn(n_train, n_dim)
    mag = 5
    for i, y_ in enumerate(y_train):
        if y_ == 0:
            x_train[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_train[i, :] += np.array([tmp, -tmp]) * mag
    for i, y_ in enumerate(y_test):
        if y_ == 0:
            x_test[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_test[i, :] += np.array([tmp, -tmp]) * mag
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data
 

def get_tree_limits(p, data):
    d_extent = {}
    x = data['x_train']
    x1_min = np.min(x[:, 1])
    x1_max = np.max(x[:, 1])
    x0_min = np.min(x[:, 0])
    x0_max = np.max(x[:, 0])
    d_extent[0] = (x0_min, x0_max, x1_min, x1_max)
    if not p.node_info:
        return ([], [])
    non_leaf_max = max(p.node_info.keys())
    p_d = p.node_info
    hlines_list = []
    vlines_list = []
    for node_id in range(non_leaf_max + 1):
        if node_id not in p.node_info:
            continue
        x0_min, x0_max, x1_min, x1_max = d_extent[node_id]
        print(p_d[node_id])
        feat_id, split, idx_split_global = p_d[node_id]
        if feat_id == 0:
            vlines_list.append([split, x1_min, x1_max])
            left_extent = (x0_min, split, x1_min, x1_max)
            right_extent = (split, x0_max, x1_min, x1_max)
        else:
            hlines_list.append([split, x0_min, x0_max])
            left_extent = (x0_min, x0_max, x1_min, split)
            right_extent = (x0_min, x0_max, split, x1_max)
        left, right = get_children_id(node_id)
        d_extent[left] = left_extent
        d_extent[right] = right_extent
    return (hlines_list, vlines_list)
            

def plot_particles(particles, data, prob, title_text):
    plt.figure(0)
    plt.clf()
    plt.hold(True)
    n_p = len(particles) 
    if n_p > 9:
        print('WARNING: number of particles > 9 in plot_particles ... printing only the first 9 particles')
    for idx_fig, p in enumerate(particles):
        if n_p <= 4:
            plt.subplot(2,2,idx_fig+1)
        else:
            plt.subplot(3,3,idx_fig+1)
        plt.hold(True)
        idx = data['y_train']==0
        plt.scatter(data['x_train'][idx,0], data['x_train'][idx,1],c='r')
        idx = data['y_train']==1
        plt.scatter(data['x_train'][idx,0], data['x_train'][idx,1],c='b')
        (hlines_list, vlines_list) = get_tree_limits(p, data)
        for hlines in hlines_list:
            split, min_val, max_val = hlines
            plt.hlines(split, min_val, max_val, color='k', linestyle='-', linewidth=2)
        for vlines in vlines_list:
            (split, min_val, max_val) = vlines
            plt.vlines(split, min_val, max_val, color='k', linestyle='-', linewidth=2)
        plt.title(prob[idx_fig])
    plt.figure(0)
    plt.suptitle(title_text)
    plt.draw()
    plt.show()


def load_test_dataset_1():
    n_dim = 2 
    n_train_pc = 1
    n_class = 2
    n_train = n_train_pc * n_class
    n_test = n_train
    y_train = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    y_test = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    x_train = np.random.randn(n_train, n_dim)
    x_test = np.random.randn(n_train, n_dim)
    mag = 5
    for i, y_ in enumerate(y_train):
        if y_ == 0:
            x_train[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_train[i, :] += np.array([tmp, -tmp]) * mag
    for i, y_ in enumerate(y_test):
        if y_ == 0:
            x_test[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_test[i, :] += np.array([tmp, -tmp]) * mag
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    print(data)
    return data
   

def gen_chipman_reg(n_points):
    n_dim = 2
    x = np.zeros((n_points, n_dim))
    y = np.zeros(n_points)
    f = np.zeros(n_points)
    x[:, 0] = [math.ceil(x_) for x_ in np.random.rand(n_points)*10]
    x[:, 1] = [math.ceil(x_) for x_ in np.random.rand(n_points)*4]
    for i, x_ in enumerate(x):
        if x_[1] <= 4.0:
            if x_[0] <= 5.0:
                f[i] = 8.0
            else:
                f[i] = 2.0
        else:
            if x_[0] <= 3.0:
                f[i] = 1.0
            elif x_[0] > 7.0:
                f[i] = 8.0
            else:
                f[i] = 5.0
    y = f + 0.02 * np.random.randn(n_points)
    return (x, y)


def load_toy_reg():
    n_dim = 2
    n_train = 200
    n_test = n_train
    x_train, y_train = gen_chipman_reg(n_train)
    x_test, y_test = gen_chipman_reg(n_test)
    data = {'x_train': x_train, 'y_train': y_train, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data


def load_toy_spam_data(dim):
    n_dim = dim
    n_dim_rel = 2   # number of relevant dimensions
    n_train_pc = 100    # 20
    n_class = 2
    n_train = n_train_pc * n_class
    n_test = n_train
    y_train = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    y_test = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    x_train = np.random.randn(n_train, n_dim)
    x_test = np.random.randn(n_train, n_dim)
    mag = 3 
    for i, y_ in enumerate(y_train):
        if y_ == 0:
            x_train[i, :n_dim_rel] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_train[i, :n_dim_rel] += np.array([tmp, -tmp]) * mag
    for i, y_ in enumerate(y_test):
        if y_ == 0:
            x_test[i, :n_dim_rel] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_test[i, :n_dim_rel] += np.array([tmp, -tmp]) * mag
    data = {'x_train': x_train, 'y_train': y_train, 'n_class': n_class, \
            'n_dim': n_dim, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'is_sparse': False}
    return data


def get_node_list(depth):
    if depth == 0:
        op = [0]
    else:
        op = [2 ** depth - 1 + x for x in range(2 ** depth)]
    return op


def get_parent_id(node_id):
    if node_id == 0:
        op = 0
    else:
        op = int(math.ceil(node_id / 2.) - 1)
    return op


def get_sibling_id(node_id):
    if node_id == 0:
        op = 0
    else:
        parent = get_parent_id(node_id)
        left, right = get_children_id(parent)
        if left == node_id:
            op = right
        else:
            op = left
    return op


def get_depth(node_id):
    op = int(math.floor(math.log(node_id + 1, 2)))
    return op


def get_children_id(node_id):
    tmp = 2 * (node_id + 1)
    return (tmp - 1, tmp)


class Param(object):
    def __init__(self, alpha_split, beta_split):
        self.alpha_split = alpha_split
        self.beta_split = beta_split


def get_filename_smc(settings):
    if settings.optype == 'class':
        param_str = '%s' % settings.alpha
    else:
        raise Exception
    if settings.tree_prior == 'cgm':
        split_str = 'cgm-%s_%s' % (settings.alpha_split, settings.beta_split)
    else:
        raise Exception
    filename = settings.op_dir + '/' + '%s-tree_prior-%s-param-%s-max_iter-%s' \
            '-init_id-%s-smc-C-%s-ess-%s'\
            '-%s-%s-%s-w-%s-tag-%s.p' % \
            (settings.dataset, split_str, param_str, settings.max_iterations,\
             settings.init_id, settings.n_particles, settings.ess_threshold, \
             settings.proposal, settings.resample, settings.grow, settings.weight_predictions, settings.tag)
    return filename


def get_filename_smc_mcmc(settings):
    if settings.optype == 'class':
        param_str = '%s' % settings.alpha
    else:
        raise Exception
    if settings.tree_prior == 'cgm':
        split_str = 'cgm-%s_%s' % (settings.alpha_split, settings.beta_split)
    else:
        raise Exception
    filename = settings.op_dir + '/' + '%s-tree_prior-%s-param-%s-max_iter-%s' \
            '-init_id-%s-smc-C-%s-ess-%s'\
            '-%s-%s-%s-mcmc-%s-%s-tag-%s.p' % \
            (settings.dataset, split_str, param_str, settings.max_iterations,\
             settings.init_id, settings.n_particles, settings.ess_threshold, \
             settings.proposal, settings.resample, settings.grow, settings.mcmc_type, \
             settings.n_iterations, settings.tag)
    return filename


def get_filename_mcmc(settings):
    if settings.optype == 'class':
        param_str = '%s' % settings.alpha
    else:
        raise Exception
    if settings.tree_prior == 'cgm':
        split_str = 'cgm-%s_%s' % (settings.alpha_split, settings.beta_split)
    else:
        raise Exception
    filename = settings.op_dir + '/' + '%s-tree_prior-%s-param-%s-max_iter-%s' \
            '-init_id-%s-mcmc-%s-sample_y-%d-tag-%s.p' % \
            (settings.dataset, split_str, param_str, settings.n_iterations,\
             settings.init_id, settings.mcmc_type, settings.sample_y, settings.tag)
    return filename


class Tree(object):
    def __init__(self, train_ids=[], param=empty(), settings=empty(), cache_tmp={}):
        self.depth = -1
        if cache_tmp:
            self.leaf_nodes = [0]
            self.non_leaf_nodes = []
            self.do_not_split = {0:False}
            if settings.optype == 'class':
                self.counts = {0: cache_tmp['y_train_counts'][:]}
                self.loglik = {0: self.compute_loglik_node(0, settings, param)}
            else:
                self.sum_y = {0: cache_tmp['sum_y']}
                self.sum_y2 = {0: cache_tmp['sum_y2']}
                self.n_points = {0: cache_tmp['n_points']}
                op_tmp, param_tmp = compute_normal_normalizer(self.sum_y[0], self.sum_y2[0], \
                        self.n_points[0], param, cache_tmp, settings)
                self.loglik = {0: op_tmp}
                self.param_n = {0: param_tmp}  
            self.train_ids = {0: train_ids[:]}
            self.node_info = {}
            self.logprior = {0: np.log(self.compute_pnosplit(0, param))}
            self.loglik_current = self.loglik[0] + 0.0

    def precomputed_proposal(self, data, param, settings, cache, node_id, \
            train_ids, log_psplit):
        # code is a bit convoluted ... however, it's faster since it minimizes the number of tests ... 
        # naive version calls find_valid_dimensions too many times unnecessarily
        n_train_ids = len(train_ids)
        pnosplit = self.compute_pnosplit(node_id, param)    # pnosplit will be verified later
        do_not_split_node_id = np.random.rand(1) <= pnosplit
        split_not_supported = False
        if not do_not_split_node_id:
            feat_id_valid, score_feat, feat_split_info, split_not_supported = self.find_valid_dimensions(data, cache, train_ids, settings)
            if split_not_supported:
                do_not_split_node_id = True
                pnosplit = 1.0
            else:
                feat_id_perm, n_feat_subset, log_prob_feat = subsample_features(settings, feat_id_valid, score_feat, split_not_supported)
                feat_id_chosen = sample_multinomial_scores(score_feat)   
                idx_min, idx_max, x_min, x_max, feat_score_cumsum_prior_current = feat_split_info[feat_id_chosen] 
                feat_score_cumsum_current = cache['feat_score_cumsum'][feat_id_chosen] 
                z = np.float64(feat_score_cumsum_current[idx_max] - feat_score_cumsum_current[idx_min])
                prob_split = np.diff(feat_score_cumsum_current[idx_min: idx_max+1] - \
                            feat_score_cumsum_current[idx_min]) / z
                z_prior = feat_score_cumsum_prior_current[idx_max] - feat_score_cumsum_prior_current[idx_min]
                prob_split_prior = np.diff(feat_score_cumsum_prior_current[idx_min: idx_max+1] - \
                            feat_score_cumsum_prior_current[idx_min]) / z_prior
                if settings.debug == 1 and settings.proposal == 'prior':
                    assert np.abs(np.sum(prob_split) - 1) < 1e-3
                    assert np.abs(np.sum(prob_split_prior) - 1) < 1e-3
                    assert np.abs(np.sum(np.abs(prob_split - prob_split_prior))) < 1e-3
                idx_split_chosen = sample_multinomial(prob_split)
                idx_split_global = idx_split_chosen + idx_min + 1
                split_chosen = cache['feat_idx2midpoint'][feat_id_chosen][idx_split_global]
                if settings.debug == 1:
                    is_split_valid(split_chosen, x_min, x_max)
                logprior_nodeid_tau = np.log(prob_split_prior[idx_split_chosen])
                logprior_nodeid = log_psplit + logprior_nodeid_tau \
                                    + log_prob_feat[feat_id_chosen]
                (train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) = \
                    compute_left_right_statistics(data, param, cache, train_ids, feat_id_chosen, split_chosen, settings)
                log_sis_ratio = loglik_left + loglik_right - self.loglik[node_id] \
                                + logprior_nodeid_tau \
                                - np.log(prob_split[idx_split_chosen])
                if settings.include_child_prob:
                    left, right = get_children_id(node_id)
                    log_sis_ratio += np.log(self.compute_pnosplit(left, param)) + np.log(self.compute_pnosplit(right, param))
                # contributions of feat_id and psplit cancel out for precomputed proposals
                if settings.verbose >= 2:
                    print('idx_split_chosen = %d, split_chosen = %f' % (idx_split_chosen, split_chosen))
                    print('feat_id_chosen = %f' % (feat_id_chosen))
        if do_not_split_node_id:
            feat_id_chosen = -1
            split_chosen = 3.14
            idx_split_global = -1
            if (not split_not_supported) and (no_valid_split_exists(data, cache, train_ids, settings)):
                # re-check logprior_nodeid contribution if you sampled do_not_split_node_id=True initially
                logprior_nodeid = 0.0
            else:
                # valid split exists (original sampling was correct) or we have reset pnosplit according to split_not_supported
                logprior_nodeid = np.log(pnosplit)
            log_sis_ratio = 0.0     # probability of not splitting under prior and proposal are both the same
            (train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) = \
                init_left_right_statistics()
        return (do_not_split_node_id, feat_id_chosen, split_chosen, idx_split_global, log_sis_ratio, logprior_nodeid, \
            train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right)

    def update_left_right_statistics(self, cache_tmp, node_id, logprior_nodeid, train_ids_left,\
            train_ids_right, loglik_left, loglik_right, feat_id_chosen, split_chosen, idx_split_global, \
            settings, param, data, cache):
        left, right = get_children_id(node_id)
        self.logprior[node_id] = logprior_nodeid
        self.node_info[node_id] = [feat_id_chosen, split_chosen, idx_split_global]
        self.loglik[left] = loglik_left
        self.loglik[right] = loglik_right
        self.do_not_split[left] = stop_split(train_ids_left, settings, data, cache)
        self.do_not_split[right] = stop_split(train_ids_right, settings, data, cache)
        if self.do_not_split[left]:
            self.logprior[left] = 0.0
        else:
            self.logprior[left] = np.log(self.compute_pnosplit(left, param))
        if self.do_not_split[right]:
            self.logprior[right] = 0.0
        else:
            self.logprior[right] = np.log(self.compute_pnosplit(right, param))
        if settings.debug == 1:
            if settings.optype == 'class':
                assert(np.sum(cache_tmp['cnt_left_chosen']) != 0)
                assert(np.sum(cache_tmp['cnt_right_chosen']) != 0)
            else:
                assert(cache_tmp['n_points_left'] > 0)
                assert(cache_tmp['n_points_right'] > 0)
        self.train_ids[left] = train_ids_left
        self.leaf_nodes.append(left)
        self.train_ids[right] = train_ids_right
        self.leaf_nodes.append(right)
        if settings.optype == 'class':
            self.counts[left] = cache_tmp['cnt_left_chosen']
            self.counts[right] = cache_tmp['cnt_right_chosen']
        else:
            self.sum_y[left] = cache_tmp['sum_y_left']
            self.sum_y2[left] = cache_tmp['sum_y2_left']
            self.n_points[left] = cache_tmp['n_points_left']
            self.param_n[left] = cache_tmp['param_left']
            self.sum_y[right] = cache_tmp['sum_y_right']
            self.sum_y2[right] = cache_tmp['sum_y2_right']
            self.n_points[right] = cache_tmp['n_points_right']
            self.param_n[right] = cache_tmp['param_right']
        self.leaf_nodes.remove(node_id)
        self.non_leaf_nodes.append(node_id)
        self.depth = max(get_depth(left), self.depth)

    def remove_leaf_node_statistics(self, node_id, settings):
        try:
            self.leaf_nodes.remove(node_id)
        except:
            print('%s is not a leaf node' % node_id)
            raise Exception
        self.loglik.pop(node_id)
        self.train_ids.pop(node_id)
        self.logprior.pop(node_id)
        if settings.optype == 'class':
            self.counts.pop(node_id)
        else:
            self.sum_y.pop(node_id)
            self.sum_y2.pop(node_id)
            self.n_points.pop(node_id)
            self.param_n.pop(node_id)

    def sample_split_prior(self, data, param, settings, cache, node_id):
        train_ids = self.train_ids[node_id]
        n_train_ids = len(train_ids)
        log_psplit = np.log(self.compute_psplit(node_id, param))
        pnosplit = self.compute_pnosplit(node_id, param)    # pnosplit will be verified later
        feat_id_valid, score_feat, feat_split_info, split_not_supported \
                    = self.find_valid_dimensions(data, cache, train_ids, settings)
        if split_not_supported:
            do_not_split_node_id = True
            feat_id_chosen = -1
            split_chosen = 3.14
            idx_split_global = -1
            logprior_nodeid = 0.0
        else: 
            do_not_split_node_id = False
            feat_id_perm, n_feat_subset, log_prob_feat = \
                    subsample_features(settings, feat_id_valid, score_feat, split_not_supported)
            feat_id_chosen = sample_multinomial_scores(score_feat)   
            idx_min, idx_max, x_min, x_max, feat_score_cumsum_prior_current = \
                    feat_split_info[feat_id_chosen] 
            z_prior = feat_score_cumsum_prior_current[idx_max] - \
                    feat_score_cumsum_prior_current[idx_min]
            prob_split_prior = np.diff(feat_score_cumsum_prior_current[idx_min: idx_max+1] - \
                        feat_score_cumsum_prior_current[idx_min]) / z_prior
            idx_split_chosen = sample_multinomial(prob_split_prior)
            idx_split_global = idx_split_chosen + idx_min + 1
            split_chosen = cache['feat_idx2midpoint'][feat_id_chosen][idx_split_global]
            if settings.debug == 1:
                is_split_valid(split_chosen, x_min, x_max)
            logprior_nodeid_tau = np.log(prob_split_prior[idx_split_chosen])
            logprior_nodeid = log_psplit + logprior_nodeid_tau \
                                + log_prob_feat[feat_id_chosen]
            if settings.verbose >= 2:
                print('idx_split_chosen = %d, split_chosen = %f' % (idx_split_chosen, split_chosen))
                print('feat_id_chosen = %f' % (feat_id_chosen))
            if settings.verbose >= 3:
                print('3 terms in sample_split_prior for node_id = %s; %s, %s, %s' \
                         % (node_id, log_psplit, logprior_nodeid_tau, log_prob_feat[feat_id_chosen]))
                print('feat_id = %s, idx_split_chosen = %d, split_chosen = %f' % (feat_id_chosen, idx_split_chosen, split_chosen))
                print('log prob_split_prior = %s' % np.log(prob_split_prior))
                print()
        return (do_not_split_node_id, feat_id_chosen, split_chosen, idx_split_global, logprior_nodeid)
    
    def create_prediction_tree(self, param, data, settings):
        if settings.optype == 'class':
            self.pred_prob = {}
        else:
            self.pred_mean = {}
            self.pred_param = {}
        for node_id in self.leaf_nodes:
            if settings.optype == 'class':
                tmp = self.counts[node_id] + float(param.alpha) / data['n_class']
                tmp = tmp / np.float64(np.sum(tmp))
                self.pred_prob[node_id] = tmp
            else:
                if settings.prior == 'cgm':
                    # param_n[node_id] contains the tuple (alpha, beta, mu, kappa)
                    # predictive distribution is a t-distribution; see (110) in [M07]
                    # scale = 1/sqrt(precision), location = mean
                    # can use scipy.stats.t: pdf(x, df, loc=0, scale=1) for 
                    #   prediction (not sure how stable pdf would be a for a continuous distribution)
                    (alpha, beta, mu, kappa) = self.param_n[node_id]
                    self.pred_mean[node_id] = mu
                    df = 2 * alpha
                    #scale = math.sqrt(beta * (kappa + 1) / (alpha * kappa))
                    prec = alpha * kappa / (beta * (kappa + 1))
                    log_const = 0.5 * (np.log(prec) - np.log(2 * alpha * math.pi)) \
                            + gammaln((df + 1) / 2.0) - gammaln(df / 2.0)
                    self.pred_param[node_id] = (mu, df, prec, log_const)
                else:
                    mu = self.param_n[node_id][0]
                    self.pred_mean[node_id] = mu
                    log_const = 0.5 * (np.log(param.lambda_bart) - np.log(2 * math.pi))
                    self.pred_param[node_id] = (mu, param.lambda_bart, log_const)

    def find_valid_dimensions(self, data, cache, train_ids, settings):
        score_feat = cache['prob_feat']
        first_time = True
        if settings.verbose >= 3:
            print('original score_feat = %s' % score_feat)
        feat_split_info = {}
        for feat_id in cache['range_n_dim']:
            x_min = np.min(data['x_train'][train_ids, feat_id])
            x_max = np.max(data['x_train'][train_ids, feat_id])
            idx_min = cache['feat_val2idx'][feat_id][x_min]
            idx_max = cache['feat_val2idx'][feat_id][x_max]
            feat_score_cumsum_prior_current = cache['feat_score_cumsum_prior'][feat_id] 
            if settings.verbose >= 3:
                print('x_min = %s, x_max = %s, idx_min = %s, idx_max = %s' % \
                        (x_min, x_max, idx_min, idx_max))
            if idx_min == idx_max:
                if first_time:          # lazy copy
                    score_feat = cache['prob_feat'].copy()
                    first_time = False
                score_feat[feat_id] = 0
            else:
                feat_split_info[feat_id] = [idx_min, idx_max, x_min, x_max, \
                        feat_score_cumsum_prior_current]
        feat_id_valid = [feat_id for feat_id in cache['range_n_dim'] if score_feat[feat_id] > 0]
        split_not_supported = (len(feat_id_valid) == 0)
        if settings.verbose >= 3:
            print('in find_valid_dimensions now')
            print('training data in current node =\n %s' % data['x_train'][train_ids, :])
            print('score_feat = %s, feat_id_valid = %s' % (score_feat, feat_id_valid))
        return (feat_id_valid, score_feat, feat_split_info, split_not_supported)

    def recompute_prob_split(self, data, param, settings, cache, node_id):
        train_ids = self.train_ids_new[node_id]
        if stop_split(train_ids, settings, data, cache):
            self.logprior_new[node_id] = -np.inf
        else:
            feat_id_chosen, split_chosen, idx_split_global = self.node_info_new[node_id]
            feat_id_valid, score_feat, feat_split_info, split_not_supported \
                        = self.find_valid_dimensions(data, cache, train_ids, settings)
            if feat_id_chosen not in feat_id_valid:
                self.logprior_new[node_id] = -np.inf
            else:
                log_prob_feat = np.log(score_feat) - np.log(np.sum(score_feat))
                idx_min, idx_max, x_min, x_max, feat_score_cumsum_prior_current = \
                        feat_split_info[feat_id_chosen] 
                if (split_chosen <= x_min) or (split_chosen >= x_max):
                    self.logprior_new[node_id] = -np.inf
                else:
                    z_prior = feat_score_cumsum_prior_current[idx_max] - \
                            feat_score_cumsum_prior_current[idx_min]
                    prob_split_prior = np.diff(feat_score_cumsum_prior_current[idx_min: idx_max+1] - \
                                feat_score_cumsum_prior_current[idx_min]) / z_prior
                    idx_split_chosen = idx_split_global - idx_min - 1
                    logprior_nodeid_tau = np.log(prob_split_prior[idx_split_chosen])
                    log_psplit = np.log(self.compute_psplit(node_id, param))
                    self.logprior_new[node_id] = log_psplit + logprior_nodeid_tau \
                                                 + log_prob_feat[feat_id_chosen]
                    if settings.verbose >= 3:
                        print('3 terms in recompute for node_id = %s; %s, %s, %s' \
                               % (node_id, log_psplit, logprior_nodeid_tau, \
                                                     log_prob_feat[feat_id_chosen]))
                        print('feat_id = %s, idx_split_chosen = %d, split_chosen = %f' % (feat_id_chosen, idx_split_chosen, split_chosen))
                        print('log prob_split_prior = %s' % np.log(prob_split_prior))
                        print()
   

    def create_prediction_tree(self, param, data, settings):
        if settings.optype == 'class':
            self.pred_prob = {}
        else:
            self.pred_mean = {}
            self.pred_param = {}
        for node_id in self.leaf_nodes:
            if settings.optype == 'class':
                tmp = self.counts[node_id] + float(param.alpha) / data['n_class']
                tmp = tmp / np.float64(np.sum(tmp))
                self.pred_prob[node_id] = tmp
            else:
                if settings.prior == 'cgm':
                    # param_n[node_id] contains the tuple (alpha, beta, mu, kappa)
                    # predictive distribution is a t-distribution; see (110) in [M07]
                    # scale = 1/sqrt(precision), location = mean
                    # can use scipy.stats.t: pdf(x, df, loc=0, scale=1) for prediction
                    (alpha, beta, mu, kappa) = self.param_n[node_id]
                    self.pred_mean[node_id] = mu
                    df = 2 * alpha
                    #scale = math.sqrt(beta * (kappa + 1) / (alpha * kappa))
                    prec = alpha * kappa / (beta * (kappa + 1))
                    log_const = 0.5 * (np.log(prec) - np.log(2 * alpha * math.pi)) \
                            + gammaln((df + 1) / 2.0) - gammaln(df / 2.0)
                    self.pred_param[node_id] = (mu, df, prec, log_const)
                else:
                    mu = self.param_n[node_id][0]
                    self.pred_mean[node_id] = mu
                    log_const = 0.5 * (np.log(param.lambda_bart) - np.log(2 * math.pi))
                    self.pred_param[node_id] = (mu, param.lambda_bart, log_const)

    def print_tree(self):
        try:
            print('leaf nodes are %s, non-leaf nodes are %s' % (self.leaf_nodes, self.non_leaf_nodes))
            print('logprior = %s, loglik = %s' % (self.logprior, self.loglik))
        except:
            print('leaf nodes are %s' % self.leaf_nodes)
        print('node_id\tdepth\tfeat_id\t\tsplit_point')
        for node_id in self.non_leaf_nodes:
            try:
                feat_id, split, idx_split_global = self.node_info[node_id]
            except (IndexError, ValueError):          # more than 2 values to unpack
                feat_id, split = -1, np.float64('nan')
            print('%3d\t%3d\t%6d\t\t%.2f' % (node_id, get_depth(node_id), \
                    feat_id, split))

    def gen_tree_key(self):
        str_leaf = '_'.join(str(n) for n in sorted(self.leaf_nodes))
        str_nonleaf = '_'.join(str(n) + '-' + str(self.node_info[n][0]) + '-' + str(self.node_info[n][1]) \
                for n in sorted(self.non_leaf_nodes))
        op = str_leaf + '.' + str_nonleaf
        return op

    def traverse(self, x):
        node_id = 0
        while True:
            if node_id in self.leaf_nodes:
                break
            left, right = get_children_id(node_id)
            feat_id, split, idx_split_global = self.node_info[node_id]
            if x[feat_id] <= split:
                node_id = left
            else:
                node_id = right
        return node_id

    def predict_class_fast(self, x_test, n_class, alpha):
    # useful when prediction tree has been contructed
        pred_prob = np.zeros((x_test.shape[0], n_class))
        for n, x in enumerate(x_test):
            pred_prob[n, :] = self.pred_prob[self.traverse(x)]
        return pred_prob
    
    def predict_real_fast(self, x_test, y_test, param, settings):
    # useful when prediction tree has been contructed
        pred_prob = np.zeros(x_test.shape[0])
        pred_mean = np.zeros(x_test.shape[0])
        for n, x in enumerate(x_test):
            node_id = self.traverse(x)
            pred_mean[n] = self.pred_mean[node_id]
            if settings.prior == 'cgm':
                pred_prob[n] = np.exp(compute_t_loglik(y_test[n], self.pred_param[node_id]))
            else:
                pred_prob[n] = np.exp(compute_nn_loglik(y_test[n], self.pred_param[node_id]))
        return (pred_mean, pred_prob)

    def predict(self, x_test, n_class, alpha):
        pred = np.zeros(x_test.shape[0])
        pred_prob = np.zeros((x_test.shape[0], n_class))
        for n, x in enumerate(x_test):
            tmp = self.counts[self.traverse(x)] + float(alpha) / n_class
            pred[n] = np.argmax(tmp)            
            tmp = tmp / np.float64(np.sum(tmp))
            pred_prob[n, :] = tmp
        return (pred, pred_prob)

    def compute_psplit(self, node_id, param):
        return param.alpha_split * math.pow(1 + get_depth(node_id), -param.beta_split)
    
    def compute_pnosplit(self, node_id, param):
        return 1.0 - self.compute_psplit(node_id, param)
    
    def compute_loglik_node(self, node_id, settings, param):
        if settings.optype == 'class':
            op = compute_dirichlet_normalizer(self.counts[node_id], param.alpha)
        else:
            op = compute_ng_normalizer(self.sum_y[node_id], self.sum_y2[node_id], \
                    self.n_points[node_id], param)
        return op
    
    def compute_loglik(self):
        tmp = [self.loglik[node_id] for node_id in self.leaf_nodes]
        return sum(tmp)
    
    def compute_logprior(self):
        tmp = sum([self.logprior[node_id] for node_id in self.leaf_nodes]) \
                + sum([self.logprior[node_id] for node_id in self.non_leaf_nodes])
        return tmp

    def compute_logprob(self):
        return self.compute_loglik() + self.compute_logprior()
    

def compute_test_metrics_classification(y_test, pred_prob):
    acc, log_prob = 0.0, 0.0
    for n, y in enumerate(y_test):
        tmp = pred_prob[n, :]
        pred = np.argmax(tmp)
        acc += (pred == y)
        log_tmp_pred = np.log(tmp[y]) 
        try:
            assert(not np.isinf(abs(log_tmp_pred)))
        except AssertionError:
            'print abs(log_tmp_pred) = inf in compute_test_metrics_classification; tmp = '
            print(tmp)
            raise AssertionError
        log_prob += log_tmp_pred
    acc /= (n + 1)
    log_prob /= (n + 1)
    metrics = {'acc': acc, 'log_prob': log_prob}
    return metrics


def test_compute_test_metrics_classification():
    n = 100
    n_class = 10
    pred_prob = np.random.rand(n, n_class)
    y = np.ones(n)
    metrics = compute_test_metrics_classification(y, pred_prob)
    print('chk if same: %s, %s' % (metrics['log_prob'], np.mean(np.log(pred_prob[:, 1]))))
    assert(np.abs(metrics['log_prob']  - np.mean(np.log(pred_prob[:, 1]))) < 1e-10)
    pred_prob[:, 1] = 1e5
    metrics = compute_test_metrics_classification(y, pred_prob)
    assert np.abs(metrics['acc'] - 1) < 1e-3
    print('chk if same: %s, 1.0' % (metrics['acc']))


def compute_test_metrics_regression(y_test, pred_mean, pred_prob):
    mse, log_prob = 0.0, 0.0
    for n, y in enumerate(y_test):
        mse += math.pow(pred_mean[n] - y, 2)
        log_tmp_pred = np.log(pred_prob[n])     # try weighted log-sum-exp?
        try:
            assert(not np.isinf(abs(log_tmp_pred)))
        except AssertionError:
            'print abs(log_tmp_pred) = inf in compute_test_metrics_regression; tmp = '
            print(tmp)
            raise AssertionError
        log_prob += log_tmp_pred
    mse /= (n + 1)
    log_prob /= (n + 1)
    metrics = {'mse': mse, 'log_prob': log_prob}
    return metrics


def test_compute_test_metrics_regression():
    n = 100
    pred_prob = np.random.rand(n)
    y = np.random.randn(n)
    pred = np.ones(n)
    metrics = compute_test_metrics_regression(y, pred, pred_prob)
    print('chk if same: %s, %s' % (metrics['mse'], np.mean((y - 1) ** 2)))
    print('chk if same: %s, %s' % (metrics['log_prob'], np.mean(np.log(pred_prob))))
    assert np.abs(metrics['mse'] - np.mean((y - 1) ** 2)) < 1e-3
    assert np.abs(metrics['log_prob'] - np.mean(np.log(pred_prob))) < 1e-3


def compute_t_loglik(x, param_t):
    (mu, df, prec, log_const) = param_t
    op = - (df + 1.0) / 2 * np.log(1 + (prec * ((x - mu) ** 2) / df)) + log_const
    return op


def compute_nn_loglik(x, param_nn):
    (mu, prec, log_const) = param_nn
    op = - 0.5 * prec * ((x - mu) ** 2) + log_const
    return op


def is_split_valid(split_chosen, x_min, x_max):
    try:
        assert(split_chosen > x_min)
        assert(split_chosen < x_max)
    except AssertionError:
        print('split_chosen <= x_min or >= x_max')
        raise AssertionError


def evaluate_performance_tree(p, param, data, settings, x_test, y_test):
    p.create_prediction_tree(param, data, settings)
    pred_all = evaluate_predictions_fast(p, x_test, y_test, data, param, settings)
    pred_prob = pred_all['pred_prob']
    if settings.optype == 'class':
        metrics = compute_test_metrics_classification(y_test, pred_prob)
    else:
        pred_mean = pred_all['pred_mean']
        metrics = compute_test_metrics_regression(y_test, pred_mean, pred_prob)
    return (metrics)


def test_compute_t_loglik(alpha=2, beta=1, mu=0, kappa=1):
    df = 2 * alpha
    scale = math.sqrt(beta * (kappa + 1) / (alpha * kappa))
    prec = alpha * kappa / (beta * (kappa + 1))
    log_const = 0.5 * (np.log(prec) - np.log(2 * alpha * math.pi)) + \
            gammaln((df + 1) / 2.0) - gammaln(df / 2.0)
    n = 10
    param_t = (mu, df, prec, log_const)
    print('check if same')
    for x in np.random.randn(n):
        t1 = compute_t_loglik(x, param_t)
        t2 = np.log(scipy.stats.t.pdf(x, df, loc=mu, scale=scale))
        print('%s, %s, %s' % (t1-t2, t1, t2))


def compute_test_metrics(y_test, pred_prob):
    acc, log_prob = 0.0, 0.0
    for n, y in enumerate(y_test):
        tmp = pred_prob[n, :]
        pred = np.argmax(tmp)
        acc += (pred == y_test[n])
        #log_tmp_pred = np.log(tmp[pred]) 
        log_tmp_pred = np.log(tmp[y]) 
        try:
            assert(not np.isinf(abs(log_tmp_pred)))
        except AssertionError:
            'print abs(log_tmp_pred) = inf in compute_test_metrics; tmp = '
            print(tmp)
            raise AssertionError
        log_prob += log_tmp_pred
    acc /= (n + 1)
    log_prob /= (n + 1)
    return (acc, log_prob)


def stop_split(train_ids, settings, data, cache):
    if (len(train_ids) <= settings.min_size):
        op = True
    else:
        op = no_valid_split_exists(data, cache, train_ids, settings)
    return op


def compute_dirichlet_normalizer(cnt, alpha=0.0, prior_term=None):
    """ cnt is np.array, alpha is concentration of Dirichlet prior 
        => alpha/K is the mass for each component of a K-dimensional Dirichlet
    """
    try:
        assert(len(cnt.shape) == 1)
    except AssertionError:
        print('cnt should be a 1-dimensional np array')
        raise AssertionError
    except:
        raise Exception
    n_class = float(len(cnt))
    if prior_term is None:
        #print 'recomputing prior_term'
        prior_term = gammaln(alpha) - n_class * gammaln(alpha / n_class)
    op = np.sum(gammaln(cnt + alpha / n_class)) - gammaln(np.sum(cnt) + alpha) \
            + prior_term
    return op


def compute_ng_normalizer(sum_y, sum_y2, n_points, param, cache):
    y_bar = sum_y / n_points
    kappa = param.kappa_0 + n_points
    alpha = param.alpha_0 + n_points / 2.0
    beta = param.beta_0 + 0.5 * (sum_y2 - n_points * (y_bar ** 2)) \
            + 0.5 * param.kappa_0 / (param.kappa_0 + n_points) * ((y_bar - param.mu_0) ** 2)
    op = cache['ng_prior_term'] - n_points * cache['half_log_2pi'] \
            + gammaln(alpha) - alpha * np.log(beta) - 0.5 * np.log(kappa)
    mu = (param.kappa_0 * param.mu_0 + sum_y) / (param.kappa_0 + n_points)
    return (op, (alpha, beta, mu, kappa))


def test_compute_ng_normalizer(alpha_0=2, beta_0=1, mu_0=0, kappa_0=1, n_points=10000):
    y = np.random.randn(n_points)
    sum_y = np.sum(y)
    sum_y2 = np.sum(y ** 2)
    param = empty()
    param.alpha_0 = alpha_0
    param.beta_0 = beta_0
    param.mu_0 = mu_0
    param.kappa_0 = kappa_0
    cache = {}
    cache['ng_prior_term'] = param.alpha_0 * np.log(param.beta_0) + 0.5 * np.log(param.kappa_0) \
            - gammaln(param.alpha_0)
    cache['half_log_2pi'] = 0.5 * np.log(2 * math.pi)
    print(compute_ng_normalizer(sum_y, sum_y2, n_points, param, cache))


def compute_nn_normalizer(sum_y, sum_y2, n_points, param, cache):
    #y_bar = sum_y / n_points
    tmp = param.lambda_bart * n_points + param.mu_prec
    mu = (param.mu_prec * param.mu_mean + sum_y) / tmp
    op = cache['nn_prior_term'] - n_points * cache['half_log_2pi'] \
            + 0.5 * n_points * np.log(param.lambda_bart) - 0.5 * np.log(tmp) \
            + 0.5 * tmp * mu * mu - 0.5 * param.lambda_bart * sum_y2
    return (op, (mu))


def compute_normal_normalizer(sum_y, sum_y2, n_points, param, cache, settings):
    if settings.prior == 'cgm':
        op, param = compute_ng_normalizer(sum_y, sum_y2, n_points, param, cache)
    else:
        op, param = compute_nn_normalizer(sum_y, sum_y2, n_points, param, cache)
    return (op, param)


def compute_dirichlet_normalizer_fast(cnt, cache):
    """ cnt is np.array, alpha is concentration of Dirichlet prior 
        => alpha/K is the mass for each component of a K-dimensional Dirichlet
    """
    op = compute_gammaln_1(cnt, cache) - compute_gammaln_2(np.sum(cnt), cache) \
            + cache['alpha_prior_term']
    return op


def compute_gammaln_1(mat, cache):
    op = np.sum(cache['lookup_gammaln_1'][mat.astype(int)])
    return op


def compute_gammaln_2(x, cache):
    return cache['lookup_gammaln_2'][int(x)]


def compute_log_pnosplit_children(node_id, param):
    left, right = get_children_id(node_id)
    tmp = np.log(1 - param.alpha_split * math.pow(1 + get_depth(left), -param.beta_split)) \
            +  np.log(1 - param.alpha_split * math.pow(1 + get_depth(right), -param.beta_split))
    return tmp


def evaluate_param(alpha_split, beta_split):
    n = 0
    max_depth = 10
    for d in range(max_depth):
        n += math.pow(2, d) * alpha_split * math.pow(1 + d, -beta_split)
    print('\sum_k psplit(k) (not sure how to interpret this quantity though) = %f' % n)
    print('\sum_k psplit(k) / num_nodes = %f' % (n/(math.pow(2,d+1)-1)))


def evaluate_predictions(p, x, y, data, param):
    (pred, pred_prob) = p.predict(x, data['n_class'], param.alpha)
    (acc, log_prob) = compute_test_metrics(y, pred_prob)
    return (pred, pred_prob, acc, log_prob)


def init_left_right_statistics():
    return(None, None, {}, -np.inf, -np.inf)


def subsample_features(settings, feat_id_valid, score_feat, split_not_supported):
    # TODO: subsampling has not been implemented till now
    feat_id_perm = feat_id_valid
    n_feat_subset = len(feat_id_perm)
    if split_not_supported:
        log_prob_feat = np.ones(score_feat.shape) * np.nan
    else:
        log_prob_feat = np.log(score_feat) - np.log(np.sum(score_feat))
        if (settings.debug == 1) and feat_id_perm:
            try:
                assert(np.abs(logsumexp(log_prob_feat)) < 1e-12)
            except AssertionError:
                print('feat_id_perm = %s' % feat_id_perm)
                print('score_feat = %s' % score_feat)
                print('logsumexp(log_prob_feat) = %s (needs to be 0)' % logsumexp(log_prob_feat))
                raise AssertionError
    return (feat_id_perm, n_feat_subset, log_prob_feat)


def compute_left_right_statistics(data, param, cache, train_ids, feat_id_chosen, split_chosen, settings):
    train_ids_left = [train_id for train_id in train_ids if \
            data['x_train'][train_id, feat_id_chosen] <= split_chosen]
    train_ids_right = [train_id for train_id in train_ids if \
            data['x_train'][train_id, feat_id_chosen] > split_chosen]
    cache_tmp = {}
    if settings.optype == 'class':
        range_n_class = cache['range_n_class']
        cnt_left_chosen = hist_count(data['y_train'][train_ids_left], \
                range_n_class)
        cnt_right_chosen = hist_count(data['y_train'][train_ids_right], \
                range_n_class)
        loglik_left = compute_dirichlet_normalizer_fast(cnt_left_chosen, cache)
        loglik_right = compute_dirichlet_normalizer_fast(cnt_right_chosen, cache)
        cache_tmp['cnt_left_chosen'] = cnt_left_chosen
        cache_tmp['cnt_right_chosen'] = cnt_right_chosen
    else:
        sum_y_left = np.sum(data['y_train'][train_ids_left])
        sum_y2_left = np.sum(data['y_train'][train_ids_left] ** 2)
        n_points_left = len(train_ids_left)
        loglik_left, param_left = compute_normal_normalizer(sum_y_left, sum_y2_left, n_points_left, param, cache, settings)
        cache_tmp['sum_y_left'] = sum_y_left
        cache_tmp['sum_y2_left'] = sum_y2_left
        cache_tmp['n_points_left'] = n_points_left
        cache_tmp['param_left'] = param_left
        sum_y_right = np.sum(data['y_train'][train_ids_right])
        sum_y2_right = np.sum(data['y_train'][train_ids_right] ** 2)
        n_points_right = len(train_ids_right)
        loglik_right, param_right = compute_normal_normalizer(sum_y_right, sum_y2_right, n_points_right, param, cache, settings)
        cache_tmp['sum_y_right'] = sum_y_right
        cache_tmp['sum_y2_right'] = sum_y2_right
        cache_tmp['n_points_right'] = n_points_right
        cache_tmp['param_right'] = param_right
    return(train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right)


def compute_entropy(cnts, alpha=0.0):
    """ returns the entropy of a multinomial distribution with 
        mean parameter \propto (cnts + alpha/len(cnts))
        entropy unit = nats """
    prob = cnts * 1.0 + alpha / len(cnts)
    prob /= float(np.sum(prob))
    entropy = 0.0
    for k in range(len(cnts)):
        if abs(prob[k]) > 1e-12:
            entropy -= prob[k] * np.log(prob[k])
    return entropy


def precompute(data, settings):
    param = Param(settings.alpha_split, settings.beta_split)
    cache_tmp = {}
    if settings.optype == 'class':
        param.alpha = settings.alpha
        cache_tmp['y_train_counts'] = np.bincount(data['y_train'])
    else:
        if settings.prior == 'cgm':
            #param.alpha_0 = settings.alpha_0
            #param.beta_0 = settings.beta_0
            #param.mu_0 = settings.mu_0
            param.kappa_0 = settings.kappa_0    # might as well set it from outside
            param.mu_0 = np.mean(data['y_train'])
            param.alpha_0 = 3   # i like 3 ;-)
            prec = 1.0 / np.var(data['y_train'])
            q_cgm = 0.9
            param.beta_0 = compute_gamma_param(prec, param.alpha_0, q_cgm)
            print('unconditional precision = %s, expected precision under the prior = %s' % (prec, param.alpha_0 / param.beta_0))
            print('gamma prior: param.alpha_0 (shape), param.beta_0 (rate) = %s, %s' % (param.alpha_0, param.beta_0))
            cache_tmp['ng_prior_term'] = param.alpha_0 * np.log(param.beta_0) + 0.5 * np.log(param.kappa_0) - gammaln(param.alpha_0)
            if settings.debug == 1:
                x = np.logspace(-3,2,100)
                log_px = param.alpha_0 * np.log(param.beta_0) + (param.alpha_0 - 1) * np.log(x) \
                        - param.beta_0 * x - gammaln(param.alpha_0)
                plt.semilogx(x, np.exp(log_px))
                plt.hold(True)
                plt.axvline(prec, color='r')
                plt.show()
        else:
            param.mu_mean = np.mean(data['y_train'])
            param.mu_prec = (2 * settings.k_bart / (np.max(data['y_train']) - np.min(data['y_train']))) ** 2
            prec = 1.0 / np.var(data['y_train'])
            param.alpha_bart = settings.alpha_bart
            param.beta_bart = compute_gamma_param(prec, param.alpha_bart, settings.q_bart)
            # ensures that 1-gamcdf(prec; shape=alpha_bart, rate=beta_bart) \approx settings.q_bart 
            # i.e. all values of precision are higher than the unconditional variance of Y
            param.lambda_bart = param.alpha_bart / param.beta_bart      #FIXME: better init?
            cache_tmp['nn_prior_term'] = 0.5 * np.log(param.mu_prec) - 0.5 * param.mu_prec * param.mu_mean * param.mu_mean
        cache_tmp['sum_y'] = np.sum(data['y_train'])
        cache_tmp['sum_y2'] = np.sum(data['y_train'] ** 2)
        cache_tmp['n_points'] = data['n_train']
        cache_tmp['half_log_2pi'] = 0.5 * np.log(2 * math.pi)
    # pre-compute stuff
    cache = {}
    cache['range_n_dim'] = list(range(data['n_dim']))
    if settings.optype == 'class':
        cache['range_n_class'] = list(range(data['n_class']))
        cache['alpha_prior_term'] = gammaln(param.alpha) - data['n_class'] * gammaln(float(param.alpha) / data['n_class'])
        assert(not np.isinf(abs(cache['alpha_prior_term'])))
        cache['lookup_gammaln_1'] = np.array([gammaln(x + param.alpha / data['n_class']) for x in range(data['n_train'] + 1)])
        cache['lookup_gammaln_2'] = np.array([gammaln(x + param.alpha) for x in range(data['n_train'] + 1)])
    else:
        if settings.prior == 'cgm':
            cache['ng_prior_term'] = param.alpha_0 * np.log(param.beta_0) + 0.5 * np.log(param.kappa_0) - gammaln(param.alpha_0)
        else:
            cache['nn_prior_term'] = 0.5 * np.log(param.mu_prec) - 0.5 * param.mu_prec * param.mu_mean * param.mu_mean
        cache['half_log_2pi'] = 0.5 * np.log(2 * math.pi)
    feat_val2idx = {}   # maps unique values to idx for feat_score_cumsum
    feat_idx2midpoint = {}   # maps idx of interval to midpoint
    feat_score_cumsum_prior = {}         # cumsum of scores of each interval for prior
    feat_k_log_prior = (-np.log(float(data['n_dim']))) * np.ones(data['n_dim'])         # log prior of k
    for feat_id in cache['range_n_dim']:
        x_tmp = data['x_train'][:, feat_id]
        idx_sort = np.argsort(x_tmp)
        feat_unique_values = np.unique(x_tmp[idx_sort])
        feat_val2idx[feat_id] = {}
        n_unique = len(feat_unique_values)
        for n, x_n in enumerate(feat_unique_values):
            feat_val2idx[feat_id][x_n] = n     # even min value may be looked up
        # first "interval" has width 0 since points to the left of that point are chosen with prob 0
        feat_idx2midpoint[feat_id] = np.zeros(n_unique)
        feat_idx2midpoint[feat_id][1:] = (feat_unique_values[1:] + feat_unique_values[:-1]) / 2.0
        # each interval is represented by its midpoint
        diff_feat_unique_values = np.diff(feat_unique_values)
        log_diff_feat_unique_values_norm = np.log(diff_feat_unique_values) \
                            - np.log(feat_unique_values[-1] - feat_unique_values[0])
        feat_score_prior_tmp = np.zeros(n_unique)
        feat_score_prior_tmp[1:] = diff_feat_unique_values
        feat_score_cumsum_prior[feat_id] = np.cumsum(feat_score_prior_tmp)
        if settings.verbose >= 2:
            print('check if all these numbers are the same:')
            print(n_unique, len(feat_score_cumsum_prior[feat_id]))
            print('x (sorted) =  %s' % (x_tmp[idx_sort]))
            print('y (corresponding to sorted x) = %s' % (data['y_train'][idx_sort]))
    cache['feat_val2idx'] = feat_val2idx
    cache['feat_idx2midpoint'] = feat_idx2midpoint
    cache['feat_score_cumsum_prior'] = feat_score_cumsum_prior
    if settings.proposal == 'prior':
        cache['feat_score_cumsum'] = cache['feat_score_cumsum_prior']
    cache['feat_k_log_prior']  = feat_k_log_prior
    # use prob_feat instead of score_feat here; else need to pass sum of scores to log_sis_ratio
    cache['prob_feat'] = np.exp(feat_k_log_prior)
    return (param, cache, cache_tmp)


def evaluate_predictions_fast(p, x, y, data, param, settings):
    if settings.optype == 'class':
        pred_prob = p.predict_class_fast(x, data['n_class'], param.alpha)
        pred_all = {'pred_prob': pred_prob}
    else:
        pred_mean, pred_prob = p.predict_real_fast(x, y, param, settings)
        pred_all =  {'pred_prob': pred_prob, 'pred_mean': pred_mean}
    return pred_all


def compute_tree_properties(alpha_split, beta_split, n_sample=1000):
    #NOTE: X is completely ignored here; in actual simulations, the tree 
    #      might stop early if features don't support splits/ there are insufficient datapoints
    tree_stats = np.zeros((n_sample, 4))
    for itr in range(n_sample):
        t, depth, num_nodes = generate_tree(alpha_split, beta_split)
        tree_stats[itr, :] = np.array((depth, num_nodes, len(t['leaf']), len(t['nonleaf'])))
    print('average stats from %d samples:' % n_sample)
    print('depth, num_nodes, num_leaf, num_nonleaf = %s' % np.mean(tree_stats, axis=0))


def generate_tree(alpha_split, beta_split):
    unknown = [0]
    t = {'leaf':[], 'nonleaf':[]}
    depth_max = 0
    num_nodes = 0
    while len(unknown) > 0:
        node_id = unknown.pop(0)
        prob = alpha_split * math.pow(1 + get_depth(node_id), -beta_split)
        tmp = np.random.rand()
        num_nodes += 1
        if tmp <= prob:
            t['leaf'].append(node_id)
            l, r = get_children_id(node_id)
            unknown.append(l)
            unknown.append(r)
            depth_max = max(depth_max, get_depth(l))
        else:
            t['nonleaf'].append(node_id)
            depth_max = max(depth_max, get_depth(node_id))
    assert((len(t['leaf']) + len(t['nonleaf'])) == num_nodes)
    return (t, depth_max, num_nodes)


def no_valid_split_exists(data, cache, train_ids, settings):
# faster way to check for existence of valid split than find_valid_dimensions
    op = True
    for feat_id in cache['range_n_dim']: 
        x_min = np.min(data['x_train'][train_ids, feat_id])
        x_max = np.max(data['x_train'][train_ids, feat_id])
        idx_min = cache['feat_val2idx'][feat_id][x_min]
        idx_max = cache['feat_val2idx'][feat_id][x_max]
        if idx_min != idx_max:
            op = False
            break
    return op


def compute_gamma_param(min_val, alpha, q, init_val=-1.0):
    # alpha, beta are shape and rate of gamma distribution
    # solves for the equation: gammacdf(min_val, shape=alpha, rate=beta) = 1 - q
    # in the gamma distribution, we find the rate parameter such that q% of the 
    #   probability mass is assigned to values > min_val
    #init_val = 1.0
    if init_val < 0:    
        init_val = alpha / min_val / 3
        # intuition: i expect mean of the gamma distribution, alpha/beta = 3 * min_val
    solution = fsolve(lambda beta: gdtrc(beta, alpha, min_val) - q, init_val)
    try:
        assert(abs(gdtrc(solution, alpha, min_val) - q) < 1e-3)
    except AssertionError:
        print('Failed to obtain the right solution: beta_init = %s, q = %s, ' \
                'gdtrc(solution, alpha, min_val) = %s' \
                % (init_val, q, gdtrc(solution, alpha, min_val)))
        print('Trying a new initial value for beta')
        new_init = alpha / min_val / 5
        # intuition: i expect mean of the gamma distribution, alpha/beta = 5 * min_val
    return solution


def check_p_value(m1, m2, v1_n1, v2_n2):
    from scipy.stats import norm
    r = (m1 - m2) / ((v1_n1 + v2_n2) ** 0.5)
    r_cdf = norm.cdf(- np.abs(r))
    print('norm cdf = %s' % r_cdf)


def main():
    print('Running test_compute_test_metrics_classification()')
    test_compute_test_metrics_classification()
    print('Running test_compute_test_metrics_regression()')
    test_compute_test_metrics_regression()
    print('Running test_compute_t_loglik()')
    test_compute_t_loglik()
    print('Running test_compute_ng_normalizer(alpha_0=2, beta_0=1, mu_0=0, kappa_0=1, n_points=1000)')
    test_compute_ng_normalizer(alpha_0=2, beta_0=1, mu_0=0, kappa_0=1, n_points=1000)


if __name__ == "__main__":
    main()
