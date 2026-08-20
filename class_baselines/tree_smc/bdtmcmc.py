#!/usr/bin/env python
# MCMC algorithm for learning decision trees: Two variants are available
# -- "chipman" based on the paper: "Bayesian CART model search" by Chipman et al.
# Example usage:
# ./bdtmcmc.py --dataset toy --alpha 5.0 --alpha_split 0.95 --beta_split 0.5 --save 0


import sys
import optparse
import math
import pickle as pickle
import numpy as np
import pprint as pp
from scipy.special import gammaln, digamma
from copy import copy
import matplotlib.pyplot as plt
from .utils import hist_count, sample_multinomial, sample_polya, check_if_one, check_if_zero
from .tree_utils import *
import random
from .bdtsmc import Particle
import time
# setting numpy options to debug RuntimeWarnings
#np.seterr(divide='raise')
np.seterr(divide='ignore')      # to avoid warnings for np.log(0)


STEP_NAMES = ['grow', 'prune', 'change', 'swap']

def process_command_line():
    parser = parser_add_common_options()
    parser = parser_add_smc_options(parser)
    parser = parser_add_mcmc_options(parser)
    settings, args = parser.parse_args()
    parser_check_common_options(parser, settings)
    parser_check_smc_options(parser, settings)
    parser_check_mcmc_options(parser, settings)
    return settings


def parser_check_mcmc_options(parser, settings):
    fail(parser, settings.n_iterations < 1, 'number of iterations needs to be >= 1')
    fail(parser, not(settings.sample_y==0 or settings.sample_y==1), 'sample_y needs to be 0/1')
    fail(parser, not(settings.mcmc_type=='chipman'), \
            'mcmc_type needs to be chipman')


def parser_add_mcmc_options(parser):
    group = optparse.OptionGroup(parser, "MCMC options")
    group.add_option('--mcmc_type', dest='mcmc_type', default='chipman',
                      help='type of MCMC (chipman/prior)')
    group.add_option('--sample_y', dest='sample_y', default=0, type='int',
                      help='do you want to sample the labels (successive conditional simulator in "Getting it right")? (1/0)')
    group.add_option('--n_iterations', dest='n_iterations', default=100, type='int',
                      help='number of MCMC iterations')
    parser.add_option_group(group)
    return parser


# class PMCMC(object):
#     def __init__(self, data, settings):
#         particles, log_pd, log_weights = init_run_smc(data, settings)
#         print '\ninitializing particle mcmc'
#         print log_pd
#         print log_weights
#         k = sample_multinomial(softmax(log_weights))
#         self.p = particles[k] 
#         self.log_pd = log_pd
#         print
# 
#     def sample(self, data, settings):
#         """ Particle Independent Metropolis Hastings (PIMH) sampler ...
#         see Section 2.4.1 of "Particle MCMC methods" by Andrieu et al., 2009 """
#         particles, log_pd, log_weights = init_run_smc(data, settings)
#         log_acc = (log_pd - self.log_pd)
#         log_r = np.log(np.random.rand(1))
#         if settings.verbose >= 1:
#             print 'log acceptance prob = %f, log rand(1) = %f' % (log_acc, log_r)
#         if log_r <= log_acc:
#             # accept new particle
#             k = sample_multinomial(softmax(log_weights))
#             self.p = particles[k]
#             self.log_pd = log_pd
#             change = True
#         else:
#             change = False
#         return change


def sample_labels_tree(p, data, alpha_vec, range_n_class, param, settings, cache):
    assert(settings.optype == 'class')
    for node_id in p.leaf_nodes:
        n_counts = sample_polya(alpha_vec, len(p.train_ids[node_id]))   # sample from polya
        new_y_list = []     # flatten list
        for y_, n_y in enumerate(n_counts):
            for iter_repeat in range(int(n_y)):
                new_y_list.append(y_)
        random.shuffle(new_y_list)
        for n_, i in enumerate(p.train_ids[node_id]):
            data['y_train'][i] = new_y_list[n_]
        count_new_y_list  = hist_count(new_y_list, range_n_class)
        if settings.debug == 1:
            print('sampling labels at node_id = %3d, old = %s, new = %s' \
                    % (node_id, p.counts[node_id], count_new_y_list))
        p.counts[node_id] = count_new_y_list
        p.loglik[node_id] = compute_dirichlet_normalizer_fast(p.counts[node_id], cache)
    p.loglik_current = sum([p.loglik[node_id] for node_id in p.leaf_nodes])
    if settings.mcmc_type == 'chipman':
        for node_id in p.non_leaf_nodes:
            new_y_list = [data['y_train'][i] for i in p.train_ids[node_id]]
            p.counts[node_id] = hist_count(new_y_list, range_n_class)
            p.loglik[node_id] = compute_dirichlet_normalizer_fast(p.counts[node_id], cache)


def sample_tree(data, settings, param, cache, cache_tmp):
    p = TreeMCMC(list(range(data['n_train'])), param, settings, cache_tmp)
    grow_nodes = [0]
    while grow_nodes:
        node_id = grow_nodes.pop(0)
        p.depth = max(p.depth, get_depth(node_id))
        log_psplit = np.log(p.compute_psplit(node_id, param))
        train_ids = p.train_ids[node_id]
        (do_not_split_node_id, feat_id_chosen, split_chosen, idx_split_global, log_sis_ratio, logprior_nodeid, \
            train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) \
            = p.precomputed_proposal(data, param, settings, cache, node_id, train_ids, log_psplit)
        if do_not_split_node_id:
            p.do_not_split[node_id] = True
        else:
            p.update_left_right_statistics(cache_tmp, node_id, logprior_nodeid, train_ids_left,\
                train_ids_right, loglik_left, loglik_right, feat_id_chosen, split_chosen, \
                idx_split_global, settings, param, data, cache)
            left, right = get_children_id(node_id)
            grow_nodes.append(left)
            grow_nodes.append(right)
            # create mcmc structures
            p.both_children_terminal.append(node_id)
            parent = get_parent_id(node_id) 
            if (node_id != 0) and (parent in p.non_leaf_nodes):
                p.inner_pc_pairs.append((parent, node_id))
            try:
                p.both_children_terminal.remove(parent)
            except ValueError:
                pass
    if settings.debug == 1:
        print('sampled new tree:')
        p.print_tree()
    return p


class TreeMCMC(Tree):
    def __init__(self, train_ids=[], param=empty(), settings=empty(), cache_tmp={}):
        Tree.__init__(self, train_ids, param, settings, cache_tmp)
        self.inner_pc_pairs = []       # list of nodes where both parent/child are non-terminal
        self.both_children_terminal = []

    def compute_log_acc_g(self, node_id, param, len_both_children_terminal, loglik, \
            train_ids_left, train_ids_right, cache, settings, data, grow_nodes):
        # effect of do_not_split does not matter for node_id since it has children
        logprior_children = 0.0
        left, right = get_children_id(node_id)
        if not no_valid_split_exists(data, cache, train_ids_left, settings):
            logprior_children += np.log(self.compute_pnosplit(left, param))
        if not no_valid_split_exists(data, cache, train_ids_right, settings):
            logprior_children += np.log(self.compute_pnosplit(right, param))
        log_acc_prior = np.log(self.compute_psplit(node_id, param)) \
                -np.log(self.compute_pnosplit(node_id, param)) \
            -np.log(len_both_children_terminal) + np.log(len(grow_nodes)) \
            + logprior_children 
        log_acc_loglik = (loglik - self.loglik[node_id])
        log_acc = log_acc_prior + log_acc_loglik
        if settings.verbose >= 2:
            print('compute_log_acc_g: log_acc_loglik = %s, log_acc_prior = %s' \
                    % (log_acc_loglik, log_acc_prior))
        if loglik == -np.inf:   # just need to ensure that an invalid split is not grown
            log_acc = -np.inf
        return log_acc

    def compute_log_inv_acc_p(self, node_id, param, len_both_children_terminal, loglik, grow_nodes, \
            cache, settings, data):
        # 1/acc for PRUNE is acc for GROW except for corrections to both_children_terminal 
        #       and grow_nodes list
        logprior_children = 0.0
        left, right = get_children_id(node_id)
        if not no_valid_split_exists(data, cache, self.train_ids[left], settings):
            logprior_children += np.log(self.compute_pnosplit(left, param))
        if not no_valid_split_exists(data, cache, self.train_ids[right], settings):
            logprior_children += np.log(self.compute_pnosplit(right, param))
        try:
            check_if_zero(logprior_children - self.logprior[left] - self.logprior[right])
        except AssertionError:
            print('oh oh ... looks like a bug in compute_log_inv_acc_p')
            print('term 1 = %s' % logprior_children)
            print('term 2 = %s, 2a = %s, 2b = %s' % (self.logprior[left]+self.logprior[right], \
                     self.logprior[left], self.logprior[right]))
            raise AssertionError
        log_inv_acc_prior = np.log(self.compute_psplit(node_id, param)) \
                - np.log(self.compute_pnosplit(node_id, param)) \
                -np.log(len_both_children_terminal) + np.log(len(grow_nodes)) \
                + logprior_children 
        log_inv_acc_loglik = (loglik - self.loglik[node_id])
        log_inv_acc = log_inv_acc_loglik + log_inv_acc_prior
        if settings.verbose >= 2:
            print('compute_log_inv_acc_p: log_acc_loglik = %s, log_acc_prior = %s' \
                    % (-log_inv_acc_loglik, -log_inv_acc_prior))
        assert(log_inv_acc > -np.inf)
        return log_inv_acc

    def sample(self, data, settings, param, cache):
        step_id = random.randint(0, 3)  # all 4 moves equally likely (or think of 50% grow/prune, 25% change, 25% swap)
        log_acc = -np.inf
        log_r = 0
        self.grow_nodes = [n_id for n_id in self.leaf_nodes \
                    if not stop_split(self.train_ids[n_id], settings, data, cache)]
        grow_nodes = self.grow_nodes
        if step_id == 0:        # GROW step
            if not grow_nodes:
                change = False
            else:
                node_id = random.choice(grow_nodes)
                if settings.verbose >= 1:
                    print('grow_nodes = %s, chosen node_id = %s' % (grow_nodes, node_id))
                do_not_split_node_id, feat_id, split, idx_split_global, logprior_nodeid = \
                        self.sample_split_prior(data, param, settings, cache, node_id)
                assert not do_not_split_node_id
                if settings.verbose >= 1:
                    print('grow: do_not_split = %s, feat_id = %s, split = %s' \
                            % (do_not_split_node_id, feat_id, split))
                train_ids = self.train_ids[node_id]
                (train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) = \
                    compute_left_right_statistics(data, param, cache, train_ids, \
                        feat_id, split, settings)
                loglik = loglik_left + loglik_right
                len_both_children_terminal_new = len(self.both_children_terminal)
                if get_sibling_id(node_id) not in self.leaf_nodes:
                    len_both_children_terminal_new += 1
                log_acc = self.compute_log_acc_g(node_id, param, len_both_children_terminal_new, \
                            loglik, train_ids_left, train_ids_right, cache, settings, data, grow_nodes)
                log_r = np.log(np.random.rand(1))
                if log_r <= log_acc:
                    self.update_left_right_statistics(cache_tmp, node_id, logprior_nodeid, \
                            train_ids_left, train_ids_right, loglik_left, loglik_right, \
                            feat_id, split, idx_split_global, settings, param, data, cache)
                    # MCMC specific data structure updates
                    self.both_children_terminal.append(node_id)
                    parent = get_parent_id(node_id) 
                    if (node_id != 0) and (parent in self.non_leaf_nodes):
                        self.inner_pc_pairs.append((parent, node_id))
                    sibling = get_sibling_id(node_id)

                     # NOTE: added check here to avoid error
                    if sibling in self.leaf_nodes and parent in self.both_children_terminal:
                        self.both_children_terminal.remove(parent)
                    change = True
                else:
                    change = False
        elif step_id == 1:      # PRUNE step
            if not self.both_children_terminal:
                change = False      # nothing to prune here
            else:
                node_id = random.choice(self.both_children_terminal)
                feat_id = self.node_info[node_id][0]
                if settings.verbose >= 1:
                    print('prune: node_id = %s, feat_id = %s' % (node_id, feat_id))
                left, right = get_children_id(node_id)
                loglik = self.loglik[left] + self.loglik[right]
                len_both_children_new = len(self.both_children_terminal)
                grow_nodes_tmp = grow_nodes[:]
                grow_nodes_tmp.append(node_id)
                try:
                    grow_nodes_tmp.remove(left)
                except ValueError:
                    pass
                try:
                    grow_nodes_tmp.remove(right)
                except ValueError:
                    pass
                log_acc = - self.compute_log_inv_acc_p(node_id, param, len_both_children_new, \
                                loglik, grow_nodes_tmp, cache, settings, data)
                log_r = np.log(np.random.rand(1))
                if log_r <= log_acc:
                    self.remove_leaf_node_statistics(left, settings)
                    self.remove_leaf_node_statistics(right, settings)
                    self.leaf_nodes.append(node_id)
                    self.non_leaf_nodes.remove(node_id)
                    self.logprior[node_id] = np.log(self.compute_pnosplit(node_id, param))
                    # OK to set logprior as above since we know that a valid split exists
                    # MCMC specific data structure updates
                    self.both_children_terminal.remove(node_id)
                    parent = get_parent_id(node_id) 
                    if (node_id != 0) and (parent in self.non_leaf_nodes):
                        self.inner_pc_pairs.remove((parent, node_id))
                    if node_id != 0:
                        sibling = get_sibling_id(node_id) 
                        if sibling in self.leaf_nodes:
                            if settings.debug == 1:
                                assert(parent not in self.both_children_terminal)
                            self.both_children_terminal.append(parent)
                    change = True
                else:
                    change = False
        elif step_id == 2:      # CHANGE
            if not self.non_leaf_nodes:
                change = False
            else:
                node_id = random.choice(self.non_leaf_nodes)
                do_not_split_node_id, feat_id, split, idx_split_global, logprior_nodeid = \
                        self.sample_split_prior(data, param, settings, cache, node_id)
                if settings.verbose >= 1:
                    print('change: node_id = %s, do_not_split = %s, feat_id = %s, split = %s' \
                            % (node_id, do_not_split_node_id, feat_id, split))
                # Note: this just samples a split criterion, not guaranteed to "change" 
                assert(not do_not_split_node_id)
                nodes_subtree = self.get_nodes_subtree(node_id)
                nodes_not_in_subtree = self.get_nodes_not_in_subtree(node_id)
                if settings.debug == 1:
                    set1 = set(list(nodes_subtree) + list(nodes_not_in_subtree))
                    set2 = set(self.leaf_nodes + self.non_leaf_nodes)
                    assert(sorted(set1) == sorted(set2))
                self.create_new_statistics(nodes_subtree, nodes_not_in_subtree, node_id, settings)
                self.node_info_new[node_id] = (feat_id, split, idx_split_global)         
                self.evaluate_new_subtree(data, node_id, param, nodes_subtree, cache, settings)
                # log_acc will be be modified below
                log_acc_tmp, loglik_diff, logprior_diff = self.compute_log_acc_cs(nodes_subtree, node_id)
                if settings.debug == 1:
                    self.check_if_same(log_acc_tmp, loglik_diff, logprior_diff)
                log_acc = log_acc_tmp + self.logprior[node_id] - self.logprior_new[node_id]
                log_r = np.log(np.random.rand(1))
                if log_r <= log_acc:
                    self.node_info[node_id] = copy(self.node_info_new[node_id])
                    self.update_subtree(node_id, nodes_subtree, settings)
                    change = True
                else:
                    change = False
        elif step_id == 3:      # SWAP
            if not self.inner_pc_pairs:
                change = False 
            else:
                node_id, child_id = random.choice(self.inner_pc_pairs)
                nodes_subtree = self.get_nodes_subtree(node_id)
                nodes_not_in_subtree = self.get_nodes_not_in_subtree(node_id)
                if settings.debug == 1:
                    set1 = set(list(nodes_subtree) + list(nodes_not_in_subtree))
                    set2 = set(self.leaf_nodes + self.non_leaf_nodes)
                    assert(sorted(set1) == sorted(set2))
                self.create_new_statistics(nodes_subtree, nodes_not_in_subtree, node_id, settings)
                self.node_info_new[node_id] = copy(self.node_info[child_id])
                self.node_info_new[child_id] = copy(self.node_info[node_id])
                if settings.verbose >= 1:
                    print('swap: node_id = %s, child_id = %s' % (node_id, child_id))
                    print('node_info[node_id] = %s, node_info[child_id] = %s' \
                            % (self.node_info[node_id], self.node_info[child_id]))
                self.evaluate_new_subtree(data, node_id, param, nodes_subtree, cache, settings)
                log_acc, loglik_diff, logprior_diff = self.compute_log_acc_cs(nodes_subtree, node_id)
                if settings.debug == 1:
                    self.check_if_same(log_acc, loglik_diff, logprior_diff)
                log_r = np.log(np.random.rand(1))
                if log_r <= log_acc:
                    self.node_info[node_id] = copy(self.node_info_new[node_id])
                    self.node_info[child_id] = copy(self.node_info_new[child_id])
                    self.update_subtree(node_id, nodes_subtree, settings)
                    change = True
                else:
                    change = False
        if settings.verbose >= 1:
            print('trying move: step_id = %d, move = %s, log_acc = %s, log_r = %s' \
                    % (step_id, STEP_NAMES[step_id], log_acc, log_r))
        if change:
            self.depth = max([get_depth(node_id) for node_id in \
                    self.leaf_nodes])
            self.loglik_current = sum([self.loglik[node_id] for node_id in \
                    self.leaf_nodes])
            if settings.verbose >= 1:
                print('accepted move: step_id = %d, move = %s' % (step_id, STEP_NAMES[step_id]))
                self.print_stuff()
        if settings.debug == 1:
            both_children_terminal, inner_pc_pairs = self.recompute_mcmc_data_structures()
            print('\nstats from recompute_mcmc_data_structures')
            print('both_children_terminal = %s' % both_children_terminal)
            print('inner_pc_pairs = %s' % inner_pc_pairs)
            assert(sorted(both_children_terminal) == sorted(self.both_children_terminal))
            assert(sorted(inner_pc_pairs) == sorted(self.inner_pc_pairs))
            grow_nodes_new = [n_id for n_id in self.leaf_nodes \
                    if not stop_split(self.train_ids[n_id], settings, data, cache)]
            if change and (step_id == 1):
                print('grow_nodes_new = %s, grow_nodes_tmp = %s' % (sorted(grow_nodes_new), sorted(grow_nodes_tmp)))
                assert(sorted(grow_nodes_new) == sorted(grow_nodes_tmp))
        return (change, step_id)

    def check_if_same(self, log_acc, loglik_diff, logprior_diff):
        # change/swap operations should depend only on what happens in current subtree
        loglik_diff_2 =  sum([self.loglik_new[node] for node in self.leaf_nodes]) \
                        - sum([self.loglik[node] for node in self.leaf_nodes])
        logprior_diff_2 = sum([self.logprior_new[node] for node in self.logprior_new]) \
                         - sum([self.logprior[node] for node in self.logprior])
        log_acc_2 = loglik_diff_2 + logprior_diff_2
        try:
            check_if_zero(log_acc - log_acc_2)
        except AssertionError:
            if not ((log_acc == -np.inf) and (log_acc_2 == -np.inf)):
                print('check if terms match:')
                print('loglik_diff = %s, loglik_diff_2 = %s' % (loglik_diff, loglik_diff_2))
                print('logprior_diff = %s, logprior_diff_2 = %s' % (logprior_diff, logprior_diff_2))
                raise AssertionError

    def compute_log_acc_cs(self, nodes_subtree, node_id):
        # for change or swap operations
        loglik_old = sum([self.loglik[node] for node in nodes_subtree if node in self.leaf_nodes])
        loglik_new = sum([self.loglik_new[node] for node in nodes_subtree if node in self.leaf_nodes])
        loglik_diff = loglik_new - loglik_old
        logprior_old = sum([self.logprior[node] for node in nodes_subtree])
        logprior_new = sum([self.logprior_new[node] for node in nodes_subtree])
        logprior_diff = logprior_new - logprior_old
        log_acc = loglik_diff + logprior_diff
        return (log_acc, loglik_diff, logprior_diff)

    def create_new_statistics(self, nodes_subtree, nodes_not_in_subtree, node_id, settings):
        self.node_info_new = self.node_info.copy()
        self.counts_new = {}
        self.train_ids_new = {}
        self.loglik_new = {}
        self.logprior_new = {}
        for node in nodes_not_in_subtree:
            self.loglik_new[node] = self.loglik[node]
            self.logprior_new[node] = self.logprior[node]
            self.train_ids_new[node] = self.train_ids[node]
            if settings.optype == 'class':
                self.counts_new[node] = self.counts[node]
            else:
                self.sum_y_new[node] = self.sum_y[node]
                self.sum_y2_new[node] = self.sum_y2[node]
                self.n_points_new[node] = self.n_points[node]
                self.param_n_new[node] = self.param_n[node]
        for node in nodes_subtree:
            self.loglik_new[node] = -np.inf
            self.logprior_new[node] = -np.inf
            self.train_ids_new[node] = []
            if settings.optype == 'class':
                self.counts_new[node] = np.zeros(self.counts[node].shape)
            else:
                self.sum_y_new[node] = np.nan
                self.sum_y2_new[node] = np.nan
                self.n_points_new[node] = 0
                self.param_n_new[node] = self.param_n[node] * 0.0

    def evaluate_new_subtree(self, data, node_id_start, param, nodes_subtree, cache, settings):
        for i in self.train_ids[node_id_start]:
            x_, y_ = data['x_train'][i, :], data['y_train'][i]
            node_id = copy(node_id_start)
            while True:
                self.counts_new[node_id][y_] += 1
                self.train_ids_new[node_id].append(i)
                if node_id in self.leaf_nodes:
                    break
                left, right = get_children_id(node_id)
                feat_id, split, idx_split_global = self.node_info_new[node_id]           # splitting on new criteria
                if x_[feat_id] <= split:
                    node_id = left
                else:
                    node_id = right
        for node_id in nodes_subtree:
            if np.sum(self.counts_new[node_id]) > 0:
                self.loglik_new[node_id] = compute_dirichlet_normalizer_fast(self.counts_new[node_id], cache)
            else:
                self.loglik_new[node_id] = -np.inf
            if node_id in self.leaf_nodes:
                if stop_split(self.train_ids_new[node_id], settings, data, cache):
                # if leaf is empty, logprior_new[node_id] = 0.0 is incorrect; however
                #      loglik_new[node_id] = -np.inf will reject empty leaves
                    self.logprior_new[node_id] = 0.0
                else:
                    # node with just 1 data point earlier could have more data points now 
                    self.logprior_new[node_id] = np.log(self.compute_pnosplit(node_id, param))
            else:
                # split probability might have changed if train_ids have changed
                self.recompute_prob_split(data, param, settings, cache, node_id)
        if settings.debug == 1:
            try:
                check_if_zero(self.loglik[node_id_start] - self.loglik_new[node_id_start])
            except AssertionError:
                print('train_ids[node_id_start] = %s, train_ids_new[node_id_start] = %s' \
                        % (self.train_ids[node_id_start], self.train_ids_new[node_id_start]))
                raise AssertionError
    
    def update_subtree(self, node_id, nodes_subtree, settings):
        for node in nodes_subtree:
            self.loglik[node] = copy(self.loglik_new[node])
            self.logprior[node] = copy(self.logprior_new[node])
            self.train_ids[node] = self.train_ids_new[node][:]
            if settings.optype == 'class':
                self.counts[node] = self.counts_new[node].copy()
            else:
                self.sum_y[node] = copy(self.sum_y_new[node])
                self.sum_y2[node] = copy(self.sum_y2_new[node])
                self.n_points[node] = copy(self.n_points_new[node])
                self.param_n[node] = self.param_n_new[node][:]

    def print_stuff(self):
        print('tree statistics:')
        print('leaf nodes = ')
        print(self.leaf_nodes) 
        print('non leaf nodes = ')
        print(self.non_leaf_nodes) 
        print('inner pc pairs')
        print(self.inner_pc_pairs) 
        print('both children terminal')
        print(self.both_children_terminal)
        print('loglik = ')
        print(self.loglik)
        print('logprior = \n%s' % self.logprior)
        print('do_not_split = \n%s'  % self.do_not_split)
        print() 

    def get_nodes_not_in_subtree(self, node_id):
        all_nodes = set(self.leaf_nodes + self.non_leaf_nodes)
        reqd_nodes = all_nodes - set(self.get_nodes_subtree(node_id))
        return list(reqd_nodes)

    def get_nodes_subtree(self, node_id):
        # NOTE: current node_id is included in nodes_subtree as well
        node_list = []
        expand = [node_id]
        while len(expand) > 0:
            node = expand.pop(0) 
            node_list.append(node)
            if node not in self.leaf_nodes:
                left, right = get_children_id(node)
                expand.append(left)
                expand.append(right)
        return node_list

    def recompute_mcmc_data_structures(self):   #, settings, param):
        nodes_to_visit = sorted(set(self.leaf_nodes + self.non_leaf_nodes))
        both_children_terminal = []
        inner_pc_pairs = []
        while nodes_to_visit:
            node_id = nodes_to_visit[0]
            parent = get_parent_id(node_id)
            if (node_id != 0) and (node_id in self.non_leaf_nodes) and (parent in self.non_leaf_nodes):
                inner_pc_pairs.append((parent, node_id))
            if node_id != 0:
                sibling = get_sibling_id(node_id)
                if (node_id in self.leaf_nodes) and (sibling in self.leaf_nodes) \
                        and (parent not in both_children_terminal):
                    both_children_terminal.append(parent)
            nodes_to_visit.remove(node_id)
        return (both_children_terminal, inner_pc_pairs)


def evaluate_predictions_mcmc(p, data, x, y, param):
    (pred, pred_prob, acc, log_prob) = evaluate_predictions(p, x, y, data, param)
    print('accuracy = %3.2f, log_prob = %.2f' % (acc, log_prob))
    return (pred, pred_prob, acc, log_prob)


def record_stats(p, time_current_iter, change):
    root_node_info = p.node_info.get(0, [-1, 3.14, -1])     # pi implies special case
    flag = 1 if change else 0
    op = np.array([p.compute_loglik(), p.compute_logprior(), p.compute_logprob(),
        p.depth, root_node_info[0], root_node_info[1], 
        len(p.leaf_nodes), len(p.non_leaf_nodes), flag, time_current_iter])
    return op


def main():
    settings = process_command_line()
    print('Current settings:')
    pp.pprint(vars(settings))

    # Resetting random seed
    np.random.seed(settings.init_id * 1000)
    random.seed(settings.init_id * 1000)

    # load data
    print('Loading data ...')
    data = load_data(settings)
    print('Loating data ... completed')
   
    # pre-compute stuff
    param, cache, cache_tmp = precompute(data, settings)
    if settings.verbose >= 1:
        print('cache_tmp=\n%s\ncache=\n%s' % (cache_tmp, cache))
    # initialize stuff for results
    mcmc_stats = np.zeros((settings.n_iterations, 10))
    mcmc_counts_total = np.zeros(4)
    mcmc_counts_acc = np.zeros(4)
    alpha_vec = np.ones(data['n_class']) * np.float(param.alpha) / data['n_class']
    range_n_class = list(range(data['n_class']))

    if settings.dataset == 'toy-small':
        true_posterior = pickle.load(open('toy-small.true_posterior.p', 'rb'))
        posterior_prob = true_posterior['prob']
        empirical_counts = {}
        for k in posterior_prob:
            empirical_counts[k] = 0

    if settings.mcmc_type == 'chipman':
        # Chipman's mcmc version
        # initialize with a random tree
        p = sample_tree(data, settings, param, cache, cache_tmp)
        # you could initialize with empty tree as well
        #p = TreeMCMC(range(data['n_train']), param, settings, cache_tmp)
        if settings.verbose >= 2:
            print('*'*80)
            print('initial tree:')
            p.print_stuff()
            print('*'*80)
    elif settings.mcmc_type == 'pmcmc':
        # Particle-MCMC
        pmcmc = PMCMC(data, settings)
        pmcmc_log_pd = np.zeros(settings.n_iterations)
    elif settings.mcmc_type == 'prior':            # marginal conditional simulator 
        assert(settings.proposal == 'prior')
        # Note tree is not being initialized here
    else:
        raise Exception

    mcmc_tree_predictions = {}
    n_run_avg = 250
    n_sample_interval = 50      # every 50th tree will be sampled and used
    n_store = int((settings.n_iterations) / n_run_avg)
    assert(n_run_avg % n_sample_interval == 0)
    n_run_avg_sample = n_run_avg / n_sample_interval
    print('n_store = %s' % n_store)
    #n_burn_in = 0       # predictions are stored for all iterations ... process burn-in separately
    n_burn_in = int(settings.n_iterations / 2)
    if settings.save == 1:
        mcmc_tree_predictions['train'] = np.zeros((data['n_train'], data['n_class'], n_store))
        mcmc_tree_predictions['test'] = np.zeros((data['n_test'], data['n_class'], n_store))
        mcmc_tree_predictions_train = np.zeros((data['n_train'], data['n_class']))
        mcmc_tree_predictions_test = np.zeros((data['n_test'], data['n_class']))
        mcmc_tree_predictions['train_sample'] = np.zeros((data['n_train'], data['n_class'], n_store))
        mcmc_tree_predictions['test_sample'] = np.zeros((data['n_test'], data['n_class'], n_store))
        mcmc_tree_predictions_train_sample = np.zeros((data['n_train'], data['n_class']))
        mcmc_tree_predictions_test_sample = np.zeros((data['n_test'], data['n_class']))
        mcmc_tree_predictions_train_tmp = None
        mcmc_tree_predictions_test_tmp = None
        mcmc_tree_predictions_tmp_valid = False
        mcmc_tree_predictions['run_avg_stats'] = np.zeros((6, n_store))
        mcmc_tree_predictions['run_avg_stats_sample'] = np.zeros((6, n_store))
    
    time_init = time.clock()
    time_init_run_avg = time.clock()
    itr_run_avg = 0
    
    for itr in range(settings.n_iterations):
        time_init_current = time.clock()
        if settings.verbose >= 1:
            print('%s iteration = %7d %s' % ('*'*30, itr, '*'*30))
        if settings.mcmc_type == 'chipman':
            if (settings.sample_y == 1):    # Successive-conditional simulator
                sample_labels_tree(p, data, alpha_vec, range_n_class, param, settings, cache)
            (change, step_id) = p.sample(data, settings, param, cache)
            mcmc_counts_total[step_id] += 1
            mcmc_counts_acc[step_id] += change
        elif settings.mcmc_type == 'prior':
            change = True
            p = sample_tree(data, settings, param, cache, cache_tmp)
            if settings.sample_y == 1:
                # NOTE: y need not be sampled if mcmc_stats is independent of y
                sample_labels_tree(p, data, alpha_vec, range_n_class, param, settings, cache)   
        if (settings.save == 1) and (change or (not mcmc_tree_predictions_tmp_valid)):
            # prediction tree creation time (but NOT prediction time) is included in timing
            #   to make a fair comparison with SMC
            p.create_prediction_tree(param, data, settings)
        mcmc_stats[itr, :] = record_stats(p, time.clock() - time_init_current, change)
        if settings.dataset == 'toy-small' and settings.mcmc_type == 'chipman' and itr > n_burn_in:
                tree_key = p.gen_tree_key()
                empirical_counts[tree_key] += 1
        if (settings.save == 1):
            if change or (not mcmc_tree_predictions_tmp_valid):
                mcmc_tree_predictions_train_tmp = evaluate_predictions_fast(p, \
                        data['x_train'], data['y_train'], data, param, settings)['pred_prob']
                mcmc_tree_predictions_test_tmp = evaluate_predictions_fast(p, \
                        data['x_test'], data['y_test'], data, param, settings)['pred_prob']
                mcmc_tree_predictions_tmp_valid = True
            mcmc_tree_predictions_train += mcmc_tree_predictions_train_tmp
            mcmc_tree_predictions_test += mcmc_tree_predictions_test_tmp
            if (itr % n_sample_interval == (n_sample_interval - 1)):
                mcmc_tree_predictions_train_sample += mcmc_tree_predictions_train_tmp
                mcmc_tree_predictions_test_sample += mcmc_tree_predictions_test_tmp
            if itr == 0:
                print('itr, itr_run_avg, [acc_train, acc_test, logprob_train, ' \
                    'logprob_test, time_mcmc, time_mcmc_prediction], time_mcmc_cumulative')
            if (itr > 0) and (itr % n_run_avg == (n_run_avg - 1)):
                mcmc_tree_predictions['train'][:, :, itr_run_avg] = mcmc_tree_predictions_train / (itr + 1) 
                mcmc_tree_predictions['test'][:, :, itr_run_avg] = mcmc_tree_predictions_test / (itr + 1)
                if settings.debug == 1:
                    check_if_one(np.sum(mcmc_tree_predictions['train'][:, :, itr_run_avg]) / data['n_train'])
                    check_if_one(np.sum(mcmc_tree_predictions['test'][:, :, itr_run_avg] / data['n_test']))
                (acc_train, log_prob_train) = compute_test_metrics(data['y_train'], \
                                                mcmc_tree_predictions['train'][:, :, itr_run_avg])
                (acc_test, log_prob_test) = compute_test_metrics(data['y_test'], \
                                                mcmc_tree_predictions['test'][:, :, itr_run_avg])
                itr_range = list(range(itr_run_avg * n_run_avg, (itr_run_avg + 1) * n_run_avg))
                if settings.debug == 1:
                    print('itr_range = %s' % itr_range)
                time_mcmc_train = np.sum(mcmc_stats[itr_range, -1]) 
                mcmc_tree_predictions['run_avg_stats'][:, itr_run_avg] = [acc_train, log_prob_train, \
                        acc_test, log_prob_test, \
                        time_mcmc_train, time.clock() - time_init_run_avg]
                print('%7d, %7d, %s, %s' % \
                    (itr, itr_run_avg, mcmc_tree_predictions['run_avg_stats'][:, itr_run_avg].T, \
                    np.sum(mcmc_tree_predictions['run_avg_stats'][-2, :itr_run_avg+1])))
                # subsampled trees
                mcmc_tree_predictions['train_sample'][:, :, itr_run_avg] = mcmc_tree_predictions_train_sample / (itr + 1) * n_sample_interval
                mcmc_tree_predictions['test_sample'][:, :, itr_run_avg] = mcmc_tree_predictions_test_sample / (itr + 1) * n_sample_interval
                if settings.debug == 1:
                    check_if_one(np.sum(mcmc_tree_predictions['train_sample'][:, :, itr_run_avg]) / data['n_train'])
                    check_if_one(np.sum(mcmc_tree_predictions['test_sample'][:, :, itr_run_avg] / data['n_test']))
                (acc_train, log_prob_train) = compute_test_metrics(data['y_train'], \
                                                mcmc_tree_predictions['train_sample'][:, :, itr_run_avg])
                (acc_test, log_prob_test) = compute_test_metrics(data['y_test'], \
                                                mcmc_tree_predictions['test_sample'][:, :, itr_run_avg])
                mcmc_tree_predictions['run_avg_stats_sample'][:, itr_run_avg] = [acc_train, log_prob_train, \
                        acc_test, log_prob_test, \
                        time_mcmc_train, time.clock() - time_init_run_avg]
                itr_run_avg += 1
                time_init_run_avg = time.clock()
    if settings.verbose >= 1:
        print('mcmc_stats = ') 
        print(mcmc_stats[:, :3])
    print('summary of mcmc_stats (after burn-in = %s iterations):' % n_burn_in)
    mcmc_stats_burn_in = mcmc_stats[n_burn_in:, :]
    print(mcmc_stats_burn_in.shape)
    print('mean = \n%s' % np.mean(mcmc_stats_burn_in, axis=0))
    print('var / (n_iterations - n_burn_in) = \n%s' \
            % (np.var(mcmc_stats_burn_in, axis=0) / mcmc_stats_burn_in.shape[0]))
    print('move type\tnum_total = %s\tnum_accepted = %s' % (np.sum(mcmc_counts_total), np.sum(mcmc_counts_acc)))
    for step_id in range(4):
        print('%s\t%s\t%s' % (STEP_NAMES[step_id], mcmc_counts_total[step_id], mcmc_counts_acc[step_id]))
    if settings.dataset == 'toy-small' and settings.mcmc_type == 'chipman':
        print('check empirical vs true posterior probability... both values should be approximately same')
        total_empirical_counts = np.float(sum([empirical_counts[k] for k in empirical_counts]))
        all_trees = true_posterior['all_trees']
        log_weight_2 = np.zeros(len(all_trees))
        for i_p, k in enumerate(all_trees):
            log_weight_2[i_p] = all_trees[k].compute_logprob()
        prob_tree = softmax(log_weight_2)
        for i_p, k in enumerate(posterior_prob):
            check_if_zero(posterior_prob[k] - prob_tree[i_p])
        for i_p, k in enumerate(posterior_prob):
            print('k = %40s, true_posterior = %10s, empirical_value = %10s' % \
                    (k, posterior_prob[k], empirical_counts[k] / total_empirical_counts))
    
    # print results
    #print '\nResults on training data (using just the final sample)'
    #evaluate_predictions_mcmc(p, data, data['x_train'], data['y_train'], param)
    #print '\nResults on test data (using just the final sample)'
    #evaluate_predictions_mcmc(p, data, data['x_test'], data['y_test'], param)
    print('\nTotal time (seconds) = %f' % (time.clock() - time_init))
    
    if settings.save == 1:
        print('predictions averaged across all previous trees:')
        print('acc_train, mean log_prob_train, acc_test, mean log_prob_test, mcmc time (current batch), mcmc+prediction time (current batch)')
        print(mcmc_tree_predictions['run_avg_stats'].T)

    # Write results to disk
    if settings.save == 1:
        filename = get_filename_mcmc(settings)
        print('filename = ' + filename)
        results = {}
        results['mcmc_stats'] = mcmc_stats
        results['settings'] = settings
        results['n_run_avg'] = n_run_avg
        results['n_sample_interval'] = n_sample_interval
        if settings.mcmc_type == 'pmcmc':
            results['log_pd'] = pmcmc_log_pd
        pickle.dump(results, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        filename2 = filename[:-1] + 'tree_predictions.p'
        print('predictions stored in file: %s' % filename2)
        pickle.dump(mcmc_tree_predictions, open(filename2, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
