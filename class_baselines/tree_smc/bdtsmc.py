#!/usr/bin/env python
# Particle filtering algorithm for Bayesian learning of decision trees
# Input parameters:
#   - alpha: concentration parameter for symmetric Dirichlet (alpha/K is the mass on each class)
#   - alpha_split, beta_split: parameters defining probability of split (tree prior)
# Example usage:
# Classification: ./bdtsmc.py --dataset toy-small --n_particles 1000 --max_iterations 5000 --proposal prior --alpha_split 0.95 --beta_split .5 --n_islands 5
# Regression: ./bdtsmc.py --dataset toy-reg --optype real --n_particles 1 --max_iterations 500 --proposal empirical --alpha_split 0.95 --beta_split .05 --mu_0 0 --alpha_0 40 --beta_0 100 --kappa_0 1
# Reference for parameter usage in the regression case:
#   [M07]: "Conjugate Bayesian analysis of the Gaussian distribution", Kevin P. Murphy, 2007


import sys
import os
import optparse
import math
import time
import pickle as pickle
import random
import pprint as pp
import numpy as np
from scipy.special import gammaln, digamma, gamma
from copy import copy
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .utils import hist_count, logsumexp, softmax, sample_multinomial, \
        sample_multinomial_scores, empty, assert_no_nan, check_if_zero
from .tree_utils import Tree, Param
from .tree_utils import parser_add_common_options, parser_add_smc_options, parser_check_common_options, \
                           parser_check_smc_options, load_data, plot_particles, \
                           get_depth, get_children_id, get_filename_smc, \
                           compute_test_metrics_classification, compute_test_metrics_regression, compute_test_metrics, \
                           stop_split, compute_dirichlet_normalizer, compute_normal_normalizer, \
                           compute_dirichlet_normalizer_fast, evaluate_predictions, init_left_right_statistics, \
                           subsample_features, compute_left_right_statistics, compute_entropy, \
                           precompute, evaluate_predictions_fast, evaluate_performance_tree
from itertools import count
# setting numpy options to debug RuntimeWarnings
#np.seterr(divide='raise')
np.seterr(divide='ignore')      # to avoid warnings for np.log(0)


def process_command_line():
    parser = parser_add_common_options()
    parser = parser_add_smc_options(parser)
    settings, args = parser.parse_args()
    parser_check_common_options(parser, settings)
    parser_check_smc_options(parser, settings)
    return settings


class Particle(Tree):
    def __init__(self, train_ids=[], param=empty(), settings=empty(), cache_tmp={}):
        Tree.__init__(self, train_ids, param, settings, cache_tmp)
        self.ancestry = [0]     # first iteration, parent chosen is 0 (all are equal)
        self.num_nodes_processed_itr = []
        if cache_tmp:
            self.do_not_grow = False
            if settings.grow == 'next':
                self.grow_nodes = [0]

    def posterior_proposal(self, data, param, settings, cache, node_id, \
            train_ids, log_psplit):
        feat_id_valid, score_feat, feat_split_info, split_not_supported = \
                self.find_valid_dimensions(data, cache, train_ids, settings)
        loglik_nosplit = self.loglik[node_id]
        sum_logprob = 0.0
        log_sis_ratio = 0.0
        feat_id_perm, n_feat_subset, log_prob_feat = \
                subsample_features(settings, feat_id_valid, score_feat, split_not_supported)
        prob_split_prior_feat = {}
        logprob_thresholds_per_feat = {}
        logprob_values = np.zeros(n_feat_subset + 1) 
        if split_not_supported:      # last column corresponds to no split
            logprob_values[-1] = 0.0
        else:
            logprob_values[-1] = np.log(self.compute_pnosplit(node_id, param))
        for n_feat_id, feat_id in enumerate(feat_id_perm):
            if settings.verbose >= 3:
                print('feat_id = %3d' % feat_id)
            x_tmp = data['x_train'][train_ids, feat_id]
            y_tmp = data['y_train'][train_ids]
            idx_sort = np.argsort(x_tmp)
            idx_min, idx_max, x_min, x_max, feat_score_cumsum_prior_current = feat_split_info[feat_id] 
            z_prior = np.float(feat_score_cumsum_prior_current[idx_max] - feat_score_cumsum_prior_current[idx_min])
            prob_split_prior = np.diff(feat_score_cumsum_prior_current[idx_min: (idx_max+1)] - \
                            feat_score_cumsum_prior_current[idx_min])/ z_prior
            log_prob_split_prior = np.log(prob_split_prior)
            prob_split_prior_feat[feat_id] = prob_split_prior
            idx_splits_local = np.arange(idx_max - idx_min)
            idx_splits_global = idx_splits_local + idx_min + 1
            feat_split_point_values = cache['feat_idx2midpoint'][feat_id][idx_splits_global]
            logprob_tmp = log_prob_split_prior + log_psplit + log_prob_feat[feat_id]
            if settings.include_child_prob == 1:
                # This is a different way of unrolling the prior, where we include the 
                #   cost of not stopping children in the weight update for the parent
                left, right = get_children_id(node_id)
                logprob_tmp += np.log(self.compute_pnosplit(left, param)) + np.log(self.compute_pnosplit(right, param))
            if settings.debug == 1:
                assert(feat_split_point_values.shape == logprob_tmp.shape)
            if settings.verbose >= 3:
                print('initial value: logprob_tmp = \n%s' % logprob_tmp)
            max_idx_split = idx_splits_local[-1]
            if settings.debug == 1:
                assert(np.abs(np.sum(prob_split_prior) - 1) < 1e-12)
                assert(min(idx_splits_global) == (idx_min+1))
                assert(max(idx_splits_global) == idx_max)
                assert (max_idx_split == (len(feat_split_point_values) - 1))
            if settings.optype == 'class':
                cnt_left = np.zeros(self.counts[node_id].shape)
                cnt_right = self.counts[node_id].copy()
            else:
                sum_y_left = 0.
                sum_y2_left = 0.
                n_points_left = 0
                sum_y_right = self.sum_y[node_id].copy()
                sum_y2_right = self.sum_y2[node_id].copy()
                n_points_right = copy(self.n_points[node_id])
            idx_split_start = 0
            x_old = x_min
            for i_tmp, i in enumerate(idx_sort):
                x_, y_ = x_tmp[i], y_tmp[i]
                if settings.verbose >= 4: 
                    print('x_ = %s' % x_)
                if x_ > x_old:      # compute only when x_ changes (saves time when multiple x_'s take on same value)
                    if settings.optype == 'class':
                        loglik_left = compute_dirichlet_normalizer_fast(cnt_left, cache)
                        loglik_right = compute_dirichlet_normalizer_fast(cnt_right, cache)
                    else:
                        loglik_left = compute_normal_normalizer(sum_y_left, sum_y2_left, \
                                            n_points_left, param, cache, settings)[0]
                        loglik_right = compute_normal_normalizer(sum_y_right, sum_y2_right, \
                                            n_points_right, param, cache, settings)[0]
                    logprob_term1 = settings.temper_factor * (loglik_left + loglik_right - loglik_nosplit)
                    if x_ < x_max:
                        idx_split_end = cache['feat_val2idx'][feat_id][x_] - idx_min - 1 # -1 required since idx_min is always 1 less than val2idx
                        if settings.verbose >= 3:
                            print('i_tmp = %s, idx_split_start = %s, idx_split_end = %s' % (i_tmp, idx_split_start, idx_split_end))
                        logprob_tmp[idx_split_start:(idx_split_end+1)] += logprob_term1
                        idx_split_start = idx_split_end + 1
                    else:
                        if settings.verbose >= 3:
                            print('i_tmp = %s, idx_split_start = %s, idx_split_end = last term' % (i_tmp, idx_split_start))
                        logprob_tmp[idx_split_start:] += logprob_term1
                        break
                    if settings.verbose >= 3:
                        print('logprob_tmp = \n%s' % logprob_tmp)
                    x_old = x_
                if settings.optype == 'class':
                    cnt_left[y_] += 1
                    cnt_right[y_] -= 1
                else:
                    y_2 = y_ ** 2
                    sum_y_left += y_
                    sum_y_right -= y_
                    sum_y2_left += y_2
                    sum_y2_right -= y_2
                    n_points_left += 1
                    n_points_right -= 1
            if settings.verbose >= 3:
                print('final value: logprob_tmp = \n%s' % logprob_tmp)
            logprob_thresholds_per_feat[feat_id] = [logprob_tmp, feat_split_point_values] 
            logprob_values[n_feat_id] = logsumexp(logprob_tmp)
        sum_logprob = logsumexp(logprob_values)
        prob = softmax(logprob_values)
        if settings.debug == 1:
            assert_no_nan(prob, 'prob')
        idx_chosen = sample_multinomial(prob)
        if idx_chosen == n_feat_subset:     # last item was the do not split option
            do_not_split_node_id = True
            feat_id_chosen = -1
            split_chosen = 3.14 
            idx_split_global = -1
            logprior_nodeid = logprob_values[-1]
            (train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) = \
                init_left_right_statistics()
        else:
            do_not_split_node_id = False
            feat_id_chosen = feat_id_perm[idx_chosen]       
            logprob_tmp, feat_split_point_values = logprob_thresholds_per_feat[feat_id_chosen]
            prob_split = softmax(logprob_tmp)
            idx_split_chosen = sample_multinomial(prob_split)
            idx_min, idx_max, x_min, x_max, feat_score_cumsum_prior_current = feat_split_info[feat_id_chosen] 
            idx_split_global = idx_split_chosen + idx_min + 1
            split_chosen = feat_split_point_values[idx_split_chosen]
            logprior_nodeid_k_tau = log_prob_feat[feat_id_chosen] \
                            + np.log(prob_split_prior_feat[feat_id_chosen][idx_split_chosen])
            logprior_nodeid = log_psplit + logprior_nodeid_k_tau
            (train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) = \
                compute_left_right_statistics(data, param, cache, train_ids, feat_id_chosen, split_chosen, settings)
        log_sis_ratio = sum_logprob
        return (do_not_split_node_id, feat_id_chosen, split_chosen, idx_split_global, log_sis_ratio, logprior_nodeid, \
            train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) 
   
    def posterior_proposal_all(self, data, param, settings, cache, node_id):
        """ generates all trees according to the one-step optimal proposal.
        code may use a different looping structure instead of posterior_proposal,
        but numbers will exactly match. 
        This routine is used only on a tiny toy dataset, hence didn't optimize this routine
        NOTE: this code hasn't been verified for correctness when features are being subsampled
        """
        train_ids = self.train_ids[node_id]
        feat_id_valid, score_feat, feat_split_info, split_not_supported = \
                self.find_valid_dimensions(data, cache, train_ids, settings)
        loglik_nosplit = self.loglik[node_id]
        feat_id_perm, n_feat_subset, log_prob_feat = \
                subsample_features(settings, feat_id_valid, score_feat, split_not_supported)
        prob_split_prior_feat = {}
        logprob_thresholds_per_feat = {}
        logprob_values = np.zeros(n_feat_subset + 1) 
        if split_not_supported:      # last column corresponds to no split
            logprob_values[-1] = 0.0
        else:
            logprob_values[-1] = np.log(self.compute_pnosplit(node_id, param))
        log_psplit = np.log(self.compute_psplit(node_id, param))
        for n_feat_id, feat_id in enumerate(feat_id_perm):
            if settings.verbose >= 3:
                print('feat_id = %3d' % feat_id)
            x_tmp = data['x_train'][train_ids, feat_id]
            y_tmp = data['y_train'][train_ids]
            idx_sort = np.argsort(x_tmp)
            n_max_idx_sort = len(idx_sort) - 1
            idx_min, idx_max, x_min, x_max, feat_score_cumsum_prior_current = feat_split_info[feat_id] 
            z_prior = np.float(feat_score_cumsum_prior_current[idx_max] - feat_score_cumsum_prior_current[idx_min])
            prob_split_prior = np.diff(feat_score_cumsum_prior_current[idx_min: (idx_max+1)] - \
                            feat_score_cumsum_prior_current[idx_min])/ z_prior
            assert(np.abs(np.sum(prob_split_prior) - 1) < 1e-12)
            prob_split_prior_feat[feat_id] = prob_split_prior
            idx_splits_local = np.arange(idx_max - idx_min)
            idx_splits_global = idx_splits_local + idx_min + 1
            if settings.debug == 1:
                assert(min(idx_splits_global) == (idx_min+1))
                assert(max(idx_splits_global) == idx_max)
            feat_split_point_values = cache['feat_idx2midpoint'][feat_id][idx_splits_global]
            logprob_tmp = np.zeros(len(feat_split_point_values))
            logprior_tmp = np.zeros(len(feat_split_point_values))
            max_idx_split = idx_splits_local[-1]
            assert (max_idx_split == (len(feat_split_point_values) - 1))
            if settings.optype == 'class':
                cnt_left = np.zeros(self.counts[node_id].shape)
                cnt_right = self.counts[node_id].copy()
            else:
                sum_y_left = 0.
                sum_y2_left = 0.
                n_points_left = 0
                sum_y_right = self.sum_y[node_id].copy()
                sum_y2_right = self.sum_y2[node_id].copy()
                n_points_right = copy(self.n_points[node_id])
            i_tmp = 0
            i = idx_sort[i_tmp] 
            x_, y_ = x_tmp[i], y_tmp[i]
            for idx_split, split in enumerate(feat_split_point_values):
                while x_ <= split:
                    if settings.verbose >= 4: 
                        print('x_ = %s, split = %s, idx_split = %s, max_idx_split = %s' % \
                                (x_, split, idx_split, max_idx_split))
                    if settings.optype == 'class':
                        cnt_left[y_] += 1
                        cnt_right[y_] -= 1
                    else:
                        y_2 = y_ ** 2
                        sum_y_left += y_
                        sum_y_right -= y_
                        sum_y2_left += y_2
                        sum_y2_right -= y_2
                        n_points_left += 1
                        n_points_right -= 1
                    i_tmp += 1
                    i = idx_sort[i_tmp]
                    x_, y_ = x_tmp[i], y_tmp[i]
                # break out now and compute the value of the split, Note: there could be multiple splits using the same likelihood
                if settings.optype == 'class':
                    loglik_left = compute_dirichlet_normalizer_fast(cnt_left, cache)
                    loglik_right = compute_dirichlet_normalizer_fast(cnt_right, cache)
                else:
                    loglik_left = compute_normal_normalizer(sum_y_left, sum_y2_left, \
                                        n_points_left, param, cache, settings)[0]
                    loglik_right = compute_normal_normalizer(sum_y_right, sum_y2_right, \
                                        n_points_right, param, cache, settings)[0]
                logprior = log_psplit + log_prob_feat[feat_id] + np.log(prob_split_prior[idx_split])
                logprob = (loglik_left + loglik_right - loglik_nosplit) + logprior
                if settings.debug == 1:
                    assert(not np.isnan(logprob))
                if settings.verbose >= 3:
                    print('feat_id = %3d, split = %f, logprob = %f' % \
                        (feat_id, split, logprob))
                logprob_tmp[idx_split] = logprob
                logprior_tmp[idx_split] = logprior
            assert (i_tmp <= n_max_idx_sort)
            logprob_thresholds_per_feat[feat_id] = [logprob_tmp, logprior_tmp, feat_split_point_values] 
            logprob_values[n_feat_id] = logsumexp(logprob_tmp)
            if settings.debug == 1:
                try:
                    assert not np.isnan(logprob_values[n_feat_id])
                except:
                    print('something is fishy in posterior_proposal_all')
                    print('logprior_values = \n%s\nlogprob_values = \n%s', (logprior_values, logprob_values))
                    raise AssertionError
        sum_logprob = logsumexp(logprob_values)
        return (logprob_thresholds_per_feat, logprob_values, sum_logprob, feat_id_perm, split_not_supported)

    def process_node_id(self, data, param, settings, cache, node_id):
        if self.do_not_split[node_id]:
            log_sis_ratio = 0.0
        else:
            log_psplit = np.log(self.compute_psplit(node_id, param))
            train_ids = self.train_ids[node_id]
            left, right = get_children_id(node_id)
            if settings.verbose >= 4:
                print('train_ids for this node = %s' % train_ids)
            if settings.proposal == 'posterior':
                (do_not_split_node_id, feat_id_chosen, split_chosen, idx_split_global, log_sis_ratio, logprior_nodeid, \
                        train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) \
                        = self.posterior_proposal(data, param, settings, cache, node_id, train_ids, log_psplit)
            else:
                (do_not_split_node_id, feat_id_chosen, split_chosen, idx_split_global, log_sis_ratio, logprior_nodeid, \
                        train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) \
                        = self.precomputed_proposal(data, param, settings, cache, node_id, train_ids, log_psplit)
            if do_not_split_node_id:
                self.do_not_split[node_id] = True
            else:
                self.update_left_right_statistics(cache_tmp, node_id, logprior_nodeid, train_ids_left,\
                    train_ids_right, loglik_left, loglik_right, feat_id_chosen, split_chosen, \
                    idx_split_global, settings, param, data, cache)
                # feat_id_chosen, split_chosen, logprior_node_id assumed to be known
                if settings.optype == 'class':
                    self.counts.pop(node_id)
                else:
                    self.sum_y.pop(node_id)
                    self.sum_y2.pop(node_id)
                    self.n_points.pop(node_id)
                    self.param_n.pop(node_id)
                self.train_ids.pop(node_id)
                self.loglik.pop(node_id)        # can't pop logprior though
                if settings.grow == 'next':
                # not checking for do_not_split criterion here; checking it in grow_next instead
                    self.grow_nodes.append(left)
                    self.grow_nodes.append(right)
        return (log_sis_ratio)

    def grow_next(self, data, param, settings, cache):
        """ grows just one node at a time unlike grow_layer
            breaks after processing the first non do_not_grow node or when grow_nodes is empty
            Note that multiple nodes could be killed in a single grow_next call
        """
        do_not_grow = True
        log_sis_ratio = 0.0
        num_nodes_processed = 0
        if not self.grow_nodes:
            if settings.verbose >= 2:
                print('None of the leaves can be grown any further: Current' \
                    'depth = %3d, Skipping grow_next' % self.depth)
        else:
            while True:
                # loop through current leaf nodes, process first "non do_not_grow" node and break; 
                # if none of the nodes can be processed, do_not_grow = True
                if settings.priority == 'loglik':
                    # pick the node with the lowest marginal loglik
                    remove_position = np.argmin(np.array([self.loglik[n_id] for n_id in self.grow_nodes]))
                else:
                    # just pop the oldest node
                    remove_position = 0
                node_id = self.grow_nodes.pop(remove_position)
                do_not_grow = do_not_grow and self.do_not_split[node_id]
                if settings.include_child_prob == 1:
                    log_sis_ratio -= np.log(self.compute_pnosplit(node_id, param))
                if self.do_not_split[node_id]:
                    if settings.verbose >= 3:
                        print('Skipping split at node_id %3d' % node_id)
                    if not self.grow_nodes:
                        break
                else:
                    num_nodes_processed += 1
                    log_sis_ratio += self.process_node_id(data, param, settings, cache, node_id)
                    break           # you have processed a non do_not_grow node, take a break!
            self.loglik_current = self.compute_loglik()
        self.log_sis_ratio = log_sis_ratio
        self.do_not_grow = do_not_grow
        self.num_nodes_processed_itr.append(num_nodes_processed)
    
    def grow_layer(self, data, param, settings, cache):
        self.log_sis_ratio = 0.0        # needs to be reset everytime
        num_nodes_processed = 0
        if self.do_not_grow:
            if settings.verbose >= 2:
                print('None of the leaves can be grown any further: Current' \
                    'depth = %3d, Skipping grow_next' % self.depth)
            self.num_nodes_processed_itr.append(num_nodes_processed)
            return     
        self.depth += 1
        if settings.verbose >=3:
            print('growing leaves at depth = %d' % self.depth)
        do_not_grow = True
        current_leaf_nodes = self.leaf_nodes[:]  # self.leaf_nodes is updated inside the loop
        if settings.verbose >= 3:
            print('\nCurrent leaf nodes = %s' % current_leaf_nodes)
        for node_id in current_leaf_nodes:
            do_not_grow = do_not_grow and self.do_not_split[node_id]
            if settings.verbose >= 3:
                print('\nprocessing node_id %3d' % node_id)
            # adjust log_sis_ratio before the continue statement
            if settings.include_child_prob == 1:
                self.log_sis_ratio -= np.log(self.compute_pnosplit(node_id, param))
            if (self.do_not_split[node_id]): 
                if settings.verbose >= 3:
                    print('Skipping split at node_id %3d' % node_id)
                continue
            num_nodes_processed += 1
            log_sis_ratio = self.process_node_id(data, param, settings, cache, node_id)
            self.log_sis_ratio += log_sis_ratio     # aggregate over all nodes that are processed
            if settings.verbose >= 2:
                print('Tree after processing current node')
                self.print_tree()
        if settings.verbose >= 2:
            print('Tree after processing all "grow" nodes')
            self.print_tree()
            print('-'*40)
        self.loglik_current = self.compute_loglik()
        self.do_not_grow = do_not_grow
        self.num_nodes_processed_itr.append(num_nodes_processed)        # very different from len(current_leaf_nodes)


def compute_average_particle_stats(particles, log_weights):
    """ computes average statistics of the particles """
    particle_stats = np.zeros(5)   # doesn't store test performance ... use record_particle_stats if you want that
    for n, p in enumerate(particles):
        particle_stats += np.array([p.depth, len(p.leaf_nodes), \
            len(p.non_leaf_nodes), p.compute_logprob(), \
            log_weights[n]])
    n += 1
    particle_stats /= n
    return particle_stats


def record_particle_stats(p, param, data, settings, log_weight=0.0):
    if settings.store_history >= 1:
        metrics = evaluate_performance_tree(p, param, data, settings, data['x_test'], data['y_test'])
        acc, log_prob = metrics['acc'], metrics['log_prob']
    else:
        acc, log_prob = 0, -np.inf
    op = np.array([p.depth, len(p.leaf_nodes), \
            len(p.non_leaf_nodes), p.compute_logprob(), \
            log_weight, acc, log_prob])
    return op


def record_particle_stats_all(particles, param, data, settings, log_weights):
    particle_stats = np.zeros((7, len(particles)))
    for n, p in enumerate(particles):
        particle_stats[:, n] = record_particle_stats(p, param, data, settings, log_weights[n])
    return particle_stats


def update_particle_weights(particles, log_weights, settings, data, param):
    for n, p in enumerate(particles):
        log_weights[n] += p.log_sis_ratio
    ess_islands = np.ones(settings.n_islands)
    log_pd_islands = np.ones(settings.n_islands)
    prob_islands = {}
    n_particles_tmp = settings.n_particles // settings.n_islands
    ess = 1.0
    for i_ in range(settings.n_islands):
        pid_min, pid_max = i_ * n_particles_tmp, (i_ + 1) * n_particles_tmp - 1
        prob_tmp = softmax(log_weights[pid_min: pid_max+1])
        ess_tmp = 1 / np.sum(prob_tmp ** 2) / n_particles_tmp
        ess_islands[i_] = ess_tmp
        prob_islands[i_] = prob_tmp
        ess = min(ess_tmp, ess)
        log_pd_islands[i_] = logsumexp(log_weights[pid_min:pid_max+1])
    log_pd = logsumexp(log_pd_islands) - np.log(settings.n_islands)     # average over all the islands
    if (settings.store_history == 2) or (ess <= settings.ess_threshold):
        particle_stats_current = record_particle_stats_all(particles, param, data, settings, log_weights)
    else:
        particle_stats_current = None
    return (log_pd, ess, particle_stats_current, log_weights, ess_islands, prob_islands, log_pd_islands)


def resample(particles, log_weights, settings, data, log_pd, ess, ess_islands, prob_islands, log_pd_islands):
    if (ess <= settings.ess_threshold):
        if settings.demo == 1 and settings.n_islands == 1:
            prob = softmax(log_weights)
            print('ess = %s, ' % (ess, prob))
            title_text = 'before resampling'
            plot_particles(particles, data, prob, title_text)
        pid_list = []
        n_particles_tmp = settings.n_particles // settings.n_islands
        pid_list_range_tmp = list(range(n_particles_tmp))
        for i_ in range(settings.n_islands):
            pid_min, pid_max = i_ * n_particles_tmp, (i_ + 1) * n_particles_tmp - 1
            pid_range_tmp = list(range(pid_min, pid_max+1))
            ess_tmp = ess_islands[i_]
            if (ess_tmp <= settings.ess_threshold):
                prob_tmp = prob_islands[i_]
                log_pd_tmp = log_pd_islands[i_]
                pid_list_tmp = resample_pids_basic(settings, n_particles_tmp, prob_tmp)
                #    random.shuffle(pid_list_tmp)    # shuffle so that particle is assigned randomly
                #                                      shuffling doesn't affect results, but messes up ancestry plots 
                log_weights[pid_range_tmp] = np.ones(n_particles_tmp) * (log_pd_tmp - np.log(n_particles_tmp)) 
            else:
                pid_list_tmp = pid_list_range_tmp
            for pid in pid_list_tmp:
                pid_list.append(pid + pid_min)
        if settings.verbose >= 2:
            print('ess = %s, ess_threshold = %s' % (ess, settings.ess_threshold))
            print('new particle ids = ')
            print(pid_list)
        assert(len(particles) == settings.n_particles)
        op = create_new_particles(particles, pid_list, settings)
        if settings.verbose >= 2:
            print('particle stats: after resampling (depth, leaf_nodes, non_leaf_nodes, logprob, log-weights): %s' % \
                compute_average_particle_stats(op, log_weights))
        if settings.demo == 1:
            title_text = 'after resampling'
            plot_particles(op, data, softmax(log_weights), title_text)
    else:
        op = particles
        pid_list = list(range(settings.n_particles))
    # update ancestry
    for pid, p in zip(pid_list, op):
        p.ancestry.append(pid)
    if settings.verbose >= 2:
        print('pid_list = %s' % pid_list)
        for pid, p in enumerate(op):
            print('pid = %s, ancestry = %s' % (pid, p.ancestry))
    if settings.verbose >= 2:
        print('\nNew version')
        op[0].print_tree()
        print('\n')
    return (op, log_weights)


def resample_pids_basic(settings, n_particles, prob):
    if settings.resample == 'multinomial':
        indices = np.random.multinomial(n_particles, prob, size=1)
        pid_list = [pid for pid, cnt in enumerate(indices.flat) \
                    for n in range(cnt)]
    elif settings.resample == 'systematic':
        pid_list = systematic_sample(n_particles, prob)
    return pid_list


def resample_pids(settings, log_weights):
    pid_list = []
    n_particles_tmp = settings.n_particles // settings.n_islands
    for i_ in range(settings.n_islands):
        pid_min, pid_max = i_ * n_particles_tmp, (i_ + 1) * n_particles_tmp - 1
        prob_tmp = softmax(log_weights[pid_min: pid_max+1])
        pid_list_tmp = resample_pids_basic(settings, n_particles_tmp, prob_tmp)
        for pid in pid_list_tmp:
            try:
                assert(pid + pid_min <= settings.n_particles)
            except AssertionError:
                print(i_, n_particles_tmp, pid_min, pid, settings.n_particles)
            pid_list.append(pid + pid_min)
    return pid_list
        

def create_new_particles(particles, pid_list, settings):
    """ particles that occur just once after resampling are not 'copied' """
    list_allocated = set([])
    op = []
    for i, pid in enumerate(pid_list):
        if pid not in list_allocated:
            op.append(particles[pid])
        else:
            op.append(copy_particle(particles[pid], settings))
        list_allocated.add(pid)
    return op


def dumb_copy_particles(particles, settings):
    op = []
    for p in particles:
        op.append(copy_particle(p, settings))
    return op


def copy_particle(p, settings):
    #op = Particle(p.pid)
    op = Particle()
    # lists
    op.leaf_nodes = p.leaf_nodes[:]
    op.non_leaf_nodes = p.non_leaf_nodes[:]
    op.ancestry = p.ancestry[:]
    op.num_nodes_processed_itr = p.num_nodes_processed_itr[:]
    try:
        op.grow_nodes = p.grow_nodes[:]
    except AttributeError:  # for layer-wise expansion
        pass
    # dictionaries
    op.do_not_split = p.do_not_split.copy()
    if settings.optype == 'class':
        op.counts = p.counts.copy()
    else:
        op.sum_y = p.sum_y.copy()
        op.sum_y2 = p.sum_y2.copy()
        op.n_points = p.n_points.copy()
        op.param_n = p.param_n.copy()
    op.train_ids = p.train_ids.copy()
    op.node_info = p.node_info.copy()
    op.loglik = p.loglik.copy()
    op.logprior = p.logprior.copy()
    # other variables
    op.depth = copy(p.depth)
    op.do_not_grow = copy(p.do_not_grow)
    op.loglik_current = copy(p.loglik_current)
    return op


def systematic_sample(n, prob):
    """ systematic re-sampling algorithm.
    Note: objects with > 1/n probability (better than average) are guaranteed to occur atleast once.
    see section 2.4 of 'Comparison of Resampling Schemes for Particle Filtering' by Douc et. al for more info.
    """
    assert(n == len(prob))
    assert(abs(np.sum(prob) - 1) < 1e-10)
    cum_prob = np.cumsum(prob)
    u = np.random.rand(1) / float(n)
    i = 0
    indices = []
    while True:
        while u > cum_prob[i]:
            i += 1
        indices.append(i)
        u += 1/float(n)
        if u > 1:
            break
    return indices


def evaluate_predictions_smc(particles, data, x, y, settings, param, weights):
    if settings.optype == 'class':
        pred_prob_overall = np.zeros((x.shape[0], data['n_class']))
    else:
        pred_prob_overall = np.zeros(x.shape[0])
        pred_mean_overall = np.zeros(x.shape[0])
    if settings.weight_predictions:
        weights_norm = weights
    else:
        weights_norm = np.ones(settings.n_particles) // settings.n_particles
    assert(np.abs(np.sum(weights_norm) - 1) < 1e-3)
    if settings.verbose >= 2:
        print('weights_norm = ')
        print(weights_norm)
    for pid, p in enumerate(particles):
        pred_all = evaluate_predictions_fast(p, x, y, data, param, settings)
        pred_prob = pred_all['pred_prob']
        pred_prob_overall += weights_norm[pid] * pred_prob
        if settings.optype == 'real':
            pred_mean_overall += weights_norm[pid] * pred_all['pred_mean']
    if settings.debug == 1:
        check_if_zero(np.mean(np.sum(pred_prob_overall, axis=1) - 1))
    if settings.optype == 'class':
        metrics = compute_test_metrics_classification(y, pred_prob_overall)
    else:
        metrics = compute_test_metrics_regression(y, pred_mean_overall, pred_prob_overall)
    if settings.verbose >= 1:
        if settings.optype == 'class':
            print('Averaging over all particles, accuracy = %f, log predictive = %f' % (metrics['acc'], metrics['log_prob']))
        else:
            print('Averaging over all particles, mse = %f, log predictive = %f' % (metrics['mse'], metrics['log_prob']))
    return (pred_prob_overall, metrics)


def init_smc(data, settings):
    param, cache, cache_tmp = precompute(data, settings)
    particles = [Particle(list(range(data['n_train'])), param, settings, cache_tmp) \
            for n in range(settings.n_particles)]
    if settings.include_child_prob == 1:
        log_weights = np.array([(p.loglik[0] + np.log(p.compute_pnosplit(0, param))) for p in particles]) \
                        - np.log(settings.n_particles) + np.log(settings.n_islands)
        # need to incorporate loglik[0] as well as log(pnosplit(0)) here (will cancel out later)
    else:
        log_weights = np.array([p.loglik[0] for p in particles]) - np.log(settings.n_particles) + np.log(settings.n_islands)
        # need to incorporate loglik[0] here (will cancel out later)
    if settings.debug == 1:
        log_weights_old = np.array([p.loglik_current for p in particles]) - np.log(settings.n_particles) + np.log(settings.n_islands)
        assert np.abs(np.sum(log_weights_old - log_weights)) < 1e-3
    # pre-compute stuff
    feat_score_cumsum = {}      # cumsum of scores of each interval
    for feat_id in cache['range_n_dim']:
        x_tmp = data['x_train'][:, feat_id]
        idx_sort = np.argsort(x_tmp)
        feat_unique_values = np.unique(x_tmp[idx_sort])
        n_unique = len(feat_unique_values)
        # first "interval" has width 0 since points to the left of that point are chosen with prob 0
        feat_score_tmp = np.zeros(n_unique)
        diff_feat_unique_values = np.diff(feat_unique_values)
        log_diff_feat_unique_values_norm = np.log(diff_feat_unique_values) \
                            - np.log(feat_unique_values[-1] - feat_unique_values[0])
        if settings.proposal == 'prior':
            feat_score_tmp[1:] = diff_feat_unique_values
        elif settings.proposal == 'empirical':
            feat_score_tmp[1:] = 1 / np.float(n_unique - 1)
        elif settings.proposal == 'posterior':
            pass
        else:
            raise Exception
        feat_score_cumsum[feat_id] = np.cumsum(feat_score_tmp)
    if settings.proposal != 'posterior':
        cache['feat_score_cumsum'] = feat_score_cumsum
    return (particles, param, log_weights, cache, cache_tmp)


def run_smc(particles, data, settings, param, log_weights, cache):
    log_weights_itr = np.zeros((settings.max_iterations+1, settings.n_particles))
    ess_itr = np.ones(settings.max_iterations+1)
    particle_stats_itr_d = {}           # _d suffix to denote a dictionary 
    particles_itr_d = {}           # _d suffix to denote a dictionary 
    for itr in range(settings.max_iterations):  # iteration is counted as the 1st iteration wrt log_weights
        if settings.verbose >= 3:
            print('\n')
            print('*'*80)
            print('Current iteration = %3d' % itr)
            print('*'*80)
        if itr != 0:
            # everything initialized with empty tree and weights are equal: no resampling required here
            if settings.verbose >= 1:
                print('iteration = %3d, log p(y|x) = %.2f, ess/n_particles = %f'  % (itr, log_pd, ess))
            (particles, log_weights) = resample(particles, log_weights, settings, data, log_pd, \
                                                    ess, ess_islands, prob_islands, log_pd_islands)
        for pid, p in enumerate(particles):
            if settings.verbose >= 2:
                print('Current particle = %3d' % pid)
            if settings.grow == 'next':
                p.grow_next(data, param, settings, cache)
            elif settings.grow == 'layer':
                p.grow_layer(data, param, settings, cache)
        (log_pd, ess, particle_stats_current, log_weights, ess_islands, prob_islands, log_pd_islands) = \
                    update_particle_weights(particles, log_weights, settings, data, param)     # in place update of log_weights
        log_weights_itr[itr+1, :] = log_weights.copy()
        ess_itr[itr+1] = ess
        if settings.store_history >= 1:
            particles_itr_d[itr+1] = dumb_copy_particles(particles, settings)
            if (settings.store_history == 2) or (ess <= settings.ess_threshold) or (check_do_not_grow(particles)) or (itr == settings.max_iterations - 1):
                if particle_stats_current is None:
                    particle_stats_current = record_particle_stats_all(particles, param, data, settings, log_weights)
                particle_stats_itr_d[itr+1] = particle_stats_current
        if check_do_not_grow(particles):
            log_weights_itr = log_weights_itr[:itr+2,]
            ess_itr = ess_itr[:itr+2]
            print('None of the particles can be grown any further; breaking out')
            break
    create_prediction_trees(particles, param, data, settings)
    return (particles, ess_itr, log_weights_itr, log_pd, particle_stats_itr_d, particles_itr_d, log_pd_islands)


def create_prediction_trees(particles, param, data, settings):
    for p in particles:
        p.create_prediction_tree(param, data, settings)


def init_run_smc(data, settings):
    (particles, param, log_weights, cache, cache_tmp) = init_smc(data, settings)
    (particles, ess_itr, log_weights_itr, log_pd, particle_stats_itr_d, particles_itr_d) = \
            run_smc(particles, data, settings, param, log_weights, cache)
    return (particles, log_pd, log_weights_itr[-1, :], param, cache, cache_tmp)


def check_do_not_grow(particles):
    """ Test if all particles have grown fully """
    do_not_grow = True
    for p in particles:
        do_not_grow = do_not_grow and p.do_not_grow
    return do_not_grow


def generate_all_trees(data, param, settings, cache, p):
    # this script is used to generate all the trees using brute force enumeration on tiny toy dataset
    # didn't optimize this code since this is not used for any experiments
    complete_trees = []
    partial_trees = [p]
    while partial_trees:
        p = partial_trees.pop(0)
        if settings.debug == 1 and settings.verbose >= 4:
            print('-'*80)
            print('current partial tree')
            p.print_tree()
            #print '-'*80
        if not p.grow_nodes:    # base case: tree is complete now
            if settings.debug == 1 and settings.verbose >= 2:
                print('-'*40)
                print('created new complete tree')
                p.print_tree()
                assert not np.isnan(p.compute_logprior())
            complete_trees.append(p)
        else: 
            node_id = p.grow_nodes.pop(0)
            if not p.do_not_split[node_id]:
                train_ids = p.train_ids[node_id]
                n_train_ids = len(train_ids)
                pnosplit = p.compute_pnosplit(node_id, param) 
                log_psplit = np.log(p.compute_psplit(node_id, param))
                feat_id_valid, score_feat, feat_split_info, split_not_supported = \
                        p.find_valid_dimensions(data, cache, train_ids, settings)
                if settings.verbose >= 4:
                    print('node_id = %s' % node_id)
                    print('train_ids = %s' % train_ids)
                feat_id_perm, n_feat_subset, log_prob_feat = subsample_features(settings, \
                        feat_id_valid, score_feat, split_not_supported)
                if settings.verbose >= 4:
                    print('training data at current node:')
                    print(data['x_train'][train_ids, :])
                if not split_not_supported:
                    for feat_id_chosen in feat_id_valid:
                        idx_min, idx_max, x_min, x_max, feat_score_cumsum_prior_current = \
                                feat_split_info[feat_id_chosen] 
                        z_prior = feat_score_cumsum_prior_current[idx_max] - \
                                    feat_score_cumsum_prior_current[idx_min]
                        prob_split_prior = np.diff(feat_score_cumsum_prior_current[idx_min: idx_max+1] - \
                                            feat_score_cumsum_prior_current[idx_min])/ z_prior
                        for idx_split_chosen, prob_split_prior_tmp in enumerate(prob_split_prior):
                            if prob_split_prior_tmp < 1e-15:
                                continue
                            idx_split_global = idx_split_chosen + idx_min + 1
                            split_chosen = cache['feat_idx2midpoint'][feat_id_chosen][idx_split_global]
                            if settings.debug == 1:
                                try:
                                    assert(split_chosen > x_min)
                                except AssertionError:
                                    print('split_chosen <= x_min')
                                    print(prob_split_prior, feat_score_cumsum_current[idx_min:idx_max+1])
                                    raise Exception
                            logprior_nodeid_tau = np.log(prob_split_prior[idx_split_chosen])
                            logprior_nodeid = log_psplit + logprior_nodeid_tau \
                                                + log_prob_feat[feat_id_chosen]
                            (train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) = \
                                compute_left_right_statistics(data, param, cache, train_ids, \
                                    feat_id_chosen, split_chosen, settings)
                            p_new = copy_particle(p, settings)
                            left, right = get_children_id(node_id)
                            p_new.logprior[node_id] = logprior_nodeid
                            p_new.node_info[node_id] = [feat_id_chosen, split_chosen, idx_split_global]
                            p_new.loglik[left] = loglik_left
                            p_new.loglik[right] = loglik_right
                            p_new.train_ids[left] = train_ids_left
                            p_new.train_ids[right] = train_ids_right
                            p_new.do_not_split[left] = stop_split(train_ids_left, settings, data, cache)
                            p_new.do_not_split[right] = stop_split(train_ids_right, settings, data, cache)
                            if p_new.do_not_split[left]:
                                p_new.logprior[left] = 0.0
                            else:
                                p_new.logprior[left] = np.log(p_new.compute_pnosplit(left, param))
                            if p_new.do_not_split[right]:
                                p_new.logprior[right] = 0.0
                            else:
                                p_new.logprior[right] = np.log(p_new.compute_pnosplit(right, param))
                            if settings.optype == 'class':
                                p_new.counts[left] = cache_tmp['cnt_left_chosen']
                                p_new.counts[right] = cache_tmp['cnt_right_chosen']
                                p_new.counts.pop(node_id)
                            else:
                                p_new.sum_y[left] = cache_tmp['sum_y_left']
                                p_new.sum_y2[left] = cache_tmp['sum_y2_left']
                                p_new.n_points[left] = cache_tmp['n_points_left']
                                p_new.param_n[left] = cache_tmp['param_left']
                                p_new.sum_y[right] = cache_tmp['sum_y_right']
                                p_new.sum_y2[right] = cache_tmp['sum_y2_right']
                                p_new.n_points[right] = cache_tmp['n_points_right']
                                p_new.param_n[right] = cache_tmp['param_right']
                            p_new.train_ids.pop(node_id)
                            p_new.loglik.pop(node_id)        # can't pop logprior though
                            p_new.non_leaf_nodes.append(node_id)
                            p_new.grow_nodes.append(left)
                            p_new.grow_nodes.append(right)
                            if settings.verbose >= 4:
                                print('-'*40)
                                print('created new partial tree')
                                p_new.print_tree()
                            partial_trees.append(p_new)
                            assert not np.isnan(p_new.compute_logprior())
            #case where this node_id is not split
            p_new = copy_particle(p, settings)
            p_new.leaf_nodes.append(node_id)
            p_new.depth = max(p_new.depth, get_depth(node_id))
            if p_new.do_not_split[node_id] or split_not_supported:
                p_new.logprior[node_id] = 0.0
                # was set inside the loop -- cannot split, hence logprior_nodeid = 0
            partial_trees.append(p_new)
            if settings.verbose >= 4:
                print('-'*40)
                print('created new partial tree')
                p_new.print_tree()
            assert not np.isnan(p_new.compute_logprior())
    return complete_trees


def generate_all_trees_posterior(data, param, settings, cache, p):
    # this script is used to generate all the trees using brute force enumeration on tiny toy dataset
    # didn't optimize this code since this is not used for any experiments
    complete_trees = []
    partial_trees = [p]
    p.log_weight = p.loglik_current
    while partial_trees:
        p = partial_trees.pop(0)
        if settings.verbose >= 4:
            print('-'*80)
            print('current partial tree')
            p.print_tree()
            #print '-'*80
        if not p.grow_nodes:    # base case: tree is complete now
            if settings.verbose >= 4:
                print('-'*40)
                print('created new complete tree')
                p.print_tree()
                assert not np.isnan(p.log_weight)
            complete_trees.append(p)
        else: 
            node_id = p.grow_nodes.pop(0)
            logprob_thresholds_per_feat, logprob_values, sum_logprob, feat_id_perm, split_not_supported \
                    = p.posterior_proposal_all(data, param, settings, cache, node_id)
            if not p.do_not_split[node_id]:
                train_ids = p.train_ids[node_id]
                if not split_not_supported:
                    for feat_id_chosen in feat_id_perm:
                        logprob_tmp, logprior_tmp, feat_split_point_values =\
                                logprob_thresholds_per_feat[feat_id_chosen]
                        for idx_split_chosen, split_chosen in enumerate(feat_split_point_values):
                            logprior_nodeid = logprior_tmp[idx_split_chosen]
                            (train_ids_left, train_ids_right, cache_tmp, loglik_left, loglik_right) = \
                                compute_left_right_statistics(data, param, cache, train_ids, feat_id_chosen, \
                                    split_chosen, settings)
                            p_new = copy_particle(p, settings)
                            p_new.log_weight = p.log_weight + logprob_tmp[idx_split_chosen]
                            try:
                                assert not np.isnan(p_new.log_weight)
                            except AssertionError:
                                print('problem in log_weight computation:')
                                print('feat_split_point_values = \n%s' % feat_split_point_values)
                                print('logprior_tmp =\n%s\nlogprob_tmp=\n%s' % (logprior_tmp, logprob_tmp))
                                raise AssertionError
                            left, right = get_children_id(node_id)
                            p_new.logprior[node_id] = logprior_nodeid
                            p_new.node_info[node_id] = [feat_id_chosen, split_chosen, idx_split_chosen]
                            p_new.loglik[left] = loglik_left
                            p_new.loglik[right] = loglik_right
                            p_new.train_ids[left] = train_ids_left
                            p_new.train_ids[right] = train_ids_right
                            p_new.do_not_split[left] = stop_split(train_ids_left, settings, data, cache)
                            p_new.do_not_split[right] = stop_split(train_ids_right, settings, data, cache)
                            if p_new.do_not_split[left]:
                                p_new.logprior[left] = 0.0
                            else:
                                p_new.logprior[left] = np.log(p_new.compute_pnosplit(left, param))
                            if p_new.do_not_split[right]:
                                p_new.logprior[right] = 0.0
                            else:
                                p_new.logprior[right] = np.log(p_new.compute_pnosplit(right, param))
                            if settings.optype == 'class':
                                p_new.counts[left] = cache_tmp['cnt_left_chosen']
                                p_new.counts[right] = cache_tmp['cnt_right_chosen']
                                p_new.counts.pop(node_id)
                            else:
                                p_new.sum_y[left] = cache_tmp['sum_y_left']
                                p_new.sum_y2[left] = cache_tmp['sum_y2_left']
                                p_new.n_points[left] = cache_tmp['n_points_left']
                                p_new.param_n[left] = cache_tmp['param_left']
                                p_new.sum_y[right] = cache_tmp['sum_y_right']
                                p_new.sum_y2[right] = cache_tmp['sum_y2_right']
                                p_new.n_points[right] = cache_tmp['n_points_right']
                                p_new.param_n[right] = cache_tmp['param_right']
                            p_new.train_ids.pop(node_id)
                            p_new.loglik.pop(node_id)        # can't pop logprior though
                            p_new.non_leaf_nodes.append(node_id)
                            p_new.grow_nodes.append(left)
                            p_new.grow_nodes.append(right)
                            if settings.verbose >= 4:
                                print('-'*40)
                                print('created new partial tree')
                                p_new.print_tree()
                            partial_trees.append(p_new)
                            assert not np.isnan(p_new.log_weight)
            #case where this node_id is not split
            p_new = copy_particle(p, settings)
            p_new.log_weight = p.log_weight + logprob_values[-1]
            p_new.leaf_nodes.append(node_id)
            p_new.depth = max(p_new.depth, get_depth(node_id))
            if p_new.do_not_split[node_id] or split_not_supported:
                p_new.logprior[node_id] = 0.0   # was set inside the loop -- cannot split, hence logprior_nodeid = 0
            partial_trees.append(p_new)
            if settings.verbose >= 4:
                print('-'*40)
                print('created new partial tree')
                p_new.print_tree()
            assert not np.isnan(p_new.log_weight)
    return complete_trees


def get_particle_positions(ancestry, pid, n_iter, itr_resampling):
    pos = np.ones(n_iter) * pid
    assert(len(ancestry) == len(itr_resampling))
    for itr, parent in zip(itr_resampling[-1::-1], ancestry[-1::-1]):
        assert(itr != 0)
        pos[itr::-1] = parent
    return pos


def postmortem_results(data, settings, particles, filename, ess_itr, log_weights_itr, \
                        particle_stats_itr_d, particles_itr_d):
        """ check ancestry of particles, check evolution of test accuracy & particle statistics 
            across time
            NOTE: need to have store_history >=1 for this to be possible """
        filename_smc_ancestry_tag = filename[:-2] + '.ancestry.'
        filename_smc_ancestry = filename_smc_ancestry_tag + 'txt'
        print('filename_smc_ancestry = %s' % filename_smc_ancestry)
        f = open(filename_smc_ancestry, 'w')
        print('\n', file=f)
        print('*'*30 + ' particle ancestries ' + '*'*30, file=f)
        print('printing stats only when resampling occurred', file=f)
        n_iter = len(ess_itr)
        assert(n_iter == log_weights_itr.shape[0])
        itr_resampling = [itr for itr in range(1, n_iter - 1) if ess_itr[itr] <= settings.ess_threshold]
        try:
            itr_resampling.remove(n_iter)
        except ValueError:
            pass
        print('iterations where resampling occurred = %s' % itr_resampling)
        itr_print = sorted(particle_stats_itr_d.keys())
        print('iterations where particle_stats was stored = %s' % itr_print)
        print('iterations where resampling occurred = %s' % itr_resampling, file=f)
        parent_last_resample = set([])
        for pid, p in enumerate(particles):
            #print >> f, 'pid = %5d, ancestry {itr: parent_id} = %s' % (pid, ', '.join([str(i) + ':' + str(x) for i, x in enumerate(p.ancestry)]))
            print('pid = %5d, ancestry {itr: parent_id} = %s' % (pid, \
                    ', '.join([str(i) + ':' + str(p.ancestry[i]) for i in itr_resampling])), file=f)
            if itr_resampling:
                parent_last_resample.add(p.ancestry[itr_resampling[-1]])
        print('parent_last_resample = %s' % parent_last_resample, file=f)
        if len(parent_last_resample) == 1:
            print('WARNING: all the final particles share ancestry from start till itr = %s' \
                        % itr_resampling[-1], file=f)
        print('parent stats (just before resampling)', file=f)
        for itr_tmp in range(1, n_iter - 1): 
            if (settings.store_history == 2) or (ess_itr[itr_tmp] <= settings.ess_threshold):
                print('\niteration = %5d' % (itr_tmp), file=f)
                particle_stats_current = particle_stats_itr_d[itr_tmp]
                for pid in range(particle_stats_current.shape[1]):
                    # this pid loops over particle stats at itr_tmp (the non-resampled ones will not survive till the end
                    # these stats are useful only to check the alternative options to the current ancestry
                    max_log_weight = np.max(particle_stats_current[4, :])
                    print('orig_pid = %d, log_weight (-max) = %.2f, depth = %d, num_leaf = %d, num_nonleaf = %d, log_prob = %.2f' \
                            % (pid, particle_stats_current[4, pid] - max_log_weight, \
                                particle_stats_current[0, pid], particle_stats_current[1, pid], \
                                particle_stats_current[2, pid], particle_stats_current[3, pid]), file=f)
        if settings.n_particles <= 50:
            fig_height = 15
        else:
            fig_height = 30
        if n_iter > 150:
            fig_width = 60
        else:
            fig_width = 30
        plt.figure(figsize = (fig_width,fig_height))
        plt.hold(True)
        plt.subplot(311)
        plt.hold(True)
        weights_itr = np.zeros(log_weights_itr.shape)   # iterations x particles
        logprob_test_itr = np.zeros(log_weights_itr.shape)   # iterations x particles
        acc_test_itr = np.zeros(log_weights_itr.shape)   # iterations x particles
        dead_itr = np.zeros(log_weights_itr.shape)   # iterations x particles
        for itr in range(log_weights_itr.shape[0]):
            weights_itr[itr, :] = softmax(log_weights_itr[itr, :])
        for itr in itr_print:
            tmp = particle_stats_itr_d[itr]
            logprob_test_itr[itr, :] = tmp[-1, :]
            acc_test_itr[itr, :] = tmp[-2, :]
            tmp_particles = particles_itr_d[itr]
            for pid, p in enumerate(tmp_particles):
                dead_itr[itr, pid] = p.do_not_grow
        itr_vec = list(range(n_iter))
        itr_print_2 = itr_print
        max_logprob_test_itr = np.max(logprob_test_itr, 1)
        logprob_test_itr_normalized = np.zeros(logprob_test_itr.shape)
        for pid in range(settings.n_particles):
            logprob_test_tmp = logprob_test_itr[:, pid]
            logprob_test_itr_normalized[:, pid] = logprob_test_tmp - max_logprob_test_itr
        vmin_logprob_test = np.min(logprob_test_itr_normalized)
        vmax_logprob_test = np.max(logprob_test_itr_normalized)
        print(max_logprob_test_itr.shape)
        for pid, p in enumerate(particles):
            plt.subplot(311)
            plt.scatter(itr_print_2, pid * np.ones(len(itr_print_2)), s=40*weights_itr[itr_print_2, pid], edgecolor='none')
            plt.subplot(312)
            if settings.n_particles >= 25:
                marker_size = 10
            else:
                marker_size = 40
            itr_dead = [itr for itr in itr_print if dead_itr[itr, pid]] 
            itr_not_dead = [itr for itr in itr_print if not dead_itr[itr, pid]] 
            plt.scatter(itr_dead, pid * np.ones(len(itr_dead)), c=logprob_test_itr_normalized[itr_dead, pid], s=marker_size, marker='^', edgecolor='none', \
                    vmin=vmin_logprob_test, vmax=vmax_logprob_test)
            plt.scatter(itr_not_dead, pid * np.ones(len(itr_not_dead)), c=logprob_test_itr_normalized[itr_not_dead, pid], s=marker_size, marker='s', edgecolor='none', \
                    vmin=vmin_logprob_test, vmax=vmax_logprob_test)
            if True:
                plt.subplot(313)
                plt.plot(itr_print, logprob_test_itr[itr_print, pid], 'b--')
            ancestry = [p.ancestry[i] for i in itr_resampling]
            pos = get_particle_positions(ancestry, pid, n_iter, itr_resampling)
            #print '\nancestry = %s\npos = %s' % (ancestry, pos)
            plt.subplot(311)
            plt.plot(itr_vec, pos, 'b-')    #, lw=3)
            plt.subplot(312)
            plt.plot(itr_vec, pos, 'k:', lw=1, markersize=0.01)
            plt.subplot(313)
            logprob_test_ancestor = [logprob_test_itr[itr, pos[itr]] for itr in itr_print]
            plt.plot(itr_print, logprob_test_ancestor, 'k-')
        plt.subplot(311)
        for i_, itr in enumerate(itr_print):
            for pid, p in enumerate(particles_itr_d[itr]):
                ancestry = p.ancestry[:]
                ancestry.append(pid)            # for "parent particle", pid would be the current position
                if len(ancestry) > 1:
                    ancestry[0] = ancestry[1]   # everything at iteration 0 is the same ... doesn't matter ... just makes the plots look nicer 
                plt.plot(list(range(len(ancestry))), ancestry, 'r--') #, lw=2)
        for itr in itr_resampling:
            plt.axvline(x=itr, color='y', linestyle=':')
        plt.subplot(311)
        plt.xlabel('Iteration')
        plt.ylabel('weights of particles')
        plt.xlim(-1, n_iter)
        plt.ylim(-1, settings.n_particles + 0)
        plt.subplot(312)
        max_vec = ['%.2f' % x for x in max_logprob_test_itr[itr_print]]
        print(itr_print)
        print(max_vec)
        for itr_print_tmp, max_vec_tmp in zip(itr_print, max_vec):
            plt.text(itr_print_tmp, settings.n_particles + 4, max_vec_tmp, rotation='vertical')
        plt.xlabel('Iteration')
        plt.ylabel('log_prob_test - max (for each column)')
        plt.xlim(-1, n_iter)
        plt.ylim(-1, settings.n_particles + 6)
        plt.colorbar(orientation='horizontal')
        plt.subplot(313)
        plt.xlabel('Iteration')
        plt.ylabel('log_prob_test of particles')
        plt.xlim(-1, n_iter)
        print('filename_smc_ancestry_tag = %spdf' % filename_smc_ancestry_tag)
        plt.savefig(filename_smc_ancestry_tag + 'pdf', format='pdf', bbox_inches='tight')
        f.close()


def brute_force_compute_posterior(data, settings, param, particles, cache):
    """ Compute exact posterior + posterior under one-step optimal proposal via
        brute force enumeration """
    p_0 = copy_particle(particles[0], settings)
    p_0.leaf_nodes = []
    all_trees = generate_all_trees(data, param, settings, cache, p_0)
    create_prediction_trees(all_trees, param, data, settings)
    p_logprior = np.array([p.compute_logprior() for p in all_trees])
    p_loglik = np.array([p.compute_loglik() for p in all_trees])
    p_logprob = np.array([p.compute_logprob() for p in all_trees])
    log_p_y_given_x = logsumexp(p_logprob)
    log_weight_trees = p_logprob - log_p_y_given_x      # normalization required for test below
    try:
        assert np.sum(log_weight_trees) < 1e-10
        assert log_p_y_given_x < 1e-10
    except:
        print('WARNING: something may be wrong here')
        print('chk if weights sum to 1: %s' % np.exp(logsumexp(log_weight_trees)))
        print('chk if logsumexp(p_logprior) sums to 0: %s' % logsumexp(p_logprior))
        print('chk if 0: %s' % np.sum(p_logprob - p_loglik - p_logprior))
        raise Exception
    if settings.verbose >= 2:
        print()
        print(data['x_train'], data['y_train'])
        print('*'*80)
        print('printing all trees')
    counts_numcuts = {}
    true_posterior_trees = {'prob': {}, 'all_trees': {}}
    for i_p, p in enumerate(all_trees):
        p_depth = len(p.non_leaf_nodes)
        counts_numcuts[p_depth] = counts_numcuts.get(p_depth, 0) + 1
        tree_key = p.gen_tree_key()
        true_posterior_trees['all_trees'][tree_key] = p
        true_posterior_trees['prob'][tree_key] = np.exp(log_weight_trees[i_p])
        if settings.verbose >= 2:
            print('*'*30 + ' beginning ' + '*'*30)
            print('posterior probability of current tree = %s, log_weight = %s' % \
                    (np.exp(log_weight_trees[i_p]), log_weight_trees[i_p]))
            p.print_tree()
            print('logprior: %s\tloglik: %s\tlogprob: %s' % (p.compute_logprior(), p.compute_loglik(), p.compute_logprob()))
            print('-'*80)
    print('total number of trees = %s' % len(all_trees))
    print('counts by number of cuts = ')
    print(counts_numcuts)
    print('Exact marginal likelihood of the data (brute force) = %s' % log_p_y_given_x)
    print('Results on test data (exact)')
    (pred_prob_overall_test_exact, metrics_test_exact) = \
        evaluate_predictions_smc(all_trees, data, data['x_test'], data['y_test'], settings, param, softmax(log_weight_trees))
    fname_true_posterior = '%s.true_posterior.p' % settings.dataset
    pickle.dump(true_posterior_trees, open(fname_true_posterior, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print('\n')
    #
    # generate all trees, weighted by probability under one-step optimal proposal
    #
    p_1 = copy_particle(particles[0], settings)
    p_1.leaf_nodes = []
    all_trees_posterior = generate_all_trees_posterior(data, param, settings, cache, p_1)
    create_prediction_trees(all_trees_posterior, param, data, settings)
    if settings.verbose >= 2:
        print()
        print(data['x_train'], data['y_train'])
        print('*'*80)
        print('printing all trees (with probability under posterior proposal)')
    counts_numcuts_posterior = {}
    for p in all_trees_posterior:
        p_depth = len(p.non_leaf_nodes)
        counts_numcuts_posterior[p_depth] = counts_numcuts_posterior.get(p_depth, 0) + 1
        if settings.verbose >= 2:
            p.print_tree()
            print('logweight = %s' % p.log_weight)
            print('-'*80)
    print('total number of trees = %s' % len(all_trees_posterior))
    print('counts by number of cuts = ')
    print(counts_numcuts_posterior)
    log_weight_trees_posterior =  np.array([p.log_weight for p in all_trees_posterior])
    print('Marginal likelihood of the data (posterior proposal, brute force enumeration of all trees) = %s' % \
            logsumexp(log_weight_trees_posterior))
    print('Results on test data (probability under posterior proposal)')
    (pred_prob_overall_test_posterior, metrics_test_posterior) = \
        evaluate_predictions_smc(all_trees_posterior, data, data['x_test'], data['y_test'], \
        settings, param, softmax(log_weight_trees_posterior))
    return (pred_prob_overall_test_exact, pred_prob_overall_test_posterior)


def main():
    time_0 = time.clock()
    settings = process_command_line()
    print()
    print('%'*160)
    print('Beginning bdtsmc.py')
    print('Current settings:')
    pp.pprint(vars(settings))

    # Resetting random seed
    np.random.seed(settings.init_id * 1000)
    random.seed(settings.init_id * 1000)

    # Loading data
    print('\nLoading data ...')
    data = load_data(settings)
    print('Loating data ... completed')
    print('Dataset name = %s' % settings.dataset)
    print('Characteristics of the dataset:')
    print('n_train = %d, n_test = %d, n_dim = %d' %\
            (data['n_train'], data['n_test'], data['n_dim']))
    if settings.optype == 'class':
        print('n_class = %d' % (data['n_class']))
    if (settings.demo == 1) and (data['n_dim'] != 2):
        print('demo==1 option valid only for 2d data')
        raise Exception

    # Initialize smc
    print('\nInitializing SMC\n')
    # precomputation
    (particles, param, log_weights, cache, cache_tmp) = init_smc(data, settings)
    time_init = time.clock() - time_0

    # Brute force computation of exact posterior and one-step optimal posterior
    if settings.dataset == 'toy-small':
        (pred_prob_overall_test_exact, pred_prob_overall_test_posterior) = \
                    brute_force_compute_posterior(data, settings, param, particles, cache)

    # Run smc
    print('\nRunning SMC')
    (particles, ess_itr, log_weights_itr, log_pd, particle_stats_itr_d, particles_itr_d, log_pd_islands) = \
            run_smc(particles, data, settings, param, log_weights, cache)
    time_method = time.clock() - time_0     # includes precomputation time
    time_method_sans_init = time.clock() - time_0 - time_init
    
    # Printing some diagnostics
    print()
    print('Estimate of log marginal probability i.e. log p(Y|X) = %s ' % log_pd)
    print('Estimate of log marginal probability for different islands = %s' % log_pd_islands)
    print('logsumexp(log_pd_islands) - np.max(log_pd_islands) = %s\n' % \
            (logsumexp(log_pd_islands) - np.max(log_pd_islands)))
    if settings.debug == 1:
        print('log_weights_itr = \n%s' % log_weights_itr)
        # check if log_weights are computed correctly
        for i_, p in enumerate(particles):
            log_w = log_weights_itr[-1, i_] + np.log(settings.n_particles) - np.log(settings.n_islands)
            logprior_p = p.compute_logprior()
            loglik_p = p.compute_loglik()
            logprob_p = p.compute_logprob()
            if (np.abs(settings.ess_threshold) < 1e-15) and (settings.proposal == 'prior'):
                # for the criterion above, only loglik should contribute to the weight update
                try:
                    check_if_zero(log_w - loglik_p)
                except AssertionError:
                    print('Incorrect weight computation: log_w (smc) = %s, loglik_p = %s' % (log_w, loglik_p))
                    raise AssertionError
            try:
                check_if_zero(logprob_p - loglik_p - logprior_p)
            except AssertionError:
                print('Incorrect weight computation')
                print('check if 0: %s, logprior_p = %s, loglik_p = %s' % (logprob_p - loglik_p - logprior_p, logprior_p, loglik_p))
                raise AssertionError

    # Evaluate
    time_predictions_init = time.clock()
    print('Results on training data (log predictive prob is bogus)')
    # log_predictive on training data is bogus ... you are computing something like \int_{\theta} p(data|\theta) p(\theta|data)
    if settings.weight_islands == 1:
        # each island's prediction is weighted by its marginal likelihood estimate which is equivalent to micro-averaging globally
        weights_prediction = softmax(log_weights_itr[-1, :])
        assert('islandv1' in settings.tag)
    else:
        # correction for macro-averaging predictions across islands
        weights_prediction = np.ones(settings.n_particles) / settings.n_islands
        n_particles_tmp = settings.n_particles // settings.n_islands
        for i_ in range(settings.n_islands):
            pid_min, pid_max = i_ * n_particles_tmp, (i_ + 1) * n_particles_tmp - 1
            pid_range_tmp = list(range(pid_min, pid_max+1))
            weights_prediction[pid_range_tmp] *= softmax(log_weights_itr[-1, pid_range_tmp]) 
    (pred_prob_overall_train, metrics_train) = \
            evaluate_predictions_smc(particles, data, data['x_train'], data['y_train'], settings, param, weights_prediction)
    print('\nResults on test data')
    (pred_prob_overall_test, metrics_test) = \
            evaluate_predictions_smc(particles, data, data['x_test'], data['y_test'], settings, param, weights_prediction)
    log_prob_train = metrics_train['log_prob']
    log_prob_test = metrics_test['log_prob']
    if settings.optype == 'class':
        acc_train = metrics_train['acc']
        acc_test = metrics_test['acc']
    else:
        mse_train = metrics_train['mse']
        mse_test = metrics_test['mse']
    time_prediction = (time.clock() - time_predictions_init)

    if (settings.debug == 1) and (settings.dataset == 'toy-small'):
        print('pred_prob_overall_test_exact = \n%s' % pred_prob_overall_test_exact)
        print('pred_prob_overall_test_posterior = \n%s' % pred_prob_overall_test_posterior)
        print('pred_prob_overall_test = \n%s' % pred_prob_overall_test)
        print('pred_prob_overall_test_exact - pred_prob_overall_test = \n%s' % (pred_prob_overall_test_exact - pred_prob_overall_test))
    
    tree_stats = np.zeros((settings.n_particles, 7))
    for i_p, p in enumerate(particles):
        tree_stats[i_p, :] = np.array([p.compute_loglik(), p.compute_logprior(), p.compute_logprob(), \
            p.depth, len(p.leaf_nodes), len(p.non_leaf_nodes), p.do_not_grow])

    # Write results to disk (timing doesn't include saving)
    time_total = time.clock() - time_0
    if settings.save == 1:
        filename = get_filename_smc(settings)
        print('filename = ' + filename)
        results = {'log_prob_test': log_prob_test, \
                'log_prob_train': log_prob_train, \
                'time_total': time_total, 'time_method': time_method, \
                'time_init': time_init, 'time_method_sans_init': time_method_sans_init,\
                'time_prediction': time_prediction}
        if settings.optype == 'class':
            results['acc_test'] = acc_test
            results['acc_train'] = acc_train
        else:
            results['mse_test'] = mse_test
            results['mse_train'] = mse_train
        results['settings'] = settings
        results['log_weights_itr'] = log_weights_itr
        results['ess_itr'] = ess_itr
        results['tree_stats'] = tree_stats
        results['particle_stats_itr_d'] = particle_stats_itr_d
        results['particles_itr_d'] = particles_itr_d
        pickle.dump(results, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        # storing final predictions as well; recreate new "results" dict
        results = {'pred_prob_overall_train': pred_prob_overall_train, \
                    'pred_prob_overall_test': pred_prob_overall_test}
        filename2 = filename[:-2] + '.tree_predictions.p'
        pickle.dump(results, open(filename2, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        # store the posterior as well
        filename_smc_posterior = filename[:-2] + '.smc_particles.p'
        smc_posterior_trees = {'log_weights': log_weights_itr[-1,], 'particles': particles, 'time_method': time_method}
        pickle.dump(smc_posterior_trees, open(filename_smc_posterior, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    
    if settings.store_history >= 1:
        postmortem_results(data, settings, particles, filename, ess_itr, log_weights_itr, \
                            particle_stats_itr_d, particles_itr_d)
    if settings.verbose >= 2:
        print('Only nodes with > 1 datapoint are processed')
        print(ess_itr)
        for p in particles:
            print('total num_nodes_processed = %s, num_leaf = %s, num_non_leaf = %s\nnum_nodes_processed_itr = %s' \
                    % (sum(p.num_nodes_processed_itr), len(p.leaf_nodes), len(p.non_leaf_nodes), p.num_nodes_processed_itr))

    time_total = time.clock() - time_0
    print()
    print('Time for initializing SMC (seconds) = %f' % (time_init))
    print('Time for running SMC (seconds) = %f' % (time_method_sans_init))
    print('Total time for running SMC, including init (seconds) = %f' % (time_method))
    print('Time for prediction/evaluation (seconds) = %f' % (time_prediction))
    print('Total time (Loading data/ initializing smc/ running smc/ predictions/ saving) (seconds) = %f\n' % (time_total))
    


if __name__ == "__main__":
    main()
