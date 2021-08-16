import time
import math
import numpy as np
import os
import random
from functools import reduce
from scipy.stats import entropy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.query_network import QueryNetworkDQN
from models.AptamerLSTM import AptamerLSTM
from utils.final_utils import get_logfile
from utils.progressbar import progress_bar

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10


def create_models():
    """Creates the 3 Networks needed for Training

    :return: sequence network, query network, target network (same construction as query network)
    """

    # Sequence network

    net = AptamerLSTM().cuda()
    print("Model has " + str(count_parameters(net)))

    # Query network (and target network for DQN)
    state_dataset_size = 30  # This depends on size of dataset V
    action_state_length = 3

    policy_net = QueryNetworkDQN(
        model_state_length=state_dataset_size,
        action_state_length=action_state_length,
        bias_average=1,  # TODO Figure out the query network bias trick from 2018 paper.
    ).cuda()
    target_net = QueryNetworkDQN(
        model_state_length=state_dataset_size,
        action_state_length=action_state_length,
        bias_average=1,  # TODO Figure out the query network bias trick from 2018 paper.
    ).cuda()
    print("Policy network has " + str(count_parameters(policy_net)))

    print("Models created!")
    return net, policy_net, target_net


def count_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def load_models(
    net,
    load_weights,
    exp_name_toload,
    snapshot,
    exp_name,
    ckpt_path,
    checkpointer,
    exp_name_toload_rl="",
    policy_net=None,
    target_net=None,
    test=False,
    dataset="cityscapes",
    al_algorithm="random",
):
    """Load model weights.
    :param net: Segmentation network
    :param load_weights: (bool) True if segmentation network is loaded from pretrained weights in 'exp_name_toload'
    :param exp_name_toload: (str) Folder where to find pre-trained segmentation network's weights.
    :param snapshot: (str) Name of the checkpoint.
    :param exp_name: (str) Experiment name.
    :param ckpt_path: (str) Checkpoint name.
    :param checkpointer: (bool) If True, load weights from the same folder.
    :param exp_name_toload_rl: (str) Folder where to find trained weights for the query network (DQN). Used to test
    query network.
    :param policy_net: Policy network.
    :param target_net: Target network.
    :param test: (bool) If True and al_algorithm='ralis' and there exists a checkpoint in 'exp_name_toload_rl',
    we will load checkpoint for trained query network (DQN).
    :param dataset: (str) Which dataset.
    :param al_algorithm: (str) Which active learning algorithm.
    """

    policy_path = os.path.join(ckpt_path, exp_name_toload_rl, "policy_" + snapshot)
    net_path = os.path.join(ckpt_path, exp_name_toload, "best_jaccard_val.pth")

    policy_checkpoint_path = os.path.join(ckpt_path, exp_name, "policy_" + snapshot)
    net_checkpoint_path = os.path.join(ckpt_path, exp_name, "last_jaccard_val.pth")

    ####------ Load policy (RL) from one folder and network from another folder ------####
    if al_algorithm == "ralis" and test and os.path.isfile(policy_path):
        print("(RL and TEST) Testing policy from another experiment folder!")
        policy_net.load_state_dict(torch.load(policy_path))
        # Load pre-trained segmentation network from another folder (best checkpoint)
        if load_weights and len(exp_name_toload) > 0:
            print("Loading pre-trained segmentation network (best checkpoint).")
            net_dict = torch.load(net_path)
            if len([key for key, value in net_dict.items() if "module" in key.lower()]) > 0:
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in net_dict.items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                net_dict = new_state_dict
            net.load_state_dict(net_dict)
            net.cuda()

        if (
            checkpointer
        ):  # In case the experiment is interrupted, load most recent segmentation network
            if os.path.isfile(net_checkpoint_path):
                print("(RL and TEST) Loading checkpoint for segmentation network!")
                net_dict = torch.load(net_checkpoint_path)
                if len([key for key, value in net_dict.items() if "module" in key.lower()]) > 0:
                    from collections import OrderedDict

                    new_state_dict = OrderedDict()
                    for k, v in net_dict.items():
                        name = k[7:]  # remove module.
                        new_state_dict[name] = v
                    net_dict = new_state_dict
                net.load_state_dict(net_dict)
    else:
        ####------ Load experiment from another folder ------####
        if load_weights and len(exp_name_toload) > 0:
            print("(From another exp) training resumes from best_jaccard_val.pth")
            net_dict = torch.load(net_path)
            if len([key for key, value in net_dict.items() if "module" in key.lower()]) > 0:
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in net_dict.items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                net_dict = new_state_dict
            net.load_state_dict(net_dict)

        ####------ Resume experiment ------####
        if checkpointer:
            ##-- Check if weights exist --##
            if os.path.isfile(net_checkpoint_path):
                print("(Checkpointer) training resumes from last_jaccard_val.pth")
                net_dict = torch.load(net_checkpoint_path)
                if len([key for key, value in net_dict.items() if "module" in key.lower()]) > 0:
                    from collections import OrderedDict

                    new_state_dict = OrderedDict()
                    for k, v in net_dict.items():
                        name = k[7:]  # remove module.
                        new_state_dict[name] = v
                    net_dict = new_state_dict
                net.load_state_dict(net_dict)
                if (
                    al_algorithm == "ralis"
                    and os.path.isfile(policy_checkpoint_path)
                    and policy_net is not None
                    and target_net is not None
                ):
                    print("(Checkpointer RL) training resumes from " + snapshot)
                    policy_net.load_state_dict(torch.load(policy_checkpoint_path))
                    policy_net.cuda()
                    target_net.load_state_dict(torch.load(policy_checkpoint_path))
                    target_net.cuda()
            else:
                print("(Checkpointer) Training starts from scratch")

    ####------ Get log file ------####
    if al_algorithm == "ralis":
        logger = None
        best_record = None
        curr_epoch = None
    else:
        num_classes = 11 if "camvid" in dataset else 19
        logger, best_record, curr_epoch = get_logfile(
            ckpt_path, exp_name, checkpointer, snapshot, num_classes=num_classes
        )

    return logger, curr_epoch, best_record


def get_region_candidates(candidates, train_set, num_regions=2):
    """Get region candidates function.
    :param candidates: (list) randomly sampled image indexes for images that contain unlabeled regions.
    :param train_set: Training set.
    :param num_regions: Number of regions to take as possible regions to be labeled.
    :return: candidate_regions: List of tuples (int(Image index), int(width_coord), int(height_coord)).
        The coordinate is the left upper corner of the region.
    """
    s = time.time()
    print("Getting region candidates...")
    total_regions = num_regions
    candidate_regions = []
    #### --- Get candidate regions --- ####
    counter_regions = 0
    available_regions = train_set.get_num_unlabeled_regions()
    rx, ry = train_set.get_unlabeled_regions()
    while (
        counter_regions < total_regions and (total_regions - counter_regions) <= available_regions
    ):
        index_ = np.random.choice(len(candidates))
        index = candidates[index_]
        num_regions_left = train_set.get_num_unlabeled_regions_image(int(index))
        if num_regions_left > 0:
            counter_x, counter_y = train_set.get_random_unlabeled_region_image(int(index))
            candidate_regions.append((int(index), counter_x, counter_y))
            available_regions -= 1
            counter_regions += 1
            if num_regions_left == 1:
                candidates.pop(int(index_))
        else:
            print("This image has no more unlabeled regions!")

    train_set.set_unlabeled_regions(rx, ry)
    print("Regions candidates indexed! Time elapsed: " + str(time.time() - s))
    print("Candidate regions are " + str(counter_regions))
    return candidate_regions


def compute_state(net, state_dataset, unlabelled_dataset, labelled_dataset):
    ## Adapted from LAL-RL, https://github.com/ksenia-konyushkova/LAL-RL
    """Function for computing the state depending on the sequence model and next available actions.

    This function computes 1) model_state that characterises
    the current state of the model and it is computed as
    a function of predictions on the hold-out dataset
    2) next_action_state that characterises all possible actions
    (unlabelled datapoints) that can be taken at the next step.

    :param: net:
    :param: state_dataset:
    :param: unlabelled_dataset:
    :param: labelled_dataset:

    :return: model_state: a numpy.ndarray of size of number of datapoints in dataset.state_data characterizing the current model and, thus, the state of the environment
    :return: next_action_state: a numpy.ndarray of size #features characterising actions (currently, 3) x #unlabelled datapoints where each column corresponds to the vector characterizing each possible action.
    """
    # COMPUTE MODEL_STATE
    net.eval()
    predictions = [net(datapoint) for datapoint in state_dataset]
    idx = np.argsort(predictions)
    # the state representation is the *sorted* list of scores
    model_state = predictions[idx]

    # COMPUTE ACTION_STATE
    # prediction (score) of model on each unlabelled sample
    a1 = [net(datapoint) for datapoint in unlabelled_dataset]
    a2 = []
    a3 = []
    for xi, idx in enumerate(unlabelled_dataset):
        # average distance to every unlabelled datapoint
        a2 += np.mean([np.sum(xi != datapoint) for datapoint in unlabelled_dataset])
        # average distance to every labelled datapoint
        a3 += np.mean([np.sum(xi != datapoint) for datapoint in labelled_dataset])
    action_state = np.concatenate(([a1], [a2], [a3]), axis=0)
    return model_state, action_state


def select_action(args, policy_net, model_state, action_state, steps_done, test=False):
    """We select the action: index of the image to label.
    :(argparse.ArgumentParser) args: The parser with all the defined arguments.
    :param policy_net: policy network.
    :param model_state: (torch.Variable) Torch tensor containing the model state representation.
    :param action_state: (torch.Variable) Torch tensor containing the action state representations.
    :param steps_done: (int) Number of images labeled so far.
    :param test: (bool) Whether we are testing the DQN or training it.
    :return: Action (indexes of the regions to label)
    """

    policy_net.eval()
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    q_val_ = []
    if sample > eps_threshold or test:
        print("Action selected with DQN!")
        with torch.no_grad():
            # Get Q-values for every action
            q_val_ = [policy_net(model_state, action_i_state) for action_i_state in action_state]

            action = q_val_.max(index)[1].view(-1).cpu()
            del q_val_
    else:
        action = Variable(
            torch.Tensor([np.random.choice(range(args.rl_pool), state.size()[0], replace=True)])
            .type(torch.LongTensor)
            .view(-1)
        )  # .cuda()

    return action


# TODO rewrite to remove all the fancy image/region candidate stuff. Make it simply add a datapoint to the labelleddataset
def add_labeled_datapoint(
    args, list_existing_images, region_candidates, train_set, action_list, budget, n_ep
):
    """This function adds an image, indicated by 'action_list' out of 'region_candidates' list
     and adds it into the labeled dataset and the list of existing images.

    :(argparse.ArgumentParser) args: The parser with all the defined arguments.
    :param list_existing_images: (list) of tuples (image idx, region_x, region_y) of all regions that have
            been selected in the past to add them to the labeled set.
    :param region_candidates: (list) List of all possible regions to add.
    :param train_set: (torch.utils.data.Dataset) Training set.
    :param action_list: Selected indexes of the regions in 'region_candidates' to be labeled.
    :param budget: (int) Number of maximum regions we want to label.
    :param n_ep: (int) Number of episode.
    :return: List of existing images, updated with the new image.
    """

    lab_set = open(
        os.path.join(args.ckpt_path, args.exp_name, "labeled_set_" + str(n_ep) + ".txt"), "a"
    )
    for i, action in enumerate(action_list):
        if train_set.get_num_labeled_regions() >= budget:
            print("Budget reached with " + str(train_set.get_num_labeled_regions()) + " regions!")
            break
        im_toadd = region_candidates[i, action, 0]
        train_set.add_index(
            im_toadd, (region_candidates[i, action, 1], region_candidates[i, action, 2])
        )
        list_existing_images.append(tuple(region_candidates[i, action]))
        lab_set.write(
            "%i,%i,%i"
            % (im_toadd, region_candidates[i, action, 1], region_candidates[i, action, 2])
        )
        lab_set.write("\n")
    print("Labeled set has now " + str(train_set.get_num_labeled_regions()) + " labeled regions.")

    return list_existing_images


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def compute_bald(predictions):
    ### Compute BALD ###
    expected_entropy = -torch.mean(
        torch.sum(predictions * torch.log(predictions + 1e-10), dim=1), dim=0
    )
    expected_p = torch.mean(predictions, dim=0)  # [batch_size, n_classes]
    pred_py = expected_p.max(0)[1]
    entropy_expected_p = -torch.sum(
        expected_p * torch.log(expected_p + 1e-10), dim=0
    )  # [batch size]
    bald_acq = entropy_expected_p - expected_entropy
    return bald_acq.unsqueeze(0), pred_py.unsqueeze(0)


def add_kl_pool2(state, n_cl=19):
    sim_matrix = torch.zeros((state.shape[0], state.shape[1], 32))
    all_cand = state[:, :, 0 : n_cl + 1].view(-1, n_cl + 1).transpose(1, 0)
    for i in range(state.shape[0]):
        pool_hist = state[i, :, 0 : n_cl + 1]
        for j in range(pool_hist.shape[0]):
            prov_sim = entropy(
                pool_hist[j : j + 1].repeat(all_cand.shape[1], 1).transpose(0, 1), all_cand
            )
            hist, _ = np.histogram(prov_sim, bins=32)
            hist = hist / hist.sum()
            sim_matrix[i, j, :] = torch.Tensor(hist)
    state = torch.cat([state, sim_matrix], dim=2)
    return state


def create_feature_vector_3H_region_kl_sim(
    pred_region, ent_region, train_set, num_classes=19, reg_sz=(128, 128)
):
    unique, counts = np.unique(pred_region, return_counts=True)
    sample_stats = np.zeros(num_classes + 1) + 1e-7
    sample_stats[unique.astype(int)] = counts
    sample_stats = sample_stats.tolist()
    sz = ent_region.size()
    ks_x = int(reg_sz[0] // 8)
    ks_y = int(reg_sz[1] // 8)
    with torch.no_grad():
        sample_stats += (
            5
            - F.max_pool2d(5 - ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(
                -1
            )
        ).tolist()  # min entropy
        sample_stats += (
            F.avg_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x))
            .view(-1)
            .tolist()
        )
        sample_stats += (
            F.max_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x))
            .view(-1)
            .tolist()
        )
    if len(train_set.balance_cl) > 0:
        inp_hist = sample_stats[0 : num_classes + 1]
        sim_sample = entropy(
            np.repeat(np.asarray(inp_hist)[:, np.newaxis], len(train_set.balance_cl), axis=1),
            np.asarray(train_set.balance_cl).transpose(1, 0),
        )
        hist, _ = np.histogram(sim_sample, bins=32)
        sim_lab = list(hist / hist.sum())
        sample_stats += sim_lab
    else:
        sample_stats += [0.0] * 32
    return sample_stats


def create_feature_vector_3H_region_kl(pred_region, ent_region, num_classes=19, reg_sz=(128, 128)):
    unique, counts = np.unique(pred_region, return_counts=True)
    sample_stats = np.zeros(num_classes + 1) + 1e-7
    sample_stats[unique.astype(int)] = counts
    sample_stats = sample_stats.tolist()
    sz = ent_region.size()
    ks_x = int(reg_sz[0] // 8)
    ks_y = int(reg_sz[1] // 8)
    with torch.no_grad():
        sample_stats += (
            5
            - F.max_pool2d(5 - ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(
                -1
            )
        ).tolist()  # min entropy
        sample_stats += (
            F.avg_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x))
            .view(-1)
            .tolist()
        )
        sample_stats += (
            F.max_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x))
            .view(-1)
            .tolist()
        )
    return sample_stats


def compute_entropy_seg(args, im_t, net):
    """
    Compute entropy function
    :param args:
    :param im_t:
    :param net:
    :return:
    """
    net.eval()
    if im_t.dim() == 3:
        im_t_sz = im_t.size()
        im_t = im_t.view(1, im_t_sz[0], im_t_sz[1], im_t_sz[2])

    out, _ = net(im_t)
    out_soft_log = F.log_softmax(out)
    out_soft = torch.exp(out_soft_log)
    ent = -torch.sum(out_soft * out_soft_log, dim=1).detach().cpu()  # .data.numpy()
    del out
    del out_soft_log
    del out_soft
    del im_t

    return ent


def optimize_q_network(
    args,
    memory,
    Transition,
    policy_net,
    target_net,
    optimizerP,
    BATCH_SIZE=32,
    GAMMA=0.999,
    dqn_epochs=1,
):
    """This function optimizes the policy network

    :(ReplayMemory) memory: Experience replay buffer
    :param Transition: definition of the experience replay tuple
    :param policy_net: Policy network
    :param target_net: Target network
    :param optimizerP: Optimizer of the policy network
    :param BATCH_SIZE: (int) Batch size to sample from the experience replay
    :param GAMMA: (float) Discount factor
    :param dqn_epochs: (int) Number of epochs to train the DQN
    """
    # Code adapted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    if len(memory) < BATCH_SIZE:
        return
    print("Optimize model...")
    print(len(memory))
    policy_net.train()
    loss_item = 0
    for ep in range(dqn_epochs):
        optimizerP.zero_grad()
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.uint8
        ).cuda()

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_next_states_subset = torch.cat(
            [s for s in batch.next_state_subset if s is not None]
        )

        state_batch = torch.cat(batch.state)
        state_batch_subset = torch.cat(batch.state_subset)
        action_batch = torch.Tensor([batch.action]).view(-1).type(torch.LongTensor)
        reward_batch = torch.Tensor([batch.reward]).view(-1).cuda()
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        q_val = policy_net(state_batch.cuda(), state_batch_subset.cuda())
        state_batch.cpu()
        state_batch_subset.cpu()
        state_action_values = q_val.gather(1, action_batch.unsqueeze(1).cuda())
        action_batch.cpu()
        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE).cuda()
        # Double dqn, so we compute the values with the target network, we choose the actions with the policy network
        if non_final_mask.sum().item() > 0:
            v_val_act = policy_net(
                non_final_next_states.cuda(), non_final_next_states_subset.cuda()
            ).detach()
            v_val = target_net(
                non_final_next_states.cuda(), non_final_next_states_subset.cuda()
            ).detach()
            non_final_next_states.cpu()
            non_final_next_states_subset.cpu()
            act = v_val_act.max(1)[1]
            next_state_values[non_final_mask] = v_val.gather(1, act.unsqueeze(1)).view(-1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.view(-1), expected_state_action_values)
        loss_item += loss.item()
        progress_bar(ep, dqn_epochs, "[DQN loss %.5f]" % (loss_item / (ep + 1)))
        loss.backward()
        optimizerP.step()

        del q_val
        del expected_state_action_values
        del loss
        del next_state_values
        del reward_batch
        if non_final_mask.sum().item() > 0:
            del act
            del v_val
            del v_val_act
        del state_action_values
        del state_batch
        del action_batch
        del non_final_mask
        del non_final_next_states
        del batch
        del transitions
    lab_set = open(os.path.join(args.ckpt_path, args.exp_name, "q_loss.txt"), "a")
    lab_set.write("%f" % (loss_item))
    lab_set.write("\n")
    lab_set.close()
