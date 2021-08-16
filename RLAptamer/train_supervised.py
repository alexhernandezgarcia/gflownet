import os
import sys
import shutil
import numpy as np
import random
import math

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, MultiStepLR

from models.model_utils import create_models, load_models
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, train, validate, final_test
import utils.parser as parser

cudnn.benchmark = False
cudnn.deterministic = True


def main(args):
    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ####------ Create experiment folder  ------####
    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))

    ####------ Print and save arguments in experiment folder  ------####
    parser.save_arguments(args)
    ####------ Copy current config file to ckpt folder ------####
    fn = sys.argv[0].rsplit('/', 1)[-1]
    shutil.copy(sys.argv[0], os.path.join(args.ckpt_path, args.exp_name, fn))

    ####------ Create segmentation, query and target networks ------####
    kwargs_models = {"dataset": args.dataset,
                     "al_algorithm": args.al_algorithm,
                     "region_size": args.region_size
                     }
    net, _, _ = create_models(**kwargs_models)

    ####------ Load weights if necessary and create log file ------####
    kwargs_load = {"net": net,
                   "load_weights": args.load_weights,
                   "exp_name_toload": args.exp_name_toload,
                   "snapshot": args.snapshot,
                   "exp_name": args.exp_name,
                   "ckpt_path": args.ckpt_path,
                   "checkpointer": args.checkpointer,
                   "exp_name_toload_rl": args.exp_name_toload_rl,
                   "policy_net": None,
                   "target_net": None,
                   "test": args.test,
                   "dataset": args.dataset,
                   "al_algorithm": 'None'}
    logger, curr_epoch, best_record = load_models(**kwargs_load)

    ####------ Load training and validation data ------####
    kwargs_data = {"data_path": args.data_path,
                   "tr_bs": args.train_batch_size,
                   "vl_bs": args.val_batch_size,
                   "n_workers": 4,
                   "scale_size": args.scale_size,
                   "input_size": args.input_size,
                   "num_each_iter": args.num_each_iter,
                   "only_last_labeled": args.only_last_labeled,
                   "dataset": args.dataset,
                   "test": args.test,
                   "al_algorithm": args.al_algorithm,
                   "full_res": args.full_res,
                   "region_size": args.region_size,
                   "supervised": True}

    train_loader, _, val_loader, _ = get_data(**kwargs_data)

    ####------ Create losses ------####
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.ignore_label).cuda()

    ####------ Create optimizers (and load them if necessary) ------####
    kwargs_load_opt = {"net": net,
                       "opt_choice": args.optimizer,
                       "lr": args.lr,
                       "wd": args.weight_decay,
                       "momentum": args.momentum,
                       "ckpt_path": args.ckpt_path,
                       "exp_name_toload": args.exp_name_toload,
                       "exp_name": args.exp_name,
                       "snapshot": args.snapshot,
                       "checkpointer": args.checkpointer,
                       "load_opt": args.load_opt,
                       "policy_net": None,
                       "lr_dqn": args.lr_dqn,
                       "al_algorithm": 'None'}

    optimizer, _ = create_and_load_optimizers(**kwargs_load_opt)

    # Early stopping params initialization
    es_val = 0
    es_counter = 0

    if args.train:
        print('Starting training...')
        scheduler = ExponentialLR(optimizer, gamma=0.998)
        net.train()
        for epoch in range(curr_epoch, args.epoch_num + 1):
            print('Epoch %i /%i' % (epoch, args.epoch_num + 1))
            tr_loss, _, tr_acc, tr_iu = train(train_loader, net, criterion,
                                              optimizer, supervised=True)

            if epoch % 1 == 0:
                vl_loss, val_acc, val_iu, iu_xclass, _ = validate(val_loader, net, criterion,
                                                                  optimizer, epoch, best_record,
                                                                  args)

            ## Append info to logger
            info = [epoch, optimizer.param_groups[0]['lr'],
                    tr_loss,
                    0, vl_loss, tr_acc, val_acc, tr_iu, val_iu]
            for cl in range(train_loader.dataset.num_classes):
                info.append(iu_xclass[cl])
            logger.append(info)
            scheduler.step()
            # Early stopping with val jaccard
            es_counter += 1
            if val_iu > es_val and not math.isnan(val_iu):
                torch.save(net.cpu().state_dict(),
                           os.path.join(args.ckpt_path, args.exp_name,
                                        'best_jaccard_val.pth'))
                net.cuda()
                es_val = val_iu
                es_counter = 0
            elif es_counter > args.patience:
                print('Patience for Early Stopping reached!')
                break

        logger.close()

    if args.final_test:
        final_test(args, net, criterion)


if __name__ == '__main__':
    ####------ Parse arguments from console  ------####
    args = parser.get_arguments()
    main(args)
