import os
import gc
import numpy as np
import pickle

import torch
import torchvision.transforms as standard_transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn

from utils.logger import Logger
from utils.progressbar import progress_bar


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def create_and_load_optimizers(
    net,
    opt_choice,
    lr,
    wd,
    momentum,
    ckpt_path,
    exp_name_toload,
    exp_name,
    snapshot,
    checkpointer,
    load_opt,
    policy_net=None,
    lr_dqn=0.0001,
):
    optimizerP = None
    opt_kwargs = {"lr": lr, "weight_decay": wd, "momentum": momentum}
    opt_kwargs_rl = {"lr": lr_dqn, "weight_decay": 0.001, "momentum": momentum}

    optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, net.parameters()), **opt_kwargs)

    if opt_choice == "SGD":
        optimizerP = optim.SGD(
            params=filter(lambda p: p.requires_grad, policy_net.parameters()), **opt_kwargs_rl
        )
    elif opt_choice == "RMSprop":
        optimizerP = optim.RMSprop(
            params=filter(lambda p: p.requires_grad, policy_net.parameters()), lr=lr_dqn
        )

    name = exp_name_toload if load_opt and len(exp_name_toload) > 0 else exp_name
    opt_path = os.path.join(ckpt_path, name, "opt_" + snapshot)
    opt_policy_path = os.path.join(ckpt_path, name, "opt_policy_" + snapshot)

    if (load_opt and len(exp_name_toload)) > 0 or (checkpointer and os.path.isfile(opt_path)):
        print("(Opt load) Loading net optimizer")
        optimizer.load_state_dict(torch.load(opt_path))

        if os.path.isfile(opt_policy_path):
            print("(Opt load) Loading policy optimizer")
            optimizerP.load_state_dict(torch.load(opt_policy_path))

    print("Optimizers created")
    return optimizer, optimizerP


def get_logfile(ckpt_path, exp_name, checkpointer, snapshot, log_name="log.txt"):
    log_columns = [
        "Epoch",
        "Learning Rate",
        "Train Loss",
        "(deprecated)",
        "Valid Loss",
        "Train Acc.",
        "Valid Acc.",
        "Train mean iu",
        "Valid mean iu",
    ]
    for cl in range(num_classes):
        log_columns.append("iu_cl" + str(cl))
    best_record = {"epoch": 0, "val_loss": 1e10, "acc": 0, "mean_iu": 0}
    curr_epoch = 0
    ##-- Check if log file exists --##
    if checkpointer:
        if os.path.isfile(os.path.join(ckpt_path, exp_name, log_name)):
            print("(Checkpointer) Log file " + log_name + " already exists, appending.")
            logger = Logger(
                os.path.join(ckpt_path, exp_name, log_name), title=exp_name, resume=True
            )
            if "best" in snapshot:
                curr_epoch = int(logger.resume_epoch)
            else:
                curr_epoch = logger.last_epoch
            best_record = {
                "epoch": int(logger.resume_epoch),
                "val_loss": 1e10,
                "mean_iu": float(logger.resume_jacc),
                "acc": 0,
            }
        else:
            print("(Checkpointer) Log file " + log_name + " did not exist before, creating")
            logger = Logger(os.path.join(ckpt_path, exp_name, log_name), title=exp_name)
            logger.set_names(log_columns)

    else:
        print("(No checkpointer activated) Log file " + log_name + " created.")
        logger = Logger(os.path.join(ckpt_path, exp_name, log_name), title=exp_name)
        logger.set_names(log_columns)
    return logger, best_record, curr_epoch


def get_training_stage(args):
    path = os.path.join(args.ckpt_path, args.exp_name, "training_stage.pkl")
    if os.path.isfile(path):
        with open(path, "rb") as f:
            stage = pickle.load(f)
    else:
        stage = None
    return stage


def set_training_stage(args, stage):
    path = os.path.join(args.ckpt_path, args.exp_name, "training_stage.pkl")
    with open(path, "wb") as f:
        pickle.dump(stage, f)


def train(train_loader, net, criterion, optimizer, supervised=False):
    net.train()
    train_loss = 0
    cm_py = (
        torch.zeros((train_loader.dataset.num_classes, train_loader.dataset.num_classes))
        .type(torch.IntTensor)
        .cuda()
    )
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        if supervised:
            im_s, t_s_, _ = data
        else:
            im_s, t_s_, _, _, _ = data

        t_s, im_s = Variable(t_s_).cuda(), Variable(im_s).cuda()
        # Get output of network
        outputs, _ = net(im_s)
        # Get segmentation maps
        predictions_py = outputs.data.max(1)[1].squeeze_(1)
        loss = criterion(outputs, t_s)
        train_loss += loss.item()

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=4)
        optimizer.step()

        cm_py = confusion_matrix_pytorch(
            cm_py, predictions_py.view(-1), t_s_.cuda().view(-1), train_loader.dataset.num_classes
        )

        progress_bar(i, len(train_loader), "[train loss %.5f]" % (train_loss / (i + 1)))

        del outputs
        del loss
        gc.collect()
    print(" ")
    acc, mean_iu, iu = evaluate(cm_py.cpu().numpy())
    print(" [train acc %.5f], [train iu %.5f]" % (acc, mean_iu))
    return train_loss / (len(train_loader)), 0, acc, mean_iu


def validate(
    val_loader, net, criterion, optimizer, epoch, best_record, args, final_final_test=False
):
    net.eval()

    val_loss = 0
    cm_py = (
        torch.zeros((val_loader.dataset.num_classes, val_loader.dataset.num_classes))
        .type(torch.IntTensor)
        .cuda()
    )
    for vi, data in enumerate(val_loader):
        inputs, gts_, _ = data
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            gts = Variable(gts_).cuda()
        outputs, _ = net(inputs)
        # Make sure both output and target have the same dimensions
        if outputs.shape[2:] != gts.shape[1:]:
            outputs = outputs[
                :,
                :,
                0 : min(outputs.shape[2], gts.shape[1]),
                0 : min(outputs.shape[3], gts.shape[2]),
            ]
            gts = gts[
                :, 0 : min(outputs.shape[2], gts.shape[1]), 0 : min(outputs.shape[3], gts.shape[2])
            ]
        predictions_py = outputs.data.max(1)[1].squeeze_(1)
        loss = criterion(outputs, gts)
        vl_loss = loss.item()
        val_loss += vl_loss

        cm_py = confusion_matrix_pytorch(
            cm_py, predictions_py.view(-1), gts_.cuda().view(-1), val_loader.dataset.num_classes
        )

        len_val = len(val_loader)
        progress_bar(vi, len_val, "[val loss %.5f]" % (val_loss / (vi + 1)))

        del outputs
        del vl_loss
        del loss
        del predictions_py
    acc, mean_iu, iu = evaluate(cm_py.cpu().numpy())
    print(" ")
    print(" [val acc %.5f], [val iu %.5f]" % (acc, mean_iu))

    if not final_final_test:
        if mean_iu > best_record["mean_iu"]:
            best_record["val_loss"] = val_loss / len(val_loader)
            best_record["epoch"] = epoch
            best_record["acc"] = acc
            best_record["iu"] = iu
            best_record["mean_iu"] = mean_iu

            torch.save(
                net.cpu().state_dict(),
                os.path.join(args.ckpt_path, args.exp_name, "best_jaccard_val.pth"),
            )
            net.cuda()
            torch.save(
                optimizer.state_dict(),
                os.path.join(args.ckpt_path, args.exp_name, "opt_best_jaccard_val.pth"),
            )

        ## Save checkpoint every epoch
        torch.save(
            net.cpu().state_dict(),
            os.path.join(args.ckpt_path, args.exp_name, "last_jaccard_val.pth"),
        )
        net.cuda()
        torch.save(
            optimizer.state_dict(),
            os.path.join(args.ckpt_path, args.exp_name, "opt_last_jaccard_val.pth"),
        )

        print(
            "best record: [val loss %.5f], [acc %.5f], [mean_iu %.5f],"
            " [epoch %d]"
            % (
                best_record["val_loss"],
                best_record["acc"],
                best_record["mean_iu"],
                best_record["epoch"],
            )
        )

    print("----------------------------------------")

    return val_loss / len(val_loader), acc, mean_iu, iu, best_record


def test(val_loader, net, criterion):
    net.eval()

    val_loss = 0
    cm_py = (
        torch.zeros((val_loader.dataset.num_classes, val_loader.dataset.num_classes))
        .type(torch.IntTensor)
        .cuda()
    )
    for vi, data in enumerate(val_loader):
        inputs, gts_, _ = data
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            gts = Variable(gts_).cuda()

        outputs, _ = net(inputs)
        predictions_py = outputs.data.max(1)[1].squeeze_(1)
        loss = criterion(outputs, gts)
        vl_loss = loss.item()
        val_loss += vl_loss

        cm_py = confusion_matrix_pytorch(
            cm_py, predictions_py.view(-1), gts_.cuda().view(-1), val_loader.dataset.num_classes
        )

        len_val = len(val_loader)
        progress_bar(vi, len_val, "[val loss %.5f]" % (val_loss / (vi + 1)))

        del outputs
        del vl_loss
    acc, mean_iu, iu = evaluate(cm_py.cpu().numpy())
    print(" ")
    print(" [val acc %.5f], [val iu %.5f]" % (acc, mean_iu))

    return val_loss / len(val_loader), acc, mean_iu, iu


def final_test(args, net, criterion):
    # Load best checkpoint for segmentation network
    net_checkpoint_path = os.path.join(args.ckpt_path, args.exp_name, "best_jaccard_val.pth")
    if os.path.isfile(net_checkpoint_path):
        print("(Final test) Load best checkpoint for segmentation network!")
        net_dict = torch.load(net_checkpoint_path)
        if len([key for key, value in net_dict.items() if "module" in key.lower()]) > 0:
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in net_dict.items():
                name = k[7:]  # remove module.
                new_state_dict[name] = v
            net_dict = new_state_dict
        net.load_state_dict(net_dict)
    net.eval()

    # Prepare data transforms
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    input_transform = standard_transforms.Compose(
        [standard_transforms.ToTensor(), standard_transforms.Normalize(*mean_std)]
    )
    target_transform = extended_transforms.MaskToTensor()

    if "camvid" in args.dataset:
        val_set = camvid.Camvid(
            "fine",
            "test" if test else "val",
            data_path=args.data_path,
            joint_transform=None,
            transform=input_transform,
            target_transform=target_transform,
        )
        val_loader = DataLoader(val_set, batch_size=4, num_workers=2, shuffle=False)
    else:
        val_set = cityscapes.CityScapes(
            "fine",
            "val",
            data_path=args.data_path,
            joint_transform=None,
            transform=input_transform,
            target_transform=target_transform,
        )
        val_loader = DataLoader(
            val_set, batch_size=args.val_batch_size, num_workers=2, shuffle=False
        )
    print("Starting test...")
    vl_loss, val_acc, val_iu, iu_xclass = test(val_loader, net, criterion)
    ## Append info to logger
    info = [vl_loss, val_acc, val_iu]
    for cl in range(val_loader.dataset.num_classes):
        info.append(iu_xclass[cl])
    rew_log = open(os.path.join(args.ckpt_path, args.exp_name, "test_results.txt"), "a")
    for inf in info:
        rew_log.write("%f," % (inf))
    rew_log.write("\n")
