#!/usr/bin/env bash

ckpt_path='/home/casanova/scratch/ckpt_seg'
data_path='/home/casanova/scratch/'

## GTA pretraining for Cityscapes
python train_supervised.py --exp-name 'gta_pretraining_cs' --checkpointer \
--ckpt-path $ckpt_path --data-path $data_path \
--input-size 256 512 --dataset 'gta' \
--epoch-num 1500 --lr 0.0001 --train-batch-size 16 --val-batch-size 1 --patience 30 \
--test --snapshot 'best_jaccard_val.pth'

## Pretraining GTA + Cityscapes D_t (lower bound, what is used as a starting point for the active learning algorithm)
python train_supervised.py --exp-name 'cityscapes_pretrained_dt' --checkpointer \
--ckpt-path $ckpt_path --data-path $data_path \
--load-weights --exp-name-toload 'gta_fpn_baseline_lr1e-3_fullres_finetune' \
--input-size 256 512 --dataset 'cityscapes_subset' \
--epoch-num 1500 --lr 0.0001 --train-batch-size 16 --val-batch-size 1 --patience 50 \
--test --snapshot 'best_jaccard_val.pth'

#Upper bound Cityscapes
python train_supervised.py --exp-name 'cityscapes_upperbound' --checkpointer \
--ckpt-path $ckpt_path --data-path $data_path \
--load-weights --exp-name-toload 'gta_fpn_baseline_lr1e-3_fullres_finetune' \
--input-size 256 512 --dataset 'cs_upper_bound' \
--epoch-num 1500 --lr 0.0001 --train-batch-size 16 --val-batch-size 1 --patience 50 \
--test --snapshot 'best_jaccard_val.pth'

## GTA pretraining for Camvid
python train_supervised.py --exp-name 'gta_pretraining_camvid' --checkpointer \
--ckpt-path $ckpt_path --data-path $data_path \
--input-size 224 224 --dataset 'gta_for_camvid' \
--epoch-num 1500 --lr 0.005 --train-batch-size 32 --val-batch-size 8 --patience 30 --scale-size 480 \
--test --snapshot 'best_jaccard_val.pth'

## Pretraining GTA + Camvid D_t (lower bound, what is used as a starting point for the active learning algorithm)
python train_supervised.py --exp-name 'camvid_pretrained_dt' --checkpointer \
--ckpt-path $ckpt_path --data-path $data_path \
--load-weights --exp-name-toload 'gta_pretraining_camvid' \
--input-size 224 224 --dataset 'camvid_subset' \
--epoch-num 1500 --lr 0.0005 --train-batch-size 32 --val-batch-size 8 --patience 30 \
--test --snapshot 'best_jaccard_val.pth'

## Upper bound Camvid
python train_supervised.py --exp-name 'camvid_upperbound' --checkpointer \
--ckpt-path $ckpt_path --data-path $data_path \
--load-weights --exp-name-toload 'gta_pretraining_camvid' \
--input-size 224 224 --dataset 'camvid' \
--epoch-num 1500 --lr 0.0005 --train-batch-size 32 --val-batch-size 8 --patience 30 \
--test --snapshot 'best_jaccard_val.pth'



