#!/usr/bin/env bash

ckpt_path='/home/casanova/scratch/ckpt_seg'
data_path='/home/casanova/scratch/'

for seed in 20 50 234 77 12
    do
    python -u run.py --exp-name 'RALIS_cs_train_seed'$seed --full-res --region-size 128 128 \
     --snapshot 'best_jaccard_val.pth' --al-algorithm 'ralis' \
    --ckpt-path $ckpt_path --data-path $data_path \
    --load-weights --exp-name-toload 'gta_pretraining_cs' \
    --dataset 'cityscapes' --lr 0.0001 --train-batch-size 16 --val-batch-size 1 --patience 10 \
    --input-size 256 512 --only-last-labeled --budget-labels 3840  --num-each-iter 256  --rl-pool 20 --seed $seed
    done


### Camvid ###
for seed in 20 50 82 12 4560
    do
    python -u run.py --exp-name 'RALIS_camvid_train_seed'$seed --full-res --region-size 80 90 \
    --snapshot 'best_jaccard_val.pth' --al-algorithm 'ralis' \
    --ckpt-path $ckpt_path --data-path $data_path \
    --rl-episodes 100 --rl-buffer 600 --lr-dqn 0.001\
    --load-weights --exp-name-toload 'gta_pretraining_camvid' \
    --dataset 'camvid' --lr 0.001 --train-batch-size 32 --val-batch-size 4 --patience 10 \
    --input-size 224 224 --only-last-labeled --budget-labels 480  --num-each-iter 24  --rl-pool 20 --seed $seed
    done

