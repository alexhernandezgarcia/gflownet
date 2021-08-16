#!/usr/bin/env bash

ckpt_path='/home/casanova/scratch/ckpt_seg'
data_path='/home/casanova/scratch/'

for budget in 1920 3840 7680 11520 19200 30720
    do
    for seed in 20 50 234 77 12
        do
        python -u run.py --exp-name 'RALIS_cs_test_seed'$seed \
        --al-algorithm 'ralis' --checkpointer\
        --ckpt-path $ckpt_path --data-path $data_path \
        --load-weights --exp-name-toload 'cityscapes_pretrained_dt' \
        --dataset 'cityscapes' --lr 0.0001 --train-batch-size 16 --val-batch-size 1 --patience 60 \
        --input-size 256 512 --only-last-labeled --budget-labels $budget  --num-each-iter 256  --rl-pool 100 --seed $seed \
        --train --test --final-test --exp-name-toload-rl 'RALIS_cs_train_seed'$seed
        done
    done



### Camvid ###
for budget in 480 720 960 1200 1440 1920
    do
    for seed in 20 50 82 12 4560
        do
        python -u run.py --exp-name 'RALIS_camvid_test_seed'$seed \
        --al-algorithm 'ralis' --checkpointer --region-size 80 90 \
        --ckpt-path $ckpt_path --data-path $data_path \
        --load-weights --exp-name-toload 'camvid_pretrained_dt' \
        --dataset 'camvid' --lr 0.001 --train-batch-size 32 --val-batch-size 4 --patience 150 \
        --input-size 224 224 --only-last-labeled --budget-labels $budget  --num-each-iter 24  --rl-pool 10 --seed $seed \
        --train --test --final-test --exp-name-toload-rl 'RALIS_camvid_train_seed'$seed
        done
    done
