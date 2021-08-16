#!/usr/bin/env bash

ckpt_path='/home/casanova/scratch/ckpt_seg'
data_path='/home/casanova/scratch/'

for al_algorithm in 'random'
    do
    for budget in 1920 3840 7680 11520 19200 30720
        do
        for seed in 20 50 234 4560 12
            do
            python run.py --exp-name 'baseline_cityscapes_'$al_algorithm'_budget_'$budget'_seed'$seed --seed $seed  --checkpointer \
            --ckpt-path $ckpt_path --data-path $data_path \
            --load-weights --exp-name-toload 'cityscapes_pretrained_dt' \
            --input-size 256 512 --only-last-labeled --dataset 'cityscapes' \
             --budget-labels $budget --num-each-iter 256 --al-algorithm $al_algorithm --rl-pool 500 --train --test --final-test
            done
        done
    done

for al_algorithm in 'entropy'
    do
    for budget in 1920 3840 7680 11520 19200 30720
        do
        for seed in 20 50 234 4560 12
            do
            python run.py --exp-name 'baseline_cityscapes_'$al_algorithm'_budget_'$budget'_seed'$seed --seed $seed  --checkpointer \
            --ckpt-path $ckpt_path --data-path $data_path \
            --load-weights --exp-name-toload 'cityscapes_pretrained_dt' \
            --input-size 256 512 --only-last-labeled --dataset 'cityscapes' \
             --budget-labels $budget --num-each-iter 256 --al-algorithm $al_algorithm --rl-pool 200 --train --test --final-test
            done
        done
    done

for al_algorithm in 'bald'
    do
    for budget in 1920 3840 7680 11520 19200 30720
        do
        for seed in 20 50 234 4560 12
            do
            python run.py --exp-name 'baseline_cityscapes_'$al_algorithm'_budget_'$budget'_seed'$seed --seed $seed  --checkpointer \
            --ckpt-path $ckpt_path --data-path $data_path \
            --load-weights --exp-name-toload 'cityscapes_pretrained_dt' \
            --input-size 256 512 --only-last-labeled --dataset 'cityscapes' \
             --budget-labels $budget --num-each-iter 256 --al-algorithm $al_algorithm --rl-pool 200 --train --test --final-test
            done
        done
    done



#### Camvid ####

for al_algorithm in 'random'
    do
    for budget in 480 720 960 1200 1440 1920
        do
        for seed in 20 50 234 4560 12
            do
            python run.py --exp-name 'baseline_camvid_'$al_algorithm'_budget_'$budget'_seed'$seed --seed $seed  --checkpointer \
            --ckpt-path $ckpt_path --data-path $data_path \
            --load-weights --exp-name-toload 'camvid_pretrained_dt' \
            --input-size 224 224 --only-last-labeled --dataset 'camvid' --lr 0.001 --train-batch-size 32 --val-batch-size 4 \
            --patience 150 --region-size 80 90 \
             --budget-labels $budget --num-each-iter 24 --al-algorithm $al_algorithm --rl-pool 50 --train --test --final-test
            done
        done
    done

for al_algorithm in 'entropy' 'bald'
    do
    for budget in 480 720 960 1200 1440 1920
        do
        for seed in 20 50 234 4560 12
            do
            python run.py --exp-name 'baseline_camvid_'$al_algorithm'_budget_'$budget'_seed'$seed --seed $seed  --checkpointer \
            --ckpt-path $ckpt_path --data-path $data_path \
            --load-weights --exp-name-toload 'camvid_pretrained_dt' \
            --input-size 224 224 --only-last-labeled --dataset 'camvid' --lr 0.001 --train-batch-size 32 --val-batch-size 4 \
            --patience 150 --region-size 80 90 \
             --budget-labels $budget --num-each-iter 24 --al-algorithm $al_algorithm --rl-pool 10 --train --test --final-test
            done
        done
    done

