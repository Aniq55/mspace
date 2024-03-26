#!/bin/bash
script=/home/chri6578/Documents/GG_SPP/markovspace/kalman.py
script0=/home/chri6578/Documents/GG_SPP/markovspace/kalman0.py

datasets_single=(tennis wikimath pedalme chickenpox)
datasets_multi=(pems03 pems04 pems07 pems08 pemsbay metrla)


datasets_synth=(test01 test02 test03 test04 test05 test06 test07 test08 test09 test10)
datasets_synth2=(test11 test12 test13 test14 test15 test16 test17 test18 test19 test20)
datasets_synth3=(test21 test22 test23 test24 test25 test26 test27 test28 test29 test30)
datasets_synth4=(test31 test32 test33 test34 test35 test36 test37 test38 test39 test40)

# What is the difference between kalman.py and kalman0.py?
# In kalman0, each node has their personal kalman filter
# kalman is graphical (all nodes together), while kalman0 is non-graphical


#e
# for dataset in "${datasets_single[@]}"; do 
#     for q in 1; do
#         python $script --dataset $dataset --train_ratio 0.7 --latent_dim 0.8 --shock --n_iters 50 --q $q
#     done
# done
# # x
# for dataset in "${datasets_single[@]}"; do 
#     for q in 1; do
#         python $script --dataset $dataset --train_ratio 0.7 --latent_dim 0.8 --n_iters 50 --q $q
#     done
# done
# eI
for dataset in "${datasets_single[@]}"; do 
    for q in 1; do
        python $script0 --dataset $dataset --train_ratio 0.9 --shock --n_iters 10 --q $q
    done
done
# # #xI
for dataset in "${datasets_single[@]}"; do 
    for q in 1; do
        python $script0 --dataset $dataset  --train_ratio 0.9  --n_iters 10 --q $q
    done
done

#e
# for dataset in "${datasets_multi[@]}"; do 
#     for q in 12; do
#         python $script --dataset $dataset  --train_ratio 0.7 --latent_dim 0.8 --shock --n_iters 10 --q $q
#     done
# done
# #x
# for dataset in "${datasets_multi[@]}"; do 
#     for q in 12; do
#         python $script --dataset $dataset  --train_ratio 0.7 --latent_dim 0.8 --n_iters 10 --q $q
#     done
# done
# #eI
# for dataset in "${datasets_multi[@]}"; do 
#     for q in 12; do
#         python $script0 --dataset $dataset  --train_ratio 0.7 --shock --n_iters 10 --q $q
#     done
# done
# # #xI
# for dataset in "${datasets_multi[@]}"; do 
#     for q in 12; do
#         python $script0 --dataset $dataset --train_ratio 0.7  --n_iters 10 --q $q
#     done
# done

# for dataset in "${datasets_synth[@]}"; do 
#     for q in 12; do
#         python $script --dataset $dataset --train_ratio 0.7  --n_iters 100 --q $q
#     done
# done

# for dataset in "${datasets_synth[@]}"; do 
#     for q in 12; do
#         python $script0 --dataset $dataset --train_ratio 0.7  --n_iters 100 --q $q
#     done
# done

# for dataset in "${datasets_synth4[@]}"; do 
#     for q in 12; do
#         python $script --dataset $dataset --train_ratio 0.7  --n_iters 100 --q $q
#     done
# done

# for dataset in "${datasets_synth4[@]}"; do 
#     for q in 12; do
#         python $script0 --dataset $dataset --train_ratio 0.7  --n_iters 100 --q $q
#     done
# done

# for dataset in "${datasets_synth4[@]}"; do 
#     for q in 12; do
#         python $script --dataset $dataset --train_ratio 0.7  --n_iters 100 --q $q --shock
#     done
# done

# for dataset in "${datasets_synth4[@]}"; do 
#     for q in 12; do
#         python $script0 --dataset $dataset --train_ratio 0.7  --n_iters 100 --q $q --shock
#     done
# done