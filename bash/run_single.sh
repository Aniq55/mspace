#!/bin/bash
script=/home/chri6578/Documents/GG_SPP/markovspace/main.py
datasets_single=(tennis wikimath pedalme chickenpox)
datasets_multi=(pems03 pems04 pems07 pems08 pemsbay metrla)

for dataset in "${datasets_single[@]}"; do 
    python $script -d $dataset -q 1 -K 1 -M 20 -r 0.7 --statemode S --samplemode mean -f single
    python $script -d $dataset -q 1 -K 1 -M 20 -r 0.7 --statemode S --samplemode normal -f single
    python $script -d $dataset -q 1 -K 1 -M 20 -r 0.7 --statemode S --samplemode mean -f single -I 1 
done

# for dataset in "${datasets_single[@]}"; do 
#     for train_ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#         python $script -d $dataset -q 1 -K 1 -M 20 -r $train_ratio --statemode S --samplemode mean -f $dataset
#     done
# done

# for dataset in "${datasets_single[@]}"; do 
#     for train_ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#         python $script -d $dataset -q 1 -K 1 -M 20 -r $train_ratio --statemode S --samplemode normal -f $dataset
#     done
# done