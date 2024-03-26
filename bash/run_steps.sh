#!/bin/bash
script=/home/chri6578/Documents/GG_SPP/markovspace/main.py
datasets_multi=(pems03 pems04 pems07 pems08)

for dataset in "${datasets_multi[@]}"; do 
    for q in 1 12 24 36 48; do
        python $script -d $dataset -q $q -K 1 -M 20 -r 0.8 --statemode S --samplemode mean -f $dataset
    done
done


for dataset in "${datasets_multi[@]}"; do 
    for q in 1 12 24 36 48; do
        python $script -d $dataset -q $q -K 1 -M 20 -r 0.8 --statemode S --samplemode normal -f $dataset
    done
done

for dataset in "${datasets_multi[@]}"; do 
    for q in 1 12 24 36 48; do
        python $script -d $dataset -q $q -K 1 -M 20 -r 0.8 --statemode T --samplemode mean -f $dataset --period 2016
    done
done


for dataset in "${datasets_multi[@]}"; do 
    for q in 1 12 24 36 48; do
        python $script -d $dataset -q $q -K 1 -M 20 -r 0.8 --statemode T --samplemode normal -f $dataset --period 2016
    done
done