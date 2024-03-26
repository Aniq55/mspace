#!/bin/bash
script=/home/chri6578/Documents/GG_SPP/markovspace/main.py
datasets_single=(tennis wikimath pedalme chickenpox)
datasets_multi=(pems03 pems04 pems07 pems08 pemsbay metrla)

for dataset in "${datasets_multi[@]}"; do 
    python $script -d $dataset -q 12 -K 1 -M 20 -r 0.8 --statemode S --samplemode mean -f indep -I True
done

for dataset in "${datasets_multi[@]}"; do 
    python $script -d $dataset -q 12 -K 1 -M 20 -r 0.8 --statemode S --samplemode normal -f indep -I True
done