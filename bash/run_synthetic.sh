#!/bin/bash
script=/home/chri6578/Documents/GG_SPP/markovspace/main.py
datasets_synth=(test01)

for dataset in "${datasets_synth[@]}"; do 
    for q in 12; do
        python $script -d $dataset -q $q -K 1 -M 100 -r 0.7 --statemode S --samplemode mean -f synth --ind 0
        python $script -d $dataset -q $q -K 1 -M 100 -r 0.7 --statemode S --samplemode normal -f synth --ind 0
        python $script -d $dataset -q $q -K 1 -M 100 -r 0.7 --statemode S --samplemode mean -f synth --ind 1
    done
done


