#!/bin/bash
markovscript=/home/chri6578/Documents/GG_SPP/markovspace/main.py
stgodescript=/home/chri6578/Documents/STGODE/run_stode.py
gramodescript=/home/chri6578/Documents/gramode/run_stode.py
lightctsscript=/home/chri6578/Documents/lightcts/multistep/flow/generic/train.py


# datasets_synth=(test01 test02 test03 test04 test05 test06 test07 test08 test09 test10)
# datasets_synth=(test11 test12 test13 test14 test15 test16 test17 test18 test19 test20)
# datasets_synth=(test21 test22 test23 test24 test25 test26 test27 test28 test29 test30)
datasets_synth=(test31 test32 test33 test34 test35 test36 test37 test38 test39 test40)
datasets_single=(tennis wikimath pedalme chickenpox)
datasets_multi=(pems03 pems04 pems07 pems08 pemsbay metrla)

logdir=/home/chri6578/Documents/GG_SPP/markovspace/logs/
logfile=experiment


q=1

# for dataset in "${datasets_synth[@]}"; do 
#     python $markovscript -d $dataset -q $q -K 1 -M 100 -r 0.7 --statemode S --samplemode mean -f $logfile --ind 0
# done

# for dataset in "${datasets_synth[@]}"; do 
#     python $markovscript -d $dataset -q $q -K 1 -M 100 -r 0.7 --statemode S --samplemode normal -f $logfile --ind 0
# done

for dataset in "${datasets_single[@]}"; do 
    # python $lightctsscript --data_key $dataset --seq_length 1 --logfile $logdir$logfile
    # python $lightctsscript --data_key $dataset --seq_length 1 --logfile $logdir$logfile --edge_weight I 
    # python $lightctsscript --data_key $dataset --seq_length 1 --logfile $logdir$logfile --edge_weight A 
    # python $stgodescript --remote --filename $dataset --num-gpu 0 --epochs 100 --batch-size 16 --train_ratio 0.6 --valid_ratio 0.1 --his-length $q --pred-length $q --sigma1 0.1 --sigma2 10 --thres1 0.6 --thres2 0.5 --lr 2e-3 --logfile $logdir$logfile
    # python $stgodescript --remote --filename $dataset --edge_weight I --num-gpu 0 --epochs 100 --batch-size 16 --train_ratio 0.6 --valid_ratio 0.1 --his-length $q --pred-length $q --sigma1 0.1 --sigma2 10 --thres1 0.6 --thres2 0.5 --lr 2e-3 --logfile $logdir$logfile
    # python $stgodescript --remote --filename $dataset --edge_weight A --num-gpu 0 --epochs 100 --batch-size 16 --train_ratio 0.6 --valid_ratio 0.1 --his-length $q --pred-length $q --sigma1 0.1 --sigma2 10 --thres1 0.6 --thres2 0.5 --lr 2e-3 --logfile $logdir$logfile
    python $gramodescript --remote --filename $dataset --num-gpu 0 --epochs 200 --batch-size 16 --train_ratio 0.6 --valid_ratio 0.1 --his-length $q --pred-length $q --sigma1 0.1 --sigma2 10 --thres1 0.6 --thres2 0.5 --lr 2e-4 --logfile $logdir$logfile
    python $gramodescript --remote --filename $dataset --edge_weight I --num-gpu 0 --epochs 200 --batch-size 16 --train_ratio 0.6 --valid_ratio 0.1 --his-length $q --pred-length $q --sigma1 0.1 --sigma2 10 --thres1 0.6 --thres2 0.5 --lr 2e-4 --logfile $logdir$logfile
    python $gramodescript --remote --filename $dataset --edge_weight A --num-gpu 0 --epochs 200 --batch-size 16 --train_ratio 0.6 --valid_ratio 0.1 --his-length $q --pred-length $q --sigma1 0.1 --sigma2 10 --thres1 0.6 --thres2 0.5 --lr 2e-4 --logfile $logdir$logfile
done



