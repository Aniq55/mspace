script=/home/chri6578/Documents/mspace/main_mv.py

python $script -d metrla -q 12 -K 1 -M 1000 -r 0.8 --statemode T --samplemode mean -f multi --period 2016
python $script -d pemsbay -q 12 -K 1 -M 1000 -r 0.8 --statemode T --samplemode mean -f multi --period 2016

python $script -d metrla -q 12 -K 1 -M 1000 -r 0.8 --statemode S --samplemode mean -f multi
python $script -d pemsbay -q 12 -K 1 -M 1000 -r 0.8 --statemode S --samplemode mean -f multi