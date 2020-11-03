#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5
module load libra/1.1.2
module load java/1.8.0
module load maven/3.5.0

dataset=$1
featureset=$2
setting=$3
stacks=$4
joint=$5
fold=$6

if [ $joint == 'None' ]; then
    python3 main.py \
        --dataset $dataset \
        --featureset $featureset \
        --setting $setting \
        --stacks $stacks \
        --fold $fold
else
    python3 main.py \
        --dataset $dataset \
        --featureset $featureset \
        --setting $setting \
        --stacks $stacks \
        --joint $joint \
        --fold $fold
fi