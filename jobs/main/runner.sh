#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5
module load libra/1.1.2
module load java/1.8.0

dataset=$1
featureset=$2
setting=$3
stacks=$4
joint=$5

if [ $joint == 'None' ]; then
    python3 main_$dataset.py \
        --featureset $featureset \
        --setting $setting \
        --stacks $stacks
else
    python3 main_$dataset.py \
        --featureset $featureset \
        --setting $setting \
        --stacks $stacks \
        --joint $joint
fi