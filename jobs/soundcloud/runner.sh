#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=uoml
module load python3/3.7.5

cd snspam_data/soundcloud/scripts/

python3 compute_features.py

python3 link_relation.py

python3 text_relation.py

python3 user_relation.py
