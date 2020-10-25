mem=$1
time=$2
partition=$3

sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --job-name=preprocess \
       --output=jobs/logs/soundcloud/preprocess \
       --error=jobs/errors/soundcloud/preprocess \
       jobs/soundcloud/runner.sh
