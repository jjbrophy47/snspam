mem=$1
time=$2
partition=$3

sbatch --mem=${mem}G \
       --time=$time \
       --partition=$partition \
       --job-name=soundcloud \
       --output=jobs/logs/soundcloud/ \
       --error=jobs/errors/soundcloud/ \
       jobs/soundcloud/runner.sh
