dataset=$1
mem=$2
time=$3
partition=$4

featureset_list=('limited' 'full')
setting_list=('inductive' 'inductive+transductive')
stacks_list=(0 1 2)
joint_list=('None' 'mrf' 'psl')

featureset_list=('limited')
setting_list=('inductive')
stacks_list=(0)
joint_list=('psl')
fold_list=(0 1 2 3 4 5 6 7 8 9)

for fold in ${fold_list[@]}; do
    for featureset in ${featureset_list[@]}; do
        for setting in ${setting_list[@]}; do
            for stacks in ${stacks_list[@]}; do
                for joint in ${joint_list[@]}; do

                    sbatch --mem=${mem}G \
                           --time=$time \
                           --partition=$partition \
                           --job-name=$dataset \
                           --output=jobs/logs/main/$dataset \
                           --error=jobs/errors/main/$dataset \
                           jobs/main/runner.sh $dataset \
                           $featureset $setting $stacks $joint $fold
                done
            done
        done
    done
done
