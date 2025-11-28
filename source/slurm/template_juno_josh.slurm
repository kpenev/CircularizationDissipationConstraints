#!/bin/bash

#SBATCH -J @@CLUSTER@@_@@JOINED_SYSTEMS@@
#SBATCH -N 1  #Number of nodes on which to distribute the tasks evenly
#SBATCH -n 1 #Total number of tasks to run at one time
#SBATCH -o ~/work/@@CLUSTER@@/sampling_output/@@JOINED_SYSTEMS@@_slurm.out
#SBATCH --mail-user=jas180011@utdallas.edu
#SBATCH --mail-type=all
#SBATCH -t 48:00:00
#SBATCH -p @@QUEUE@@

trainsystem() {
    systemdir=~/work/@@CLUSTER@@/nn_data
    systemname=$1

    cd ~/work/code/CircularizationDissipationConstraints/source/bayesian
    python3 ml_clean.py ${systemname} ${systemdir}

    cd ~
    python3 trainsystem.py ${systemname} ${systemdir}
}

run_system() {
    source_dir=~/work/code/CircularizationDissipationConstraints/source
    config_dir=~/system_configs
    output_root=~/work/@@CLUSTER@@
    output_dir=${output_root}/sampling_output
    sysid=$1

    mkdir -p ${output_dir}/${sysid}
    mkdir -p ${output_root}/samples

    cd ~

    for (( i=0; i< 10; i++ )); do
        python3 \
            ${source_dir}/bayesian/@@SAMPLER@@${sysid} \
            -c ${config_dir}/'windemuth_template.cfg'\
            --track-final-eccentricity\
            >\
            ${output_dir}/sample_@@CLUSTER@@_${sysid}.outerr\
            2>&1
    done
}

cd ~
source engage.txt
source module_load_order.txt

for SYSID in @@SYSTEM_LIST@@; do 
    trainsystem $SYSID
done

for SYSID in @@SYSTEM_LIST@@; do 
    run_system $SYSID &
done

wait
