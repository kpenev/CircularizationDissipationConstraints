#!/bin/bash

#SBATCH -J @@CLUSTER@@_@@JOINED_SYSTEMS@@
#SBATCH -N 1  #Number of nodes on which to distribute the tasks evenly
#SBATCH -n 128 #Total number of tasks to run at one time
#SBATCH -o /work/05392/kpenev/ls6/open_cluster_circularization/sampling_output/@@CLUSTER@@_@@JOINED_SYSTEMS@@_slurm.out
#SBATCH --mail-user=kxp174430@utdallas.edu
#SBATCH --mail-type=all
#SBATCH -t 48:00:00
#SBATCH -p normal

run_system() {
    source_dir=/home1/05392/kpenev/projects/git/CircularizationDissipationConstraints/source
    output_root=/work/05392/kpenev/ls6/open_cluster_circularization/@@CLUSTER@@
    output_dir=${output_root}/sampling_output
    sysid=$1

    mkdir -p ${output_dir}/${sysid}
    mkdir -p ${output_root}/samples

    cd /home1/05392/kpenev/

    while true; do
        python3 \
            ${source_dir}/bayesian/@@SAMPLER@@${sysid} \
            -c ${source_dir}/sampling_config/@@CONFIG_FNAME@@\
            --std-out-err-fname ${output_dir}/${sysid}'/%(system)s_%(now)s_%(pid)d.outerr'\
            --logging-fname ${output_dir}/${sysid}'/%(system)s_%(now)s_%(pid)d.log'\
            --samples-fname ${output_root}'/samples/%(system)s_%(sampling)s_powerlawlgQ_samples'\
            >\
            ${output_dir}/sample_@@CLUSTER@@_${sysid}.outerr\
            2>&1
    done
}

mkdir -p /work/05392/kpenev/ls6/open_cluster_circularization/sampling_output/

source /home1/05392/kpenev/module_load_order.txt

export PYTHONPATH=/home1/05392/kpenev/projects/git/poet/PythonPackage:/home1/05392/kpenev/projects/git/CircularizationDissipationConstraints/source:/home1/05392/kpenev/projects/git
export PATH=$PATH:/home1/05392/kpenev/bin

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home1/05392/kpenev/lib
export LIBRARY_PATH=$LIBRARY_PATH:/home1/05392/kpenev/lib

for SYSID in @@SYSTEM_LIST@@; do 
    run_system $SYSID &
done

wait
