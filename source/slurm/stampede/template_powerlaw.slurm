#!/bin/bash

#SBATCH -J @@CLUSTER@@_@@SYS1@@/@@SYS2@@/@@SYS3@@_pwrlaw
#SBATCH -N 1  #Number of nodes on which to distribute the tasks evenly
#SBATCH -n 1 #Total number of tasks to run at one time
#SBATCH -o /work2/05392/kpenev/stampede2/open_cluster_circularization/sampling_output/@@CLUSTER@@_@@SYS1@@_@@SYS2@@_@@SYS3@@_slurm.out
#SBATCH --mail-user=kxp174430@utdallas.edu
#SBATCH --mail-type=all
#SBATCH -t 48:00:00
#SBATCH -p skx-normal

run_system() {
    source_dir=/home1/05392/kpenev/projects/git/CircularizationDissipationConstraints/source
    output_root=/work2/05392/kpenev/stampede2/open_cluster_circularization
    output_dir=${output_root}/sampling_output
    sysid=$1

    cd /home1/05392/kpenev/

    while true; do
        python3 \
            ${source_dir}/bayesian/sample_sb1.py @@CLUSTER@@_${sysid} \
            -c ${source_dir}/sampling_config/open_cluster_powerlaw_alllock.cfg\
            --std-out-err-fname ${output_dir}'/%(system)s_%(now)s_%(pid)d.outerr'\
            --logging-fname ${output_dir}'/%(system)s_%(now)s_%(pid)d.log'\
            --samples-fname '/home1/05392/kpenev/%(system)s_%(sampling)s_powerlawlgQ_samples'\
            >\
            ${output_dir}/sample_@@CLUSTER@@_${sysid}_1.outerr\
            2>&1
    done
}

source /home1/05392/kpenev/module_load_order.txt

export PYTHONPATH=/home1/05392/kpenev/projects/git/poet/PythonPackage:/home1/05392/kpenev/projects/git/CircularizationDissipationConstraints/source:/home1/05392/kpenev/projects/git/general_purpose_python_modules
export PATH=$PATH:/home1/05392/kpenev/bin

for d in /home1/05392/kpenev/projects/git/poet/build/libs/*/shared/release; do
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(readlink -e $d)
    export LIBRARY_PATH=$LIBRARY_PATH:$(readlink -e $d)
done

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home1/05392/kpenev/lib
export LIBRARY_PATH=$LIBRARY_PATH:/home1/05392/kpenev/lib

for SYSID in @@SYS1@@ @@SYS2@@ @@SYS3@@; do 
    run_system $SYSID &
done

wait
