#!/bin/bash

#SBATCH -J NGC188_6292/4965/5797_pwrlaw
#SBATCH -N 1  #Number of nodes on which to distribute the tasks evenly
#SBATCH -n 1 #Total number of tasks to run at one time
#SBATCH -o /work2/05392/kpenev/stampede2/open_cluster_circularization/sampling_output/NGC188_6292_4965_5797_slurm.out
#SBATCH --mail-user=kxp174430@utdallas.edu
#SBATCH --mail-type=all
#SBATCH -t 48:00:00
#SBATCH -p skx-normal

source /home1/05392/kpenev/module_load_order.txt

export PYTHONPATH=/home1/05392/kpenev/projects/git/poet/PythonPackage:/home1/05392/kpenev/projects/git/CircularizationDissipationConstraints/source:/home1/05392/kpenev/projects/git/general_purpose_python_modules
export PATH=$PATH:/home1/05392/kpenev/bin

for d in /home1/05392/kpenev/projects/git/poet/build/libs/*/shared/release; do
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(readlink -e $d)
    export LIBRARY_PATH=$LIBRARY_PATH:$(readlink -e $d)
done

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home1/05392/kpenev/lib
export LIBRARY_PATH=$LIBRARY_PATH:/home1/05392/kpenev/lib

SOURCE_DIR=/home1/05392/kpenev/projects/git/CircularizationDissipationConstraints/source
OUTPUT_ROOT=/work2/05392/kpenev/stampede2/open_cluster_circularization
OUTPUT_DIR=${OUTPUT_ROOT}/sampling_output

for PKM in 6292 4965 5797; do 
    python3 \
        ${SOURCE_DIR}/bayesian/sample_sb1.py NGC188_${PKM} \
        -c ${SOURCE_DIR}/sampling_config/open_cluster_powerlaw_nolock.cfg\
        --std-out-err-fname ${OUTPUT_DIR}'/%(system)s_%(now)s_%(pid)d.outerr'\
        --logging-fname ${OUTPUT_DIR}'/%(system)s_%(now)s_%(pid)d.log'\
        --samples-fname '/home1/05392/kpenev/%(system)s_%(sampling)s_powerlawlgQ_samples'\
        >\
        ${OUTPUT_DIR}/sample_NGC188_${PKM}_1.outerr\
        2>&1\
        &
done

wait
